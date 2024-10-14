import os
import sys
import logging
from tqdm import tqdm
import argparse
import torch.nn.functional as F
from data.data_loaders import dist_flow_map_optimized,dist_flow_augment_optimized

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='synapse')
parser.add_argument('--exp', type=str)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--split_labeled', type=str, default='labeled_20p')
parser.add_argument('--split_unlabeled', type=str, default='unlabeled_20p')
parser.add_argument('--split_eval', type=str, default='eval')
parser.add_argument('--max_epoch', type=int, default=500)
parser.add_argument('--use_stac', type=int, default=1)
parser.add_argument('--cps_loss', type=str, default='wce')
parser.add_argument('--sup_loss', type=str, default='w_ce+dice')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--base_lr', type=float, default=0.001)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--cps_w', type=float, default=1)
parser.add_argument('--alpha', type=float, default=2)
parser.add_argument('--bata', type=float, default=-2)
parser.add_argument('--max_extent', type=float, default=3)
parser.add_argument('--cps_rampup', action='store_true', default=True) # <--
parser.add_argument('--consistency_rampup', type=float, default=None)
parser.add_argument('--stac_start_epoch', type=int, default=40)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from models.vnet import VNet,VNetDist
from utils import EMA, maybe_mkdir, get_lr, fetch_data, seed_worker, poly_lr, print_func, kaiming_normal_init_weight
from utils.loss import DC_and_CE_loss, RobustCrossEntropyLoss, SoftDiceLoss, CrossEntropyLoss2d
from data.transforms import RandomCrop, CenterCrop, ToTensor, RandomFlip_LR, RandomFlip_UD
from data.data_loaders import Synapse_AMOS
from utils.config import Config

config = Config(args.task)



def sigmoid_rampup(current, rampup_length):
    '''Exponential rampup from https://arxiv.org/abs/1610.02242'''
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(epoch):
    if args.cps_rampup:
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        if args.consistency_rampup is None:
            args.consistency_rampup = args.max_epoch
        return args.cps_w * sigmoid_rampup(epoch, args.consistency_rampup)
    else:
        return args.cps_w



def make_loss_function(name, weight=None):
    if name == 'ce':
        return RobustCrossEntropyLoss()
    elif name == 'wce':
        return RobustCrossEntropyLoss(weight=weight)
    elif name == 'ce+dice':
        return DC_and_CE_loss()
    elif name == 'wce+dice':
        return DC_and_CE_loss(w_ce=weight)
    elif name == 'w_ce+dice':
        return DC_and_CE_loss(w_dc=weight, w_ce=weight)
    else:
        raise ValueError(name)


def make_loader(split, dst_cls=Synapse_AMOS, repeat=None, is_training=True, unlabeled=False,use_stac=0):
    if is_training:
        dst = dst_cls(
            task=args.task,
            split=split,
            repeat=repeat,
            unlabeled=unlabeled,
            num_cls=config.num_cls,
            use_stac=use_stac,
            length_weight=args.length_weight,
            transform=transforms.Compose([
                RandomCrop(config.patch_size),
                RandomFlip_LR(),
                RandomFlip_UD(),
                ToTensor()
            ])
        )
        return DataLoader(
            dst,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker
        )
    else:
        dst = dst_cls(
            task=args.task,
            split=split,
            is_val=True,
            num_cls=config.num_cls,
            transform=transforms.Compose([
                CenterCrop(config.patch_size),
                ToTensor()
            ])
        )
        return DataLoader(dst, pin_memory=True)


def make_model_all():
    if args.use_stac:
        model = VNetDist(
            n_channels=config.num_channels,
            n_classes=config.num_cls,
            n_filters=config.n_filters,
            normalization='batchnorm',
            has_dropout=True
        ).cuda()
    else:
        model = VNet(
            n_channels=config.num_channels,
            n_classes=config.num_cls,
            n_filters=config.n_filters,
            normalization='batchnorm',
            has_dropout=True
        ).cuda()

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=3e-5,
        nesterov=True
    )

    return model, optimizer




class DistDW:
    def __init__(self, num_cls, do_bg=False, momentum=0.95):
        self.num_cls = num_cls
        self.do_bg = do_bg
        self.momentum = momentum

    def _cal_weights(self, num_each_class):
        num_each_class = torch.FloatTensor(num_each_class).cuda()
        P = (num_each_class.max()+1e-8) / (num_each_class+1e-8)
        P_log = torch.log(P)
        weight = P_log / P_log.max()
        return weight

    def init_weights(self, labeled_dataset):
        if labeled_dataset.unlabeled:
            raise ValueError
        num_each_class = np.zeros(self.num_cls)
        for data_id in labeled_dataset.ids_list:
            _, _, label = labeled_dataset._get_data(data_id)
            label = label.reshape(-1)
            tmp, _ = np.histogram(label, range(self.num_cls + 1))
            num_each_class += tmp
        weights = self._cal_weights(num_each_class)
        self.weights = weights * self.num_cls
        return self.weights.data.cpu().numpy()

    def get_ema_weights(self, pseudo_label):
        pseudo_label = torch.argmax(pseudo_label.detach(), dim=1, keepdim=True).long()
        label_numpy = pseudo_label.data.cpu().numpy()
        num_each_class = np.zeros(self.num_cls)
        for i in range(label_numpy.shape[0]):
            label = label_numpy[i].reshape(-1)
            tmp, _ = np.histogram(label, range(self.num_cls + 1))
            num_each_class += tmp

        cur_weights = self._cal_weights(num_each_class) * self.num_cls
        self.weights = EMA(cur_weights, self.weights, momentum=self.momentum)
        return self.weights



class DiffDW:
    def __init__(self, num_cls, accumulate_iters=20):
        self.last_dice = torch.zeros(num_cls).float().cuda() + 1e-8
        self.dice_func = SoftDiceLoss(smooth=1e-8, do_bg=True)
        self.cls_learn = torch.zeros(num_cls).float().cuda()
        self.cls_unlearn = torch.zeros(num_cls).float().cuda()
        self.num_cls = num_cls
        self.dice_weight = torch.ones(num_cls).float().cuda()
        self.accumulate_iters = accumulate_iters

    def init_weights(self):
        weights = np.ones(config.num_cls) * self.num_cls
        self.weights = torch.FloatTensor(weights).cuda()
        return weights

    def cal_weights(self, pred,  label):
        x_onehot = torch.zeros(pred.shape).cuda()
        output = torch.argmax(pred, dim=1, keepdim=True).long()
        x_onehot.scatter_(1, output, 1)
        y_onehot = torch.zeros(pred.shape).cuda()
        y_onehot.scatter_(1, label, 1)
        cur_dice = self.dice_func(x_onehot, y_onehot, is_training=False)
        delta_dice = cur_dice - self.last_dice
        cur_cls_learn = torch.where(delta_dice>0, delta_dice, 0) * torch.log(cur_dice / self.last_dice)
        cur_cls_unlearn = torch.where(delta_dice<=0, delta_dice, 0) * torch.log(cur_dice / self.last_dice)
        self.last_dice = cur_dice
        self.cls_learn = EMA(cur_cls_learn, self.cls_learn, momentum=(self.accumulate_iters-1)/self.accumulate_iters)
        self.cls_unlearn = EMA(cur_cls_unlearn, self.cls_unlearn, momentum=(self.accumulate_iters-1)/self.accumulate_iters)
        cur_diff = (self.cls_unlearn + 1e-8) / (self.cls_learn + 1e-8)
        cur_diff = torch.pow(cur_diff, 1/5)
        self.dice_weight = EMA(1. - cur_dice, self.dice_weight, momentum=(self.accumulate_iters-1)/self.accumulate_iters)
        weights = cur_diff * self.dice_weight
        weights = weights / weights.max()
        return weights * self.num_cls





if __name__ == '__main__':
    import random

    SEED=args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    snapshot_path = f'./logs/{args.exp}_stac_{str(args.use_stac)}_length_{str(args.alpha)}_{str(args.bata)}_{str(args.max_extent)}_seed_{str(args.seed)}/'

    maybe_mkdir(snapshot_path)
    maybe_mkdir(os.path.join(snapshot_path, 'ckpts'))

    # make logger
    writer = SummaryWriter(os.path.join(snapshot_path, 'tensorboard'))
    logging.basicConfig(
        filename=os.path.join(snapshot_path, 'train.log'),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    ce_perpixel_loss = torch.nn.CrossEntropyLoss(reduce=False)
    # make data loader
    args.length_weight = (args.alpha, args.bata, args.max_extent)
    unlabeled_loader = make_loader(args.split_unlabeled, unlabeled=True, use_stac=args.use_stac)
    labeled_loader = make_loader(args.split_labeled, repeat=len(unlabeled_loader.dataset), use_stac=args.use_stac)
    eval_loader = make_loader(args.split_eval, is_training=False)



    logging.info(f'{len(labeled_loader)} itertations per epoch (labeled)')
    logging.info(f'{len(unlabeled_loader)} itertations per epoch (unlabeled)')

    # make model, optimizer, and lr scheduler
    model_A, optimizer_A = make_model_all()
    model_B, optimizer_B = make_model_all()
    model_A = kaiming_normal_init_weight(model_A)
    model_B = kaiming_normal_init_weight(model_B)


    # make loss function
    diffdw = DiffDW(config.num_cls, accumulate_iters=50)
    distdw = DistDW(config.num_cls, momentum=0.99)

    weight_A = diffdw.init_weights()
    weight_B = distdw.init_weights(labeled_loader.dataset)

    loss_func_A     = make_loss_function(args.sup_loss, weight_A)
    loss_func_B     = make_loss_function(args.sup_loss, weight_B)
    cps_loss_func_A = make_loss_function(args.cps_loss, weight_A)
    cps_loss_func_B = make_loss_function(args.cps_loss, weight_B)


    amp_grad_scaler = GradScaler()

    cps_w = get_current_consistency_weight(0)
    best_eval = 0.0
    best_epoch = 0
    for epoch_num in range(args.max_epoch + 1):
        loss_list = []
        loss_cps_list = []
        loss_sup_list = []

        model_A.train()
        model_B.train()
        for batch_l, batch_u in tqdm(zip(labeled_loader, unlabeled_loader)):
            optimizer_A.zero_grad()
            optimizer_B.zero_grad()

            image_l, label_l = fetch_data(batch_l)
            image_u = fetch_data(batch_u, labeled=False)


            image_l_id = batch_l['data_id']
            image_u_id = batch_u['data_id']
            image = torch.cat([image_l, image_u], dim=0)
            tmp_bs = image.shape[0] // 2


            with autocast():
                if args.use_stac:
                    output_A, output_A_dist = model_A(image)
                    output_B, output_B_dist = model_B(image)

                    output_A_dist = torch.tanh(output_A_dist)
                    output_B_dist = torch.tanh(output_B_dist)

                    distance_gt = batch_l['distance_gt'].cuda()
                    
                    output_A_dist_l, output_A_dist_u = output_A_dist[:tmp_bs, ...], output_A_dist[tmp_bs:, ...]
                    output_B_dist_l, output_B_dist_u = output_B_dist[:tmp_bs, ...], output_B_dist[tmp_bs:, ...]
                    loss_dist_A_l = torch.norm(output_A_dist_l - distance_gt, 1)/torch.numel(output_A_dist_l) + F.mse_loss(output_A_dist_l, distance_gt)
                    loss_dist_B_l = torch.norm(output_B_dist_l - distance_gt, 1) / torch.numel(output_B_dist_l) + F.mse_loss(output_B_dist_l, distance_gt)


                    loss_dist_A_u = torch.norm(output_A_dist_u - output_B_dist_u.detach(), 1)/torch.numel(output_A_dist_u) + F.mse_loss(output_A_dist_u, output_B_dist_u.detach())
                    loss_dist_B_u = torch.norm(output_B_dist_u - output_A_dist_u.detach(), 1)/torch.numel(output_B_dist_u) + F.mse_loss(output_B_dist_u, output_A_dist_u.detach())
                else:
                    output_A = model_A(image)
                    output_B = model_B(image)

                del image

                # sup (ce + dice)
                output_A_l, output_A_u = output_A[:tmp_bs, ...], output_A[tmp_bs:, ...]
                output_B_l, output_B_u = output_B[:tmp_bs, ...], output_B[tmp_bs:, ...]


                # cps (ce only)
                max_A = torch.argmax(output_A_u.detach(), dim=1, keepdim=True).long()
                max_B = torch.argmax(output_B_u.detach(), dim=1, keepdim=True).long()

                weight_A = diffdw.cal_weights(output_A_l.detach(), label_l.detach())
                weight_B = distdw.get_ema_weights(output_B_u.detach())



                loss_func_A.update_weight(weight_A)
                loss_func_B.update_weight(weight_B)
                cps_loss_func_A.update_weight(weight_A)
                cps_loss_func_B.update_weight(weight_B)

                loss_sup = loss_func_A(output_A_l, label_l) + loss_func_B(output_B_l, label_l)
                loss_cps = cps_loss_func_A(output_A_u, max_B) + cps_loss_func_B(output_B_u, max_A)
                
                if args.use_stac:
                    loss_sup += (loss_dist_A_l + loss_dist_B_l) * 1
                    loss_cps += (loss_dist_A_u + loss_dist_B_u) * 1

                loss = loss_sup + cps_w * loss_cps


                if args.use_stac and epoch_num >= args.stac_start_epoch:
                    image_u_copy = image_u.detach().cpu().numpy()
                    output_A_dist_u_copy = output_A_dist_u.detach().cpu().numpy()
                    output_B_dist_u_copy = output_B_dist_u.detach().cpu().numpy()
                    output_dist_copy  = (output_A_dist_u_copy + output_B_dist_u_copy) / 2 
                    max_A_copy = max_A.detach().cpu().numpy()
                    max_B_copy = max_B.detach().cpu().numpy()
                    max_copy =  np.round((max_A_copy + max_B_copy) / 2)
                
                
                    image_u_aug = np.zeros_like(image_u_copy)
                    label_u_aug = np.zeros_like(max_copy)
                
                    for batch_index in range(output_dist_copy.shape[0]):
                        output_dist_u_now = output_dist_copy[batch_index,0, ...]
                        grad_flow_u = dist_flow_map_optimized(output_dist_u_now)[0]

                        image_u_now = image_u_copy[batch_index, 0, ...]
                        label_u_now = max_copy[batch_index, 0, ...]

                        image_u_augnow, label_u_augnow = dist_flow_augment_optimized(image_u_now, label_u_now, output_dist_u_now, grad_flow_u)

                        image_u_aug[batch_index,0, ...] = image_u_augnow
                        label_u_aug[batch_index,0, ...] = label_u_augnow
                    image_u_aug = torch.from_numpy(image_u_aug).cuda()
                    label_u_aug = torch.from_numpy(label_u_aug).cuda()

                    output_A_aug, _ = model_A(image_u_aug)
                    output_B_aug, _ = model_B(image_u_aug)

                    loss_u_aug = loss_func_A(output_A_aug, label_u_aug) + loss_func_B(output_B_aug, label_u_aug) 
                
                    loss = loss + cps_w * loss_u_aug * 1

                amp_grad_scaler.scale(loss).backward()
                amp_grad_scaler.step(optimizer_A)
                amp_grad_scaler.step(optimizer_B)
                amp_grad_scaler.update()


            loss_list.append(loss.item())
            loss_sup_list.append(loss_sup.item())
            loss_cps_list.append(loss_cps.item())

        writer.add_scalar('lr', get_lr(optimizer_A), epoch_num)
        writer.add_scalar('cps_w', cps_w, epoch_num)
        writer.add_scalar('loss/loss', np.mean(loss_list), epoch_num)
        writer.add_scalar('loss/sup', np.mean(loss_sup_list), epoch_num)
        writer.add_scalar('loss/cps', np.mean(loss_cps_list), epoch_num)

        writer.add_scalars('class_weights/A', dict(zip([str(i) for i in range(config.num_cls)] ,print_func(weight_A))), epoch_num)
        writer.add_scalars('class_weights/B', dict(zip([str(i) for i in range(config.num_cls)] ,print_func(weight_B))), epoch_num)
        logging.info(f'epoch {epoch_num} : loss : {np.mean(loss_list)}')

        logging.info(f"     Class Weights A: {print_func(weight_A)}, lr: {get_lr(optimizer_A)}")
        logging.info(f"     Class Weights B: {print_func(weight_B)}")

        optimizer_A.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)
        optimizer_B.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)
        cps_w = get_current_consistency_weight(epoch_num)

        if epoch_num % 10 == 0:

            # ''' ===== evaluation
            dice_list = [[] for _ in range(config.num_cls-1)]
            model_A.eval()
            model_B.eval()
            dice_func = SoftDiceLoss(smooth=1e-8)
            for batch in tqdm(eval_loader):
                with torch.no_grad():
                    image, gt = fetch_data(batch)

                    output = (model_A(image) + model_B(image))/2.0
                    # output = model_B(image)
                    del image

                    shp = output.shape
                    gt = gt.long()
                    y_onehot = torch.zeros(shp).cuda()
                    y_onehot.scatter_(1, gt, 1)

                    x_onehot = torch.zeros(shp).cuda()
                    output = torch.argmax(output, dim=1, keepdim=True).long()
                    x_onehot.scatter_(1, output, 1)


                    dice = dice_func(x_onehot, y_onehot, is_training=False)
                    dice = dice.data.cpu().numpy()
                    for i, d in enumerate(dice):
                        dice_list[i].append(d)

            dice_mean = []
            for dice in dice_list:
                dice_mean.append(np.mean(dice))
            logging.info(f'evaluation epoch {epoch_num}, dice: {np.mean(dice_mean)}, {dice_mean}')
            # '''
            if np.mean(dice_mean) > best_eval:
                best_eval = np.mean(dice_mean)
                best_epoch = epoch_num
                save_path = os.path.join(snapshot_path, f'ckpts/best_model.pth')
                torch.save({
                    'A': model_A.state_dict(),
                    'B': model_B.state_dict()
                }, save_path)
                logging.info(f'saving best model to {save_path}')
            logging.info(f'\t best eval dice is {best_eval} in epoch {best_epoch}')
            if epoch_num - best_epoch == config.early_stop_patience:
                logging.info(f'Early stop.')
                break

    writer.close()
