import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='synapse')
parser.add_argument('--exp', type=str, default='dist')
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--speed', type=int, default=0)
parser.add_argument('--gpu', type=str,  default='0')
parser.add_argument('--cps', type=str, default=None)
parser.add_argument('--use_stac', type=int, default=0)
parser.add_argument('--cps_w', type=float, default=1)
parser.add_argument('--alpha', type=float, default=2)
parser.add_argument('--bata', type=float, default=-2)
parser.add_argument('--max_extent', type=float, default=3)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
from models.vnet import VNet,VNetDist
from utils import test_all_case, read_list, maybe_mkdir, test_all_case_AB
from utils.config import Config
config = Config(args.task)

if __name__ == '__main__':
    stride_dict = {
        0: (32, 16),
        1: (64, 16),
        2: (128, 32),
    }
    stride = stride_dict[args.speed]

    # if 'dist' not in args.exp:
    #     snapshot_path = f'./logs/{args.exp}/'
    #     test_save_path = f'./logs/{args.exp}/predictions_{args.cps}/'
    # else:
    snapshot_path = f'./logs/{args.exp}_stac_{str(args.use_stac)}_length_{str(args.alpha)}_{str(args.bata)}_{str(args.max_extent)}_seed_{str(args.seed)}/'
    test_save_path = f'./logs/{args.exp}_stac_{str(args.use_stac)}_length_{str(args.alpha)}_{str(args.bata)}_{str(args.max_extent)}_seed_{str(args.seed)}/predictions_{args.cps}/'
    maybe_mkdir(test_save_path)

    if args.use_stac: 
        model_A = VNetDist(
            n_channels=config.num_channels,
            n_classes=config.num_cls,
            n_filters=config.n_filters,
            normalization='batchnorm',
            has_dropout=True
        ).cuda()
        model_B = VNetDist(
            n_channels=config.num_channels,
            n_classes=config.num_cls,
            n_filters=config.n_filters,
            normalization='batchnorm',
            has_dropout=True
        ).cuda()
    else:
        model_A = VNet(
            n_channels=config.num_channels,
            n_classes=config.num_cls,
            n_filters=config.n_filters,
            normalization='batchnorm',
            has_dropout=False
        ).cuda()
        model_B = VNet(
            n_channels=config.num_channels,
            n_classes=config.num_cls,
            n_filters=config.n_filters,
            normalization='batchnorm',
            has_dropout=False
        ).cuda()
        model_A.eval()
        model_B.eval()


    ckpt_path = os.path.join(snapshot_path, f'ckpts/best_model.pth')

    with torch.no_grad():
        if args.cps == "AB":
            model_A.load_state_dict(torch.load(ckpt_path)["A"])
            model_B.load_state_dict(torch.load(ckpt_path)["B"])
            print(f'load checkpoint from {ckpt_path}')
            test_all_case_AB(
                model_A, model_B,
                read_list(args.split, task=args.task),
                task=args.task,
                num_classes=config.num_cls,
                patch_size=config.patch_size,
                stride_xy=stride[0],
                stride_z=stride[1],
                test_save_path=test_save_path
            )

