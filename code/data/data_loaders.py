import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from batchgenerators.augmentations.utils import create_zero_centered_coordinate_mesh
from utils import read_list, read_data, softmax
from utils.direct_field.utils_df import class2dist
from scipy.ndimage import map_coordinates
from scipy.ndimage import distance_transform_edt as distance

from skimage import segmentation as skimage_seg
import time
import torch
import cv2
import random
from scipy.ndimage.filters import gaussian_filter


def compute_sdf_optimize(img_gt):
    z_num,w_num, h_num  = img_gt.shape

    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros((z_num,w_num,h_num))
    if np.max(img_gt) == 0:
        return normalized_sdf

    posmask = img_gt.astype(np.bool)
    if posmask.any():
        negmask = ~posmask
        posdis = distance(posmask)
        negdis = distance(negmask)
        sdf = negdis - posdis
        normalized_sdf = sdf
    return normalized_sdf



def dist_flow_map_optimized(distance):

    grad_x = np.gradient(distance, axis=0)
    grad_y = np.gradient(distance, axis=1)
    grad_z = np.gradient(distance, axis=2)

    grad_all_class = np.stack([grad_x, grad_y, grad_z], axis=0)

    return grad_all_class


def dist_flow_augment_optimized(img, gt, distance, grad, length_weight = (1,-1,3)):
    gt = gt.astype(np.uint8)
    alpha_para = length_weight[0]
    beta_para = length_weight[1]
    max_extent = length_weight[2]

    x_coords_gt, y_coords_gt, z_coords_gt = np.meshgrid(range(img.shape[0]), range(img.shape[1]), range(img.shape[2]),indexing='ij')
    coordinate_mesh_gt = np.stack([x_coords_gt, y_coords_gt, z_coords_gt], axis=0)



    distance_gt_p = np.abs(distance)
    activte_weight_gt = np.exp(beta_para * distance_gt_p) * alpha_para
    activte_weight_gt = np.repeat(activte_weight_gt[np.newaxis, :, :, :], 3, axis=0)


    dil_magnitude_gt = random.randint(-max_extent, max_extent)


    grad = grad * activte_weight_gt * dil_magnitude_gt
    grad_copy = grad.copy().astype(np.float32)
    smoothed_grad_ori = np.zeros_like(grad_copy)
    for i in range(grad_copy.shape[0]):
        smoothed_grad_ori[i] = gaussian_filter(grad_copy[i], sigma=1)

    new_coordinate_mesh_gt = coordinate_mesh_gt + smoothed_grad_ori

    img_aug_gt = map_coordinates(img, new_coordinate_mesh_gt, order=1, mode='constant')
    gt_aug_gt = map_coordinates(gt, new_coordinate_mesh_gt, order=0, mode='constant')

    return img_aug_gt, gt_aug_gt

def apply_dist_anatomyaug(image,label,active_class, length_weight):
    active_label_dist = np.zeros_like(label)
    for i, is_active in enumerate(active_class):
        if is_active == 1:
            active_label_dist[label == (i + 1)] = 1

    distance_gt_numpy = compute_sdf_optimize(active_label_dist)
    grad_flow = dist_flow_map_optimized(distance_gt_numpy)
    image, label = dist_flow_augment_optimized(image, label, distance_gt_numpy,grad_flow, length_weight = length_weight)

    return image, label

class Synapse_AMOS(Dataset):
    def __init__(self, split='train', repeat=None, transform=None, unlabeled=False, is_val=False, task="synapse", num_cls=1, use_stac=0, length_weight=(1,-1,3) ):
        self.ids_list = read_list(split, task=task)
        self.repeat = repeat
        self.task=task
        if self.repeat is None:
            self.repeat = len(self.ids_list)
        print('total {} datas'.format(self.repeat))
        self.transform = transform
        self.unlabeled = unlabeled
        self.num_cls = num_cls
        self._weight = None
        self.is_val = is_val
        self.use_stac = use_stac
        self.length_weight = length_weight


        if self.is_val:
            self.data_list = {}
            for data_id in tqdm(self.ids_list): # <-- load data to memory
                image, label = read_data(data_id, task=task)
                self.data_list[data_id] = (image, label)


    def __len__(self):
        return self.repeat

    def _get_data(self, data_id):

        if self.is_val:
            image, label = self.data_list[data_id]
        else:
            image, label = read_data(data_id, task=self.task)
        return data_id, image, label


    def __getitem__(self, index):
        index = index % len(self.ids_list)
        data_id = self.ids_list[index]
        _, image, label = self._get_data(data_id)

        if self.unlabeled: # <-- for safety
            label[:] = 0

        if self.task == "synapse":
            active_class = (0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1)
        else:
            active_class = (0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1)

        image = image.clip(min=-75, max=275)
        image = (image - image.min()) / (image.max() - image.min())


        if self.use_stac and not self.unlabeled and not self.is_val:
            image, label = apply_dist_anatomyaug(image, label, active_class, self.length_weight)


        sample = {'image': image, 'label': label}


        if self.transform:
            sample = self.transform(sample)


        label = sample['label']


        if self.use_stac and not self.unlabeled and not self.is_val:
            label_copy = label.clone().detach().cpu().numpy()
            label_dist_gt = np.zeros_like(label_copy)
            for i, is_active in enumerate(active_class):
                if is_active == 1:
                    label_dist_gt[label == i] = 1

            distance_gt_numpy = compute_sdf_optimize(label_dist_gt)
            distance_gt_numpy_abs = np.abs(distance_gt_numpy)
            distance_gt_numpy = distance_gt_numpy/distance_gt_numpy_abs.max()

            distance_gt = torch.from_numpy(distance_gt_numpy).float()
            sample['distance_gt'] = distance_gt

        sample['data_id'] = data_id


        return sample


