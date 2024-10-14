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

def compute_sdf(img_gt,active_class):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size,c, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """
    z_num,w_num, h_num  = img_gt.shape
    active_num = len([x for x in active_class if x != 0])
    img_gt_split = np.zeros((z_num,w_num,h_num,active_num))
    active_num = 0
    sum_num = 1
    for active in active_class:
        if active != 0:
            img_gt_split[...,active_num][img_gt==sum_num] =1
            active_num += 1
        sum_num += 1

    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros((active_num,z_num,w_num,h_num))
    if np.max(img_gt) == 0:
        return normalized_sdf
    for c in range(active_num):
        img_gt = img_gt_split[...,c]
        posmask = img_gt.astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
            sdf[boundary==1] = 0
            normalized_sdf[c,...] = sdf

            # assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
            # assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))
    return normalized_sdf

def compute_sdf_optimize(img_gt):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size,c, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """
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
        # boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
        sdf = negdis - posdis
        # sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
        # sdf[boundary == 1] = 0
        normalized_sdf = sdf

        # assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
        # assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))
    return normalized_sdf

def draw_mask(mask):
    w,h = mask.shape
    overlay1 = np.zeros((w,h,3))

    overlay1[mask == 1, 0] = 50
    overlay1[mask == 1, 1] = 0
    overlay1[mask == 1, 2] = 0

    overlay1[mask == 2, 0] = 0
    overlay1[mask == 2, 1] = 50
    overlay1[mask == 2, 2] = 0

    overlay1[mask == 3, 0] = 0
    overlay1[mask == 3, 1] = 0
    overlay1[mask == 3, 2] = 50

    overlay1[mask == 4, 0] = 50
    overlay1[mask == 4, 1] = 100
    overlay1[mask == 4, 2] = 0

    overlay1[mask == 5, 0] = 100
    overlay1[mask == 5, 1] = 0
    overlay1[mask == 5, 2] = 50

    overlay1[mask == 6, 0] = 0
    overlay1[mask == 6, 1] = 50
    overlay1[mask == 6, 2] = 100

    overlay1[mask == 7, 0] = 100
    overlay1[mask == 7, 1] = 50
    overlay1[mask == 7, 2] = 50

    overlay1[mask == 8, 0] = 150
    overlay1[mask == 8, 1] = 0
    overlay1[mask == 8, 2] = 0

    overlay1[mask == 9, 0] = 0
    overlay1[mask == 9, 1] = 150
    overlay1[mask == 9, 2] = 0

    overlay1[mask == 10, 0] = 0
    overlay1[mask == 10, 1] = 0
    overlay1[mask == 10, 2] = 150

    overlay1[mask == 11, 0] = 150
    overlay1[mask == 11, 1] = 0
    overlay1[mask == 11, 2] = 100

    overlay1[mask == 12, 0] = 100
    overlay1[mask == 12, 1] = 150
    overlay1[mask == 12, 2] = 0

    overlay1[mask == 13, 0] = 100
    overlay1[mask == 13, 1] = 0
    overlay1[mask == 13, 2] = 150

    overlay1[mask == 14, 0] = 100
    overlay1[mask == 14, 1] = 100
    overlay1[mask == 14, 2] = 150

    overlay1[mask == 15, 0] = 200
    overlay1[mask == 15, 1] = 50
    overlay1[mask == 15, 2] = 150
    return overlay1

def get_organ_gradient_field(organ, spacing_ratio=1.5/3.0, blur=4):
    """
    Calculates the gradient field around the organ segmentations for the anatomy-informed augmentation

    :param organ: binary organ segmentation
    :param spacing_ratio: ratio of the axial spacing and the slice thickness, needed for the right vector field calculation
    :param blur: kernel constant
    """
    organ_blurred = gaussian_filter(organ.astype(float),
                                    sigma=(blur * spacing_ratio, blur, blur),
                                    order=0,
                                    mode='nearest')

    t, u, v = np.gradient(organ_blurred)
    t = t * spacing_ratio

    return t, u, v

def ignore_anatomy(segm, max_annotation_value=1, replace_value=0):
    segm[segm > max_annotation_value] = replace_value
    return segm



def augment_anatomy_informed(data, seg,
                             active_organs, dilation_ranges, directions_of_trans, modalities,
                             spacing_ratio=1.5/3.0, blur=4, anisotropy_safety= True,
                             max_annotation_value=15, replace_value=0):
    if sum(active_organs) > 0:
        data_shape = data.shape
        coords = create_zero_centered_coordinate_mesh(data_shape[-3:])

        for organ_idx, active in reversed(list(enumerate(active_organs))):
            if active:
                dil_magnitude = np.random.uniform(low=dilation_ranges[organ_idx][0], high=dilation_ranges[organ_idx][1])

                t, u, v = get_organ_gradient_field(seg == organ_idx + 1,
                                                   spacing_ratio=spacing_ratio,
                                                   blur=blur)

                if directions_of_trans[organ_idx][0]:
                    coords[0, :, :, :] = coords[0, :, :, :] + t * dil_magnitude * spacing_ratio
                if directions_of_trans[organ_idx][1]:
                    coords[1, :, :, :] = coords[1, :, :, :] + u * dil_magnitude
                if directions_of_trans[organ_idx][2]:
                    coords[2, :, :, :] = coords[2, :, :, :] + v * dil_magnitude

        for d in range(3):
            ctr = data.shape[d+1] / 2  # !!!
            coords[d] += ctr - 0.5  # !!!

        if anisotropy_safety:
            coords[0, 0, :, :][coords[0, 0, :, :] < 0] = 0.0
            coords[0, 1, :, :][coords[0, 1, :, :] < 0] = 0.0
            coords[0, -1, :, :][coords[0, -1, :, :] > (data_shape[-3] - 1)] = data_shape[-3] - 1
            coords[0, -2, :, :][coords[0, -2, :, :] > (data_shape[-3] - 1)] = data_shape[-3] - 1


        for modality in modalities:
            data[modality, :, :, :] = map_coordinates(data[modality, :, :, :], coords, order=1, mode='constant')
            seg[:, :, :] = map_coordinates(seg[:, :, :], coords, order=0, mode='constant')
        seg[:, :, :] = ignore_anatomy(seg[:, :, :], max_annotation_value=max_annotation_value, replace_value=replace_value)
        # seg[:, :, :] = map_coordinates(seg[:, :, :], coords, order=0, mode='constant')

    else:
        seg[:, :, :] = ignore_anatomy(seg[:, :, :], max_annotation_value=max_annotation_value, replace_value=replace_value)

    return data, seg

# def dist_flow_map(distance):
#     grad_all_class = np.zeros((distance.shape[0], 3, distance.shape[1], distance.shape[2], distance.shape[3]))
#     for index in range(distance.shape[0]):
#         distance_now = distance[index, :, :, :]
#         grad_x = np.gradient(distance_now, axis=0)
#         grad_y = np.gradient(distance_now, axis=1)
#         grad_z = np.gradient(distance_now, axis=2)
#         grad_x = grad_x / (np.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2 + 1e-5))
#         grad_y = grad_y / (np.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2 + 1e-5))
#         grad_z = grad_z / (np.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2 + 1e-5))
#
#         grad = np.stack([grad_x, grad_y, grad_z], axis=0)
#         grad_all_class[index, :, :, :, :] = grad
#     return grad_all_class
def dist_flow_map_optimized(distance):
    # 计算每个通道的梯度
    grad_x = np.gradient(distance, axis=0)
    grad_y = np.gradient(distance, axis=1)
    grad_z = np.gradient(distance, axis=2)

    # 计算梯度的模
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-5)

    # 归一化梯度
    grad_x_normalized = grad_x / gradient_magnitude
    grad_y_normalized = grad_y / gradient_magnitude
    grad_z_normalized = grad_z / gradient_magnitude

    grad_xx = np.gradient(grad_x, axis=0)
    grad_yy = np.gradient(grad_y, axis=1)
    grad_zz = np.gradient(grad_z, axis=2)
    cuvature = grad_xx + grad_yy + grad_zz
    # 将归一化后的梯度叠加
    # grad_all_class = np.stack([grad_x_normalized, grad_y_normalized, grad_z_normalized], axis=0)
    grad_all_class = np.stack([grad_x, grad_y, grad_z], axis=0)

    return grad_all_class, cuvature
# def dist_flow_augment(img, gt, distance ,grad):
#     gt = gt.astype(np.uint8)
#     # img = img[0]
#     x_coords_gt, y_coords_gt, z_coords_gt = np.meshgrid(range(img.shape[0]), range(img.shape[1]), range(img.shape[2]),indexing='ij')
#     coordinate_mesh_gt = np.stack([x_coords_gt, y_coords_gt, z_coords_gt])
#     for index in range(grad.shape[0]):
#         activte_weight_gt_now = 1 - np.abs(distance[index])
#         activte_weight_gt_now = np.repeat(activte_weight_gt_now[np.newaxis, :, :, :], 3, axis=0)
#         grad_now = grad[index, :, :, :, :]
#         grad_now = grad_now * activte_weight_gt_now
#
#         # dil_magnitude_gt = random.randint(-10, 10)
#         dil_magnitude_gt = -10
#         grad_now = grad_now * dil_magnitude_gt
#         # smoothed_grad_now = np.zeros_like(grad_now)
#         # for i in range(grad_now.shape[0]):
#         #     smoothed_grad_now[i] = gaussian_filter(grad_now[i], sigma=1)
#         coordinate_mesh_gt = coordinate_mesh_gt + grad_now
#
#     img_aug_gt = map_coordinates(img, coordinate_mesh_gt, order=1, mode='constant')
#     gt_aug_gt = map_coordinates(gt, coordinate_mesh_gt, order=0, mode='constant')
#
#     return img_aug_gt, gt_aug_gt

def dist_flow_augment_optimized(img, gt, distance, grad):
    gt = gt.astype(np.uint8)

    # 创建坐标网格
    x_coords_gt, y_coords_gt, z_coords_gt = np.meshgrid(range(img.shape[0]), range(img.shape[1]), range(img.shape[2]),
                                                        indexing='ij')
    coordinate_mesh_gt = np.stack([x_coords_gt, y_coords_gt, z_coords_gt], axis=0)

    # 计算激活权重并重复扩展
    # #todo
    distance_gt_p = np.abs(distance)
    # distance_gt_p = distance_gt_p / distance_gt_p.max()
    # activte_weight_gt = 1 - distance_gt_p

    activte_weight_gt = np.exp(-1 * distance_gt_p)

    #
    # activte_weight_gt = activte_weight_gt * curvature



    # activte_weight_gt = 1 - np.abs(distance)
    activte_weight_gt = np.repeat(activte_weight_gt[np.newaxis, :, :, :], 3, axis=0)

    # 随机扩张幅度
    dil_magnitude_gt = random.randint(-3, 3)
    # dil_magnitude_gt = -1
    # 应用激活权重和扩张幅度，同时高斯滤波
    grad = grad * activte_weight_gt * dil_magnitude_gt
    grad_copy = grad.copy().astype(np.float32)
    smoothed_grad_ori = np.zeros_like(grad_copy)
    for i in range(grad_copy.shape[0]):
        smoothed_grad_ori[i] = gaussian_filter(grad_copy[i], sigma=1)
    # grad_transposed = np.moveaxis(grad, [2, 3, 4], [0, 1, 2])
    # grad_smoothed_transposed = np.array([gaussian_filter(slice, 1) for slice in grad_transposed])
    # smoothed_grad = np.moveaxis(grad_smoothed_transposed, [0, 1, 2], [2, 3, 4])



    # 更新坐标网格
    # coordinate_mesh_gt += np.sum(smoothed_grad_ori, axis=0)
    new_coordinate_mesh_gt = coordinate_mesh_gt + smoothed_grad_ori
    # 应用坐标映射

    img_aug_gt = map_coordinates(img, new_coordinate_mesh_gt, order=1, mode='constant')
    gt_aug_gt = map_coordinates(gt, new_coordinate_mesh_gt, order=0, mode='constant')

    return img_aug_gt, gt_aug_gt

def apply_dist_anatomyaug(image,label,active_class):
    active_label_dist = np.zeros_like(label)
    for i, is_active in enumerate(active_class):
        if is_active == 1:
            active_label_dist[label == (i + 1)] = 1

    distance_gt_numpy = compute_sdf_optimize(active_label_dist)
    grad_flow, curvature = dist_flow_map_optimized(distance_gt_numpy)
    image, label = dist_flow_augment_optimized(image, label, distance_gt_numpy,grad_flow)

    return image, label

class Synapse_AMOS(Dataset):
    def __init__(self, split='train', repeat=None, transform=None, unlabeled=False, is_val=False, task="synapse", num_cls=1,use_anatomyaug=0, use_dist_anatomyaug=0):
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
        self.use_anatomyaug = use_anatomyaug
        self.use_dist_anatomyaug = use_dist_anatomyaug



        if self.is_val:
            self.data_list = {}
            for data_id in tqdm(self.ids_list): # <-- load data to memory
                image, label = read_data(data_id, task=task)
                self.data_list[data_id] = (image, label)


    def __len__(self):
        return self.repeat

    def _get_data(self, data_id):
        # [160, 384, 384]
        if self.is_val:
            image, label = self.data_list[data_id]
        else:
            image, label = read_data(data_id, task=self.task)
        return data_id, image, label


    def __getitem__(self, index):
        index = index % len(self.ids_list)
        data_id = self.ids_list[index]
        _, image, label = self._get_data(data_id)
        # todo
        if self.unlabeled: # <-- for safety
            label[:] = 0

        # print("before",image.min(), image.max())
        # image = (image - image.min()) / (image.max() - image.min())
        if self.task == "synapse":
            p_per_sample = (0, 0, 0, 0.3, 0.3, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5)
            #原来正常设置
            # dilation_ranges = (
            # (0, 0), (0, 0), (0, 0), (20, 30), (30, 40), (0, 0), (0, 0), (0, 0), (0, 0), (10, 20), (20, 30), (100, 110),
            # (100, 110)
            # )

            dilation_ranges = (
            (0, 0), (0, 0), (0, 0), (100, 200), (100, 200), (0, 0), (0, 0), (0, 0), (0, 0), (100, 200), (100, 200), (100, 200),
            (100, 200)
            )

            directions_of_trans = (
            (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1),
            (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)
            )
            active_class = (0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1)
            # unactive_class =  (1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0 ,0 ,0)
        else:
            p_per_sample = (0, 0, 0, 0.3, 0.3, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0, 0, 0)
            #原来正常设置
            # dilation_ranges = (
            # (0, 0), (0, 0), (0, 0), (20, 30), (30, 40), (0, 0), (0, 0), (0, 0), (0, 0), (20, 30), (100, 110),
            # (100, 110), (0, 0), (0, 0), (0, 0)
            # )
            dilation_ranges = (
            (0, 0), (0, 0), (0, 0), (100, 200), (100, 200), (0, 0), (0, 0), (0, 0), (0, 0), (100, 200), (100, 200), (100, 200),
            (100, 200)
            )

            directions_of_trans = (
            (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1),
            (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)
            )
            # active_class   = (0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0)
            active_class   = (0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1)
            # unactive_class = (1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0 ,0, 1, 1, 1)
        image = image.clip(min=-75, max=275)
        image = (image - image.min()) / (image.max() - image.min())
        # image = (image - image.mean()) / (image.std() + 1e-8)
        # print("after",image.min(), image.max())
        # print("ss",image.max())
        # image = image.astype(np.float32)
        # label = label.astype(np.int8)

        # print(image.shape, label.shape)
        # if data_id == 'amos_0017':
        #     for slice_index in range(image.shape[0]):
        #         cv2.imwrite('/data/ylgu/Medical/Semi_medical_data/amos22_processed/view/ori/' + str(slice_index) + '.png', image[slice_index, :, :] * 255)
        #         cv2.imwrite('/data/ylgu/Medical/Semi_medical_data/amos22_processed/view/ori/' + str(slice_index) + '_label.png', draw_mask(label[slice_index, :, :]))
        #     print('1')

        if self.use_anatomyaug and not self.unlabeled and not self.is_val:
            # print('1'*50)
            active_organs = []
            for prob in p_per_sample:
                if np.random.uniform() < prob:
                    active_organs.append(1)
                else:
                    active_organs.append(0)
            image = image[np.newaxis, ...]
            if random.random() > 0:
                image, label = augment_anatomy_informed(image, label, active_organs=active_organs,
                                                        dilation_ranges=dilation_ranges,
                                                        modalities=(0,0,0),
                                                        directions_of_trans=directions_of_trans,
                                                        )
            image = image[0, ...]
            # if data_id == 'amos_0017':
            #     for slice_index in range(image.shape[0]):
            #         cv2.imwrite(
            #             '/data/ylgu/Medical/Semi_medical_data/amos22_processed/view/aug/' + str(slice_index) + '.png',
            #             image[slice_index, :, :] * 255)
            #         cv2.imwrite(
            #             '/data/ylgu/Medical/Semi_medical_data/amos22_processed/view/aug/' + str(slice_index) + '_label.png',
            #             draw_mask(label[slice_index, :, :]))
            #     print('1')

        if self.use_dist_anatomyaug and not self.unlabeled and not self.is_val:
            # print('1')
            # active_label_dist = np.zeros_like(label)
            # for i, is_active in enumerate(active_class):
            #     if is_active == 1:
            #         active_label_dist[label == (i+1)] = 1
            #
            # distance_gt_numpy = compute_sdf_optimize(active_label_dist)
            # grad_flow, curvature = dist_flow_map_optimized(distance_gt_numpy)
            # image, label = dist_flow_augment_optimized(image, label, distance_gt_numpy,grad_flow,curvature)
            # aug_num = random.randint(1, 3)
            # aug_num = 1
            # for aug_index in range(aug_num):
            image, label = apply_dist_anatomyaug(image, label, active_class)


        sample = {'image': image, 'label': label}

        # print(sample['image'])

        if self.transform:
            # if not self.unlabeled and not self.is_val:
            #     sample = self.transform(sample, weights=self.transform.weights)
            # else:
            sample = self.transform(sample)


        label = sample['label']


        if self.use_dist_anatomyaug and not self.unlabeled and not self.is_val:
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


