import open3d
import torch.utils.data as data
import random
import numbers
import os
import os.path
import numpy as np
import struct
import math
import torch
import torchvision
import cv2
from PIL import Image
from torchvision import transforms
import bisect

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from data import augmentation
from util import vis_tools
from carla import options
from data.kitti_helper import FarthestSampler, camera_matrix_cropping, camera_matrix_scaling, projection_pc_img
from kapture.io.csv import kapture_from_dir
import tqdm
import quaternion
from torch.utils.data import (
    Dataset,
    DataLoader,
    ConcatDataset,
    DistributedSampler,
    RandomSampler,
    dataloader,
    Sampler
)

def downsample_with_reflectance(pointcloud, reflectance, voxel_grid_downsample_size):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.transpose(pointcloud[0:3, :]))
    reflectance_max = np.max(reflectance)

    fake_colors = np.zeros((pointcloud.shape[1], 3))
    fake_colors[:, 0] = reflectance / reflectance_max
    pcd.colors = open3d.utility.Vector3dVector(fake_colors)
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_grid_downsample_size)
    down_pcd_points = np.transpose(np.asarray(down_pcd.points))  # 3xN
    pointcloud = down_pcd_points
    reflectance = np.asarray(down_pcd.colors)[:, 0] * reflectance_max

    return pointcloud, reflectance

class RandomConcatSampler(Sampler):
    def __init__(self,
                 data_source: ConcatDataset,
                 n_samples_per_subset: int,
                 subset_replacement: bool=True,
                 seed: int=None,
                 ):
        if not isinstance(data_source, ConcatDataset):
            raise TypeError("data_source should be torch.utils.data.ConcatDataset")
        
        self.data_source = data_source
        self.n_subset = len(self.data_source.datasets)
        self.n_samples_per_subset = n_samples_per_subset
        self.n_samples = self.n_subset * self.n_samples_per_subset 
        self.subset_replacement = subset_replacement
        self.generator = torch.manual_seed(seed)
        
    def __len__(self):
        return self.n_samples
    
    def __iter__(self):
        indices = []
        # sample from each sub-dataset
        for d_idx in range(self.n_subset):
            low = 0 if d_idx==0 else self.data_source.cumulative_sizes[d_idx-1]
            high = self.data_source.cumulative_sizes[d_idx]
            
            if self.subset_replacement:
                rand_tensor = torch.randint(low, high, (self.n_samples_per_subset, ),
                                            generator=self.generator, dtype=torch.int64)
            else:  # sample without replacement
                len_subset = len(self.data_source.datasets[d_idx])
                rand_tensor = torch.randperm(len_subset, generator=self.generator) + low
                if len_subset >= self.n_samples_per_subset:
                    rand_tensor = rand_tensor[:self.n_samples_per_subset]
                else: # padding with replacement
                    rand_tensor_replacement = torch.randint(low, high, (self.n_samples_per_subset - len_subset, ),
                                                            generator=self.generator, dtype=torch.int64)
                    rand_tensor = torch.cat([rand_tensor, rand_tensor_replacement])
            indices.append(rand_tensor)
        indices = torch.stack(indices)
        indices=indices.permute(1, 0).reshape(-1)
       
        assert indices.shape[0] == self.n_samples
        return iter(indices.tolist())
    
def make_carla_dataloader(mode, opt: options.Options):
    data_root = opt.dataroot  # 'dataset_large_int_train'
    
    train_subdir = opt.train_subdir  # 'mapping'
    val_subdir = opt.val_subdir  # 'mapping'
    test_subdir = opt.test_subdir  # 'query'
    
    train_txt = opt.train_txt  # "dataset_large_int_train/train_list/train_t1_int1_v50_s25_io03_vo025.txt"
    val_txt = opt.val_txt 
    test_txt = opt.test_txt
    
    
    if mode == 'train':
        data_txt = train_txt
    elif mode == 'val':
        data_txt = val_txt
    elif mode == 'test':
        data_txt = test_txt
        
    with open(data_txt, 'r') as f:
        voxel_list = f.readlines()
        voxel_list = [voxel_name.rstrip() for voxel_name in voxel_list]
        
    
    kapture_datas={}
    sensor_datas={}
    input_path_datas={}
    train_list_kapture_map={}
    for train_path in voxel_list:
        scene=os.path.dirname(os.path.dirname(train_path))
        if scene not in kapture_datas:
            if mode=='test':
                input_path=os.path.join(data_root,scene, test_subdir)
            elif mode=='train':
                input_path=os.path.join(data_root,scene, train_subdir)
            else:
                input_path=os.path.join(data_root, scene, val_subdir)
            kapture_data=kapture_from_dir(input_path)
            sensor_dict={}
            for timestep in kapture_data.records_camera:
                _sensor_dict=kapture_data.records_camera[timestep]
                for k, v in _sensor_dict.items():
                    sensor_dict[v]=(timestep, k)
            kapture_datas[scene]=kapture_data
            sensor_datas[scene]=sensor_dict
            input_path_datas[scene]=input_path
        train_list_kapture_map[train_path]=(kapture_datas[scene], sensor_datas[scene], input_path_datas[scene])
        
    datasets = []
    
    for train_path in tqdm.tqdm(voxel_list):
        kapture_data, sensor_data, input_path=train_list_kapture_map[train_path]
        one_dataset = CarlaLoader(root_path=data_root, train_path=train_path, mode=mode, opt=opt,
                                kapture_data=kapture_data, sensor_data=sensor_data, input_path=input_path)
        
        one_dataset[10]
        datasets.append(one_dataset)
        
    
    final_dataset = ConcatDataset(datasets)
    
    if mode=='train':
        sampler = RandomConcatSampler(data_source=final_dataset,
                                    n_samples_per_subset=opt.n_samples_per_subset,
                                    subset_replacement=True,
                                    seed=opt.seed)
    
        dataloader = DataLoader(final_dataset, sampler=sampler, batch_size=opt.batch_size,
                                num_workers=opt.dataloader_threads, pin_memory=opt.pin_memory
                                )
    elif mode=='val' or mode=='test':
        # sampler = DistributedSampler(final_dataset, shuffle=False)
        
        dataloader = DataLoader(final_dataset, batch_size=opt.batch_size, shuffle=False,
                                num_workers=opt.dataloader_threads, pin_memory=opt.pin_memory
                                )
    else:
        raise ValueError
    
    return final_dataset, dataloader
        

class CarlaLoader(data.Dataset):
    def __init__(self,
                root_path, train_path, mode, opt,
                kapture_data, sensor_data, input_path):
        super(CarlaLoader, self).__init__()
        self.root_path = root_path  # "dataset_large_int_train"
        self.train_path = train_path  # 't1_int1/train_list_v50_s25_io03_vo025/train_all_50_0.npy'
        self.mode = mode
        self.opt = opt
        
        # farthest point sample
        self.farthest_sampler = FarthestSampler(dim=3)
        
        self.sensor_dict = sensor_data
        self.kaptures = kapture_data
        self.input_path = input_path
        
        self.dataset = self.make_carla_dataset(root_path, train_path, mode)
        
        # ------------- load每个voxel初始所有点的point cloud，避免每次读取整个map的点云文件 ----------------
        if mode == "train":
            self.voxel_points = self.make_voxel_pcd()
        else:
            self.voxel_id_to_voxel_points, self.voxel_id_to_voxel_centers = self.make_voxel_pcd_dict()
            
    
        print(f"{len(self.dataset)} image-voxel pairs")

    def make_carla_dataset(self, root_path, train_path, mode):
        dataset = []

        if mode == "train":
            voxel_data = np.load(os.path.join(root_path, train_path), allow_pickle=True).item()
            dataset = voxel_data['image_names']
        elif mode == "val" or mode == "test":
            voxel_data = np.load(os.path.join(root_path, train_path), allow_pickle=True).item()
            for img_name, voxel_list in voxel_data.items():
                for voxel_name in voxel_list:
                    dataset.append((img_name, voxel_name))
        else:
            raise ValueError
        
        return dataset
    
    def make_voxel_pcd(self):
        scene_name = self.train_path.split('/')[0]
        point_cloud_file = os.path.join(self.input_path, f'pcd_{scene_name}_train_down.ply')
        print(f"load pcd file from {point_cloud_file}")
        pcd = open3d.io.read_point_cloud(point_cloud_file)
        pcd_points = np.array(pcd.points)

        voxel_path = os.path.join(self.root_path, self.train_path)
        voxel_info=np.load(voxel_path, allow_pickle=True).item()

        mean = voxel_info['xyz_mean'][:3]
        median = voxel_info['xyz_median'][:3]
        std = voxel_info['xyz_std'][:3]
        voxel_min = voxel_info['xyz_min'][:3].astype(np.float32)
        voxel_max = voxel_info['xyz_max'][:3].astype(np.float32)
        voxel_center = (voxel_min + voxel_max) / 2
        voxel_size = (voxel_max - voxel_min)[0]
        
        self.voxel_min = voxel_min
        self.voxel_max = voxel_max
        self.voxel_center = voxel_center
        self.voxel_size = voxel_size

        if self.opt.use_centered_voxel:
            # voxel_points = pcd_points[np.all(pcd_points >= voxel_min - 50, axis=1) & np.all(pcd_points <= voxel_max + 50, axis=1)]
            voxel_points = pcd_points
        else:
            voxel_points = pcd_points[np.all(pcd_points >= voxel_min, axis=1) & np.all(pcd_points <= voxel_max, axis=1)]
        voxel_points = voxel_points.astype(np.float32)
        voxel_points = voxel_points.T
        
        return voxel_points
    
    def make_voxel_pcd_dict(self):
        # save pointmap for each voxel
        voxel_id_to_voxel_points = {}
        voxel_id_to_voxel_centers = {}
        
        scene_name = self.train_path.split('/')[0]
        point_cloud_file = os.path.join(self.input_path, f'pcd_{scene_name}_train_down.ply')
        print(f"load pcd file from {point_cloud_file}")
        pcd = open3d.io.read_point_cloud(point_cloud_file)
        pcd_points = np.array(pcd.points)

        for _, voxel_id in self.dataset:
            if voxel_id not in voxel_id_to_voxel_points:

                voxel_path = os.path.join(self.root_path, voxel_id)
                voxel_info=np.load(voxel_path, allow_pickle=True).item()

                mean = voxel_info['xyz_median'][:3]
                std = voxel_info['xyz_std'][:3]
                voxel_min = voxel_info['xyz_min'][:3].astype(np.float32)
                voxel_max = voxel_info['xyz_max'][:3].astype(np.float32)
                voxel_center = (voxel_min + voxel_max) / 2
                voxel_size = (voxel_max - voxel_min)[0]

                voxel_points = pcd_points[np.all(pcd_points >= voxel_min, axis=1) & np.all(pcd_points <= voxel_max, axis=1)]
                voxel_points = voxel_points.astype(np.float32)
                voxel_points = voxel_points.T

                voxel_id_to_voxel_points[voxel_id] = voxel_points
                voxel_id_to_voxel_centers[voxel_id] = voxel_center
                
        return voxel_id_to_voxel_points, voxel_id_to_voxel_centers
    
    
    def augment_pc(self, pc_np, intensity_np):
        """

        :param pc_np: 3xN, np.ndarray
        :param intensity_np: 3xN, np.ndarray
        :param sn_np: 1xN, np.ndarray
        :return:
        """
        # add Gaussian noise
        pc_np = augmentation.jitter_point_cloud(pc_np, sigma=0.01, clip=0.05)
        intensity_np = augmentation.jitter_point_cloud(intensity_np, sigma=0.01, clip=0.05)
        return pc_np, intensity_np

    def augment_img(self, img_np):
        """

        :param img: HxWx3, np.ndarray
        :return:
        """
        # color perturbation
        brightness = (0.8, 1.2)
        contrast = (0.8, 1.2)
        saturation = (0.8, 1.2)
        hue = (-0.1, 0.1)
        color_aug = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        # color_aug = transforms.ColorJitter.get_params(brightness, contrast, saturation, hue)
        img_color_aug_np = np.array(color_aug(Image.fromarray(img_np)))

        # debug image
        # debug_image = Image.fromarray(img_np)
        # debug_image.save("debug_image.png")
        # debug_image = Image.fromarray(img_color_aug_np)
        # debug_image.save("debug_image_aug.png")
        
        return img_color_aug_np
    
    def generate_random_transform(self,
                                  P_tx_amplitude, P_ty_amplitude, P_tz_amplitude,
                                  P_Rx_amplitude, P_Ry_amplitude, P_Rz_amplitude):
        """

        :param pc_np: pc in NWU coordinate
        :return:
        """
        t = [random.uniform(-P_tx_amplitude, P_tx_amplitude),
             random.uniform(-P_ty_amplitude, P_ty_amplitude),
             random.uniform(-P_tz_amplitude, P_tz_amplitude)]
        angles = [random.uniform(-P_Rx_amplitude, P_Rx_amplitude),
                  random.uniform(-P_Ry_amplitude, P_Ry_amplitude),
                  random.uniform(-P_Rz_amplitude, P_Rz_amplitude)]

        rotation_mat = augmentation.angles2rotation_matrix(angles)
        P_random = np.identity(4, dtype=np.float32)
        P_random[0:3, 0:3] = rotation_mat
        P_random[0:3, 3] = t

        return P_random.astype(np.float32)

    def downsample_np(self, pc_np, intensity_np, k):
        if pc_np.shape[1] >= k:
            choice_idx = np.random.choice(pc_np.shape[1], k, replace=False)
        else:
            fix_idx = np.asarray(range(pc_np.shape[1]))
            while pc_np.shape[1] + fix_idx.shape[0] < k:
                fix_idx = np.concatenate((fix_idx, np.asarray(range(pc_np.shape[1]))), axis=0)
            random_idx = np.random.choice(pc_np.shape[1], k - fix_idx.shape[0], replace=False)
            choice_idx = np.concatenate((fix_idx, random_idx), axis=0)
        pc_np = pc_np[:, choice_idx]
        intensity_np = intensity_np[:, choice_idx]

        return pc_np, intensity_np
    
    def __len__(self):
        return len(self.dataset)
    
    def load_pose(self, timestep, sensor_id):
        if self.kaptures.trajectories is not None and (timestep, sensor_id) in self.kaptures.trajectories:
            pose_world_to_cam = self.kaptures.trajectories[(timestep, sensor_id)]
            pose_world_to_cam_matrix = np.zeros((4, 4), dtype=np.float)
            pose_world_to_cam_matrix[0:3, 0:3] = quaternion.as_rotation_matrix(pose_world_to_cam.r)
            pose_world_to_cam_matrix[0:3, 3] = pose_world_to_cam.t_raw
            pose_world_to_cam_matrix[3, 3] = 1.0
            T = torch.tensor(pose_world_to_cam_matrix).float()
            gt_pose=T.inverse() # gt_pose为从cam_to_world
        else:
            gt_pose=T=torch.eye(4)
        return gt_pose, pose_world_to_cam
    
    def __getitem__(self, index):
        if self.mode == 'train':
            image_id = self.dataset[index]
            voxel_center = self.voxel_center
        else:
            image_id, voxel_id = self.dataset[index]
            voxel_center = self.voxel_id_to_voxel_centers[voxel_id]
            
        timestep, sensor_id=self.sensor_dict[image_id]

        # camera intrinsics
        camera_params=np.array(self.kaptures.sensors[sensor_id].camera_params[2:])
        K = np.array([[camera_params[0],0,camera_params[1]],
                    [0,camera_params[0],camera_params[2]],
                    [0,0,1]])
        
        # T from point cloud to camera
        gt_pose, gt_pose_world_to_cam_q=self.load_pose(timestep, sensor_id) # camera to world
        gt_pose_world_to_cam_q = np.concatenate((gt_pose_world_to_cam_q.t_raw, gt_pose_world_to_cam_q.r_raw))
        T_c2w = gt_pose.numpy() # camera to world
        T_w2c = np.linalg.inv(T_c2w)
        
        if self.opt.use_centered_voxel:
            # T from world to voxel coordinate 将坐标系移到camera附近，高度不动
            T_w2v = np.eye(4).astype(np.float32)
            T_w2v[:2,3] = -T_c2w[:2,3]
            T_w2v_inv = np.linalg.inv(T_w2v).copy()
        else:
            # T from world to voxel coordinate 将坐标系移到voxel中心
            T_w2v = np.eye(4).astype(np.float32)
            T_w2v[:3,3] = -voxel_center
            T_w2v_inv = np.linalg.inv(T_w2v).copy()
        
        
        # ------------- load image, original size is 1080x1920 -------------
        img = cv2.imread(os.path.join(self.input_path, 'sensors/records_data', image_id))

        # scale to 540x960
        img = cv2.resize(img,
                         (int(round(img.shape[1] * self.opt.img_scale)), int(round((img.shape[0] * self.opt.img_scale)))),
                         interpolation=cv2.INTER_LINEAR)
        K = camera_matrix_scaling(K, self.opt.img_scale)
        
        # random crop into input size
        if 'train' == self.mode:
            img_crop_dx = random.randint(0, img.shape[1] - self.opt.img_W)
            img_crop_dy = random.randint(0, img.shape[0] - self.opt.img_H)
        else:
            img_crop_dx = int((img.shape[1] - self.opt.img_W) / 2)
            img_crop_dy = int((img.shape[0] - self.opt.img_H) / 2)
        # crop image
        img = img[img_crop_dy:img_crop_dy + self.opt.img_H,
              img_crop_dx:img_crop_dx + self.opt.img_W, :]
        K = camera_matrix_cropping(K, dx=img_crop_dx, dy=img_crop_dy)
        
        # ------------- load point cloud ----------------
        if self.mode == "train":
            npy_data = self.voxel_points.copy() # important! keep self.voxel points unchanged
        else:
            npy_data = self.voxel_id_to_voxel_points[voxel_id].copy()
        npy_data = npy_data[:, np.random.permutation(npy_data.shape[1])]
        pc_np = npy_data[0:3, :]  # 3xN
        intensity_np = np.zeros((1, pc_np.shape[1]), dtype=np.float32)  # 1xN
        
        # origin pcd
        if self.opt.vis_debug:
            debug_point_cloud = open3d.geometry.PointCloud()
            debug_point_cloud.points = open3d.utility.Vector3dVector(pc_np.T)
            open3d.io.write_point_cloud(f'z_dataset/input_pcd_{index}.ply', debug_point_cloud)
        
        # transform frame to voxel center
        pc_homo_np = np.concatenate((pc_np, np.ones((1, pc_np.shape[1]), dtype=pc_np.dtype)), axis=0)  # 4xN
        Pr_pc_homo_np = np.dot(T_w2v, pc_homo_np)  # 4xN
        pc_np = Pr_pc_homo_np[0:3, :]  # 3xN
        if self.opt.vis_debug:
            debug_point_cloud = open3d.geometry.PointCloud()
            debug_point_cloud.points = open3d.utility.Vector3dVector(pc_np.T)
            open3d.io.write_point_cloud(f'z_dataset/input_pcd_T_w2v_{index}.ply', debug_point_cloud)
            
        # limit max_z, the pc is in CAMERA coordinate
        if self.opt.use_centered_voxel:
            pc_np_x_square = np.square(pc_np[0, :])
            pc_np_y_square = np.square(pc_np[1, :])
            pc_np_range_square = pc_np_x_square + pc_np_y_square
            pc_mask_range = pc_np_range_square < self.opt.pc_max_range * self.opt.pc_max_range
            pc_np = pc_np[:, pc_mask_range]
            intensity_np = intensity_np[:, pc_mask_range]
            if self.opt.vis_debug:
                debug_point_cloud = open3d.geometry.PointCloud()
                debug_point_cloud.points = open3d.utility.Vector3dVector(pc_np.T)
                open3d.io.write_point_cloud(f'z_dataset/input_pcd_T_w2v_limit_{index}.ply', debug_point_cloud)
        else:
            None
            
        # point cloud too huge, voxel grid downsample first
        # 暂无

        # random sampling
        pc_np, intensity_np = self.downsample_np(pc_np, intensity_np, self.opt.input_pt_num)
        
        #  ------------- apply random transform on points under the NWU coordinate ------------
        if 'train' == self.mode:
            Pr = self.generate_random_transform(self.opt.P_tx_amplitude, self.opt.P_ty_amplitude, self.opt.P_tz_amplitude,
                                                self.opt.P_Rx_amplitude, self.opt.P_Ry_amplitude, self.opt.P_Rz_amplitude)
            Pr_inv = np.linalg.inv(Pr)

            # -------------- augmentation ----------------------
            pc_np, intensity_np = self.augment_pc(pc_np, intensity_np)
            if random.random() > 0.5:
                img = self.augment_img(img)
        elif 'val_random_Ry' == self.mode:
            Pr = self.generate_random_transform(0, 0, 0,
                                                0, math.pi*2, 0)
            Pr_inv = np.linalg.inv(Pr)
        elif 'val' == self.mode or 'test' == self.mode:
            Pr = np.identity(4, dtype=np.float32)
            Pr_inv = np.identity(4, dtype=np.float32)
        
        t_ij = (T_w2c @ T_w2v_inv)[0:3, 3]
        P = T_w2c @ T_w2v_inv @ Pr_inv # 对于输入点云的新GT Pose
        
        # then aug to get final input pcd
        pc_homo_np = np.concatenate((pc_np, np.ones((1, pc_np.shape[1]), dtype=pc_np.dtype)), axis=0)  # 4xN
        Pr_pc_homo_np = np.dot(Pr, pc_homo_np)  # 4xN
        pc_np = Pr_pc_homo_np[0:3, :]  # 3xN
        if self.opt.vis_debug:
            debug_point_cloud = open3d.geometry.PointCloud()
            debug_point_cloud.points = open3d.utility.Vector3dVector(pc_np.T)
            open3d.io.write_point_cloud(f'z_dataset/input_pcd_T_w2v_aug_{index}.ply', debug_point_cloud)
        
        # input pcd in cam coordinate frame
        if self.opt.vis_debug:
            pc_homo_np = np.concatenate((pc_np, np.ones((1, pc_np.shape[1]), dtype=pc_np.dtype)), axis=0)  # 4xN
            Pr_pc_homo_np = np.dot(P, pc_homo_np)  # 4xN
            pc_np_in_cam = Pr_pc_homo_np[0:3, :]  # 3xN
            debug_point_cloud = open3d.geometry.PointCloud()
            debug_point_cloud.points = open3d.utility.Vector3dVector(pc_np_in_cam.T)
            open3d.io.write_point_cloud(f'z_dataset/input_pcd_in_cam_{index}.ply', debug_point_cloud)
            
        # ------------ Farthest Point Sampling ------------------
        # node_a_np = fps_approximate(pc_np, voxel_size=4.0, node_num=self.opt.node_a_num)
        node_a_np, _ = self.farthest_sampler.sample(pc_np[:, np.random.choice(pc_np.shape[1],
                                                                              int(self.opt.node_a_num*8),
                                                                              replace=False)],
                                                    k=self.opt.node_a_num)
        node_b_np, _ = self.farthest_sampler.sample(pc_np[:, np.random.choice(pc_np.shape[1],
                                                                            int(self.opt.node_b_num*8),
                                                                            replace=False)],
                                                  k=self.opt.node_b_num)
        
        # -------------- convert to torch tensor ---------------------
        pc = torch.from_numpy(pc_np)  # 3xN
        intensity = torch.from_numpy(intensity_np)  # 1xN
        sn = torch.zeros(pc.size(), dtype=pc.dtype, device=pc.device)
        node_a = torch.from_numpy(node_a_np)  # 3xMa
        node_b = torch.from_numpy(node_b_np)  # 3xMb

        P = torch.from_numpy(P[0:3, :].astype(np.float32))  # 3x4

        img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).contiguous()  # 3xHxW
        K = torch.from_numpy(K.astype(np.float32))  # 3x3

        t_ij = torch.from_numpy(t_ij.astype(np.float32))  # 3

        # print(P)
        # print(t_ij)
        # print(pc)
        # print(intensity)
            
        return pc, intensity, sn, node_a, node_b, \
               P, img, K, \
               t_ij
               


if __name__ == '__main__':
    opt = options.Options()
    dataset = make_carla_dataloader(mode="train", opt=opt)
    
    # root_path = '/extssd/jiaxin/oxford'
    # opt = options.Options()
    # oxfordloader = OxfordLoader(root_path, 'train', opt)

    # for i in range(0, len(oxfordloader), 10000):
    #     print('--- %d ---' % i)
    #     data = oxfordloader[i]
    #     for item in data:
    #         print(item.size())
