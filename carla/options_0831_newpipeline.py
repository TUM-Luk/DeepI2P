import numpy as np
import math
import torch


class Options:
    def __init__(self):
        self.n_samples_per_subset = 64
        self.seed = 66
        self.pin_memory = True
        self.use_centered_voxel = False
        
        # data config
        self.dataroot = './dataset_large_int_train'
        self.train_subdir = 'mapping'
        self.val_subdir = 'query'
        self.test_subdir = 'query'
        
        self.train_txt = "dataset_large_int_train/train_list_deepi2p/train_75scene.txt"
        self.val_txt = "dataset_large_int_train/train_list_deepi2p/val_75scene_t3_int4.txt"
        self.test_txt = "dataset_large_int_train/train_list_deepi2p/val_75scene_t10_int1.txt"
        
        self.checkpoints_dir = 'checkpoints_carla'
        self.version = '0831_fine_new_pipeline'
        self.is_debug = False
        self.vis_debug = False
        self.is_fine_resolution = True
        self.is_remove_ground = False

        self.pc_build_interval = 2
        self.translation_max = 10.0
        self.test_translation_max = 10.0

        self.crop_original_bottom_rows = 0
        self.img_scale = 1/3
        self.img_H = 320  # after scale before crop 1080 * 1/3 = 360
        self.img_W = 640  # after scale before crop 1920 * 1/3 = 640
        # the fine resolution is img_H/scale x img_W/scale
        self.img_fine_resolution_scale = 32

        self.input_pt_num = 20480
        self.pc_min_range = -1.0
        self.pc_max_range = 40.0  # deepi2p: 50
        self.node_a_num = 128
        self.node_b_num = 128
        self.k_ab = 16
        self.k_interp_ab = 3
        self.k_interp_point_a = 3
        self.k_interp_point_b = 3

        # CAM coordinate
        self.P_tx_amplitude = self.translation_max
        self.P_ty_amplitude = self.translation_max
        self.P_tz_amplitude = self.translation_max * 0.5
        self.P_Rx_amplitude = 0.0 * math.pi / 12.0
        self.P_Ry_amplitude = 0.0 * math.pi / 12.0
        self.P_Rz_amplitude = 2.0 * math.pi
        self.dataloader_threads = 8

        self.batch_size = 8
        self.gpu_ids = [0]
        self.device = torch.device('cuda', self.gpu_ids[0])
        self.normalization = 'batch'
        self.norm_momentum = 0.1
        self.activation = 'relu'
        self.lr = 0.001
        self.lr_decay_step = 10
        self.lr_decay_scale = 0.5
        self.vis_max_batch = 4
        if self.is_fine_resolution:
            self.coarse_loss_alpha = 50
        else:
            self.coarse_loss_alpha = 1




