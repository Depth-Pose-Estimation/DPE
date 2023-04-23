import torch 
from PIL import Image, ImageOps
import torchvision
import pandas as pd
import numpy as np
import os

class DepthDataset(torch.utils.data.Dataset):
    def __init__(self, rgb_img_txt, depth_img_txt, rgb_img_dir, depth_img_dir):
        
        self.rgb_imgs = pd.read_csv(rgb_img_txt, sep = '\s+')
        self.depth_imgs = pd.read_csv(depth_img_txt, sep = '\s+')
        self.rgb_img_dir = rgb_img_dir
        self.depth_img_dir = depth_img_dir

        # filter out valid pairs based on timestamps (acceptable difference : 1ms)
        self.valid_pair_mask = np.where(self.rgb_imgs['timestamp'] - self.depth_imgs['timestamp'] < 1e-3)

        self.rgb_imgs = self.rgb_imgs.loc[self.valid_pair_mask[0],:]
        self.depth_imgs = self.depth_imgs.loc[self.valid_pair_mask[0],:]

        # RGB-D SLAM has scaled depth by 5000
        self.depth_scale = 5000

        assert self.rgb_imgs.shape[0] == self.depth_imgs.shape[0]

    def __len__(self):
        return self.depth_imgs.shape[0] - 1
    
    def __getitem__(self, idx):
        rgb_img_t_path = os.path.join(self.rgb_img_dir, self.rgb_imgs['filename'].iloc[idx])
        rgb_img_t1_path = os.path.join(self.rgb_img_dir, self.rgb_imgs['filename'].iloc[idx + 1])
        depth_img_path = os.path.join(self.depth_img_dir, self.depth_imgs['filename'].iloc[idx])

        # TODO : Augment data
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        rgb_img_t = ImageOps.grayscale(Image.open(rgb_img_t_path))
        rgb_img_t1 = ImageOps.grayscale(Image.open(rgb_img_t1_path))
        depth_img = np.array(Image.open(depth_img_path)).astype(np.float32)
        #TODO - hardcoded for now. RGB-D SLAM has scaled depth by 5000 (depth_scale). Don't normalize depth
        depth_img /= self.depth_scale
        # depth_img /= np.max(depth_img)

        rgb_img_t = transform(rgb_img_t)
        rgb_img_t1 = transform(rgb_img_t1)
        depth_img = transform(depth_img)

        return rgb_img_t, rgb_img_t1, depth_img
    
###################### TESTER CODE ##############################################

# dataset = DepthDataset(rgb_img_txt= "/home/arjun/Desktop/spring23/vlr/project/DPE/data/rgbd_dataset_freiburg2_pioneer_360/rgb.txt",
#                        depth_img_txt= "/home/arjun/Desktop/spring23/vlr/project/DPE/data/rgbd_dataset_freiburg2_pioneer_360/depth.txt",
#                        rgb_img_dir= "/home/arjun/Desktop/spring23/vlr/project/DPE/data/rgbd_dataset_freiburg2_pioneer_360",
#                        depth_img_dir="/home/arjun/Desktop/spring23/vlr/project/DPE/data/rgbd_dataset_freiburg2_pioneer_360")

# for i, (rgb_t, rgb_t1, depth) in enumerate(dataset):
#     print(rgb_t.shape)
#     print(rgb_t1.shape)
#     print(depth.shape)
#     break