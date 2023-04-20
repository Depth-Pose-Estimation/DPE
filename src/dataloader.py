import torch 
from PIL import Image
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

        assert self.rgb_imgs.shape[0] == self.depth_imgs.shape[0]

    def __len__(self):
        return self.depth_imgs.shape[0]
    
    def __getitem__(self, idx):
        rgb_img_path = os.path.join(self.rgb_img_dir, self.rgb_imgs['filename'].iloc[idx])
        depth_img_path = os.path.join(self.depth_img_dir, self.depth_imgs['filename'].iloc[idx])

        # TODO : Augment data
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        rgb_img = np.array(Image.open(rgb_img_path))
        depth_img = np.array(Image.open(depth_img_path)).astype(np.float32)
        depth_img /= np.max(depth_img)

        rgb_img = transform(rgb_img)
        depth_img = transform(depth_img)

        return rgb_img, depth_img
    
###################### TESTER CODE ##############################################

# dataset = DepthDataset(rgb_img_txt= "/home/arjun/Desktop/spring23/vlr/project/DPE/data/rgbd_dataset_freiburg2_pioneer_360/rgb.txt",
#                        depth_img_txt= "/home/arjun/Desktop/spring23/vlr/project/DPE/data/rgbd_dataset_freiburg2_pioneer_360/depth.txt",
#                        rgb_img_dir= "/home/arjun/Desktop/spring23/vlr/project/DPE/data/rgbd_dataset_freiburg2_pioneer_360",
#                        depth_img_dir="/home/arjun/Desktop/spring23/vlr/project/DPE/data/rgbd_dataset_freiburg2_pioneer_360")

# for i, (rgb, depth) in enumerate(dataset):
#     print(rgb.shape)
#     print(depth.shape)
#     break