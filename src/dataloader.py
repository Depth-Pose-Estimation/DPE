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
        depth_img /= np.max(depth_img)

        rgb_img_t = transform(rgb_img_t)
        rgb_img_t1 = transform(rgb_img_t1)
        depth_img = transform(depth_img)

        return rgb_img_t, rgb_img_t1, depth_img
    
class DepthDatasetKitti(torch.utils.data.Dataset):
    def __init__(self, split = "train", data_dir = None):
        super (DepthDatasetKitti, self).__init__()

        self.split_type = split
        self.data_dir = data_dir

        if self.split_type == "train":
            self.txt_file = "train.txt"

        elif self.split_type == "val":
            self.txt_file = "val.txt"
        
        f = open(os.path.join(data_dir, self.txt_file))
        self.folders = f
        
        self.all_files = []
        self.all_folders = []
        for folders in self.folders:
            files_in_folders = os.listdir(os.path.join(data_dir, folders.strip("\n")))
            for f in files_in_folders:
                if (".jpg") in f:
                    self.all_files.append(os.path.join(folders.strip("\n"), f))

        #TODO : Handle poses

    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, idx):
        rgb_image_t_path = os.path.join(self.data_dir, self.all_files[idx])
        depth_image_t_path = rgb_image_t_path.strip(".jpg") + "_depth_interp.npy"

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        rgb_image_t = ImageOps.grayscale(Image.open(rgb_image_t_path))
        depth_image_t = np.load(depth_image_t_path)

        rgb_image_t = transform(rgb_image_t)
        depth_image_t = transform(depth_image_t)

        return rgb_image_t, depth_image_t


    
###################### TESTER CODE ##############################################

# dataset = DepthDataset(rgb_img_txt= "/home/arjun/Desktop/spring23/vlr/project/DPE/data/rgbd_dataset_freiburg2_pioneer_360/rgb.txt",
#                        depth_img_txt= "/home/arjun/Desktop/spring23/vlr/project/DPE/data/rgbd_dataset_freiburg2_pioneer_360/depth.txt",
#                        rgb_img_dir= "/home/arjun/Desktop/spring23/vlr/project/DPE/data/rgbd_dataset_freiburg2_pioneer_360",
#                        depth_img_dir="/home/arjun/Desktop/spring23/vlr/project/DPE/data/rgbd_dataset_freiburg2_pioneer_360")

# print(len(dataset))
# for i, (rgb_t, rgb_t1, depth) in enumerate(dataset):
#     print(rgb_t.shape)
#     print(rgb_t1.shape)
#     print(depth.shape)
#     break
    # print(i)

# dataset = DepthDatasetKitti(split="train", data_dir = "/home/arjun/Desktop/vlr_project/data/dump")

# print(len(dataset))
# for i, (rgb_t, depth) in enumerate(dataset):
#     print(rgb_t.shape)
#     # print(rgb_t1.shape)
#     print(depth.shape)
#     break
#     print(i)