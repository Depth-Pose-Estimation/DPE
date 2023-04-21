import torch
import torchvision
import argparse
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from dataloader import DepthDataset
from model import DepthPredictor
import matplotlib.pyplot as plt
import numpy as np

class Trainer:
    def __init__(self, model, train_dataloader, test_dataloader, learning_rate = 0.0006, batch_size = 100, 
                    num_epochs = 20, scheduler = None, device = 'cuda'):
      
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device

        self.optim = torch.optim.Adam(self.model.parameters(), self.learning_rate)
        self.scheduler = scheduler
    
    def loss(self, model_out, depth_imgs_gt):
        loss_fn = torch.nn.MSELoss()
        # loss_fn = torch.nn.CrossEntropyLoss()
        mask = (depth_imgs_gt != 0)
        loss_val = loss_fn(model_out[mask], depth_imgs_gt[mask])
        return loss_val
    
    def train(self):


        self.model.to(self.device)
        for i in range(self.num_epochs):
            train_loss = 0
            for rgb_imgs, depth_imgs_gt in self.train_dataloader:
                rgb_imgs = rgb_imgs.to(self.device)
                depth_imgs_gt = depth_imgs_gt.to(self.device)
                model_out = self.model(rgb_imgs)

                loss = self.loss(model_out, depth_imgs_gt)
                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                train_loss += loss.item()
            
            print(f"EPOCH {i + 1} [TRAIN LOSS] {train_loss}")

            if self.scheduler is not None:
                self.scheduler.step()

        return train_loss
    
    def eval(self):
        val_loss = 0
        self.model.eval()

        with torch.no_grad():
            for rgb_imgs, depth_imgs_gt in self.test_dataloader:
                rgb_imgs = rgb_imgs.to(self.device)
                depth_imgs_gt = depth_imgs_gt.to(self.device)
                model_out = self.model(rgb_imgs)

                loss = self.loss(model_out, depth_imgs_gt)
                val_loss += loss.item()

            print(f"[VAL LOSS] {val_loss}")

        images, depth = iter(self.test_dataloader).next()
        self.show_image(torchvision.utils.make_grid(images[1:10],10,1), torchvision.utils.make_grid(depth[1:10],10,1))
        # plt.show()
        # plt.figure()

        self.visualise_output(images, self.model)

        self.model.train()
        
        return val_loss
    
    def to_img(self, x):
        x = x.clamp(0, 1)
        return x

    def show_image(self, img, depth = None):
        # img = self.to_img(img)
        npimg = img.numpy()
        print(f"[INFO] Saving gt img plot")
        plt.imsave("img.png", np.transpose(npimg, (1, 2, 0)))

        npdepth = depth.numpy()
        print(f"[INFO] Saving gt plot")
        plt.imsave("gt_depth.png", np.transpose(npdepth, (1, 2, 0)))


    def visualise_output(self, images, model):

        with torch.no_grad():
        
            images = images.to(self.device)
            images = model(images)
            images = images.cpu()
            # images = self.to_img(images)
            np_imagegrid = torchvision.utils.make_grid(images[1:10], 10, 1).numpy()
            print(f"[INFO] Saving pred plot")
            plt.imsave("pred_depth.png", np.transpose(np_imagegrid, (1, 2, 0)))
            # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset directory
    parser.add_argument('--data-dir', type=str, help='path to dataset directory', default="/home/arjun/Desktop/spring23/vlr/project/DPE/data/rgbd_dataset_freiburg2_pioneer_360")
    parser.add_argument('--batch-size', type=str, help='batch size', default=20)

    args = parser.parse_args()
    dataset_dir = args.data_dir
    batch_size = args.batch_size

    rgb_txt = dataset_dir + "/rgb.txt"
    depth_txt = dataset_dir + "/depth.txt"
    rgb_dir = dataset_dir 
    depth_dir = dataset_dir

    # load dataset
    dataset = DepthDataset(rgb_img_txt=rgb_txt, depth_img_txt=depth_txt, rgb_img_dir=rgb_dir, depth_img_dir=depth_dir)
    # train-test split
    dataset_len = dataset.__len__()
    test_split = int(0.2 * dataset_len)
    lengths = [(dataset_len - test_split), test_split ] # [train ratio, test ratio]
    train_dataset, test_dataset = random_split(dataset, lengths)

    train_dataloader =  DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    test_dataloader =  DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)

    model = DepthPredictor()
    trainer = Trainer(model, train_dataloader, test_dataloader, batch_size=batch_size)
    trainer.train()
    trainer.eval()