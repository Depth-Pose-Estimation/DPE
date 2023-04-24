import torch
import torchvision
import torchgeometry as tgm
import argparse
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from dataloader import DepthDataset, DepthDatasetKitti
from model import DepthPosePredictor
from unet.unet_model import UNet
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

class Trainer:
    def __init__(self, model, train_dataloader, test_dataloader, learning_rate = 1e-3, batch_size = 100, 
                    num_epochs = 10, scheduler = None, device = 'cuda'):

        print(f"[INFO] Training for {num_epochs} epochs")
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        self.val_every = 5

        self.optim = torch.optim.Adam(self.model.parameters(), self.learning_rate)
        self.scheduler = scheduler
    
    def loss(self, model_out, depth_imgs_gt):
        loss_fn = torch.nn.MSELoss()
        # loss_fn = torch.nn.CrossEntropyLoss()
        loss_val = loss_fn(model_out, depth_imgs_gt)

        # mask = (depth_imgs_gt != 0)
        # loss_val = loss_fn(model_out[mask], depth_imgs_gt[mask])

        return loss_val
    
    def depth_smoothness_loss(self, depth_img, rgb_img):
        # smooth = tgm.losses.DepthSmoothnessLoss()
        smooth = tgm.losses.InverseDepthSmoothnessLoss()
        loss = smooth(depth_img, rgb_img)
        return loss
    
    def smooth_loss(self, pred_map):
        def gradient(pred):
            D_dy = pred[:, :, 1:] - pred[:, :, :-1]
            D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            return D_dx, D_dy

        if type(pred_map) not in [tuple, list]:
            pred_map = [pred_map]

        loss = 0
        weight = 1.

        for scaled_map in pred_map:
            dx, dy = gradient(scaled_map)
            dx2, dxdy = gradient(dx)
            dydx, dy2 = gradient(dy)
            loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())*weight
            weight /= 2.3  # don't ask me why it works better
        return loss

    def train(self):


        self.model.to(self.device)
        for i in range(self.num_epochs):
            train_loss = 0
            for rgb_imgs_t, depth_imgs_gt in self.train_dataloader:
                rgb_imgs_t = rgb_imgs_t.to(self.device)
                depth_imgs_gt = depth_imgs_gt.to(self.device)
                depth_out = self.model(rgb_imgs_t) #, xt = rgb_imgs_t)

                mse_loss = self.loss(depth_out, depth_imgs_gt)
                depth_loss = self.depth_smoothness_loss(depth_out, rgb_imgs_t)
                # smooth_loss = self.smooth_loss(depth_out)
                loss = depth_loss + mse_loss
                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                train_loss += loss.item()

            if i % self.val_every == 0:
                self.eval(i)

            train_loss /= len(self.train_dataloader)
            
            print(f"EPOCH {i + 1} [TRAIN LOSS] {train_loss}")
            writer.add_scalar('Loss/train', train_loss, i)

            if self.scheduler is not None:
                self.scheduler.step()

        return train_loss
    
    def eval(self,i = None):
        val_loss = 0
        self.model.eval()

        with torch.no_grad():
            for rgb_imgs_t, depth_imgs_gt in self.test_dataloader:
                rgb_imgs_t = rgb_imgs_t.to(self.device)
                # rgb_imgs_t1 = rgb_imgs_t1.to(self.device)
                depth_imgs_gt = depth_imgs_gt.to(self.device)
                depth_out = self.model(rgb_imgs_t)#, xt = rgb_imgs_t)

                loss = self.loss(depth_out, depth_imgs_gt)
                val_loss += loss.item()
            val_loss /= len(self.test_dataloader)

            print(f"[VAL LOSS] {val_loss}")
            writer.add_scalar('Loss/val', val_loss, i)

            torch.save(model.state_dict(), "best.pth")

        images, depth = iter(self.test_dataloader).next()
        self.show_image(torchvision.utils.make_grid(images[1:10],10,1), torchvision.utils.make_grid(depth[1:10],10,1))
        # plt.show()
        # plt.figure()

        self.visualise_output(images, self.model, i)

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
        npdepth /= np.max(npdepth)
        print(f"[INFO] Saving gt plot")
        plt.imsave("gt_depth.png", np.transpose(npdepth, (1, 2, 0)))


    def visualise_output(self, images, model, i = None):

        with torch.no_grad():
        
            images = images.to(self.device)
            images = model(images)
            images = images.cpu()
            # images /= torch.max(images)
            # images = self.to_img(images)
            np_imagegrid = torchvision.utils.make_grid(images[1:10], 10, 1).numpy()
            if i == None: idx = 0 
            else: idx = i
            writer.add_image('Predicted Depth', np_imagegrid, idx)
            print(f"[INFO] Saving pred plot")
            np_imagegrid /= np.max(np_imagegrid)
            plt.imsave("pred_depth.png", np.transpose(np_imagegrid, (1, 2, 0)))
            # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset directory
    parser.add_argument('--data-dir', type=str, help='path to dataset directory', default="/home/arjun/Desktop/spring23/vlr/project/DPE/data/rgbd_dataset_freiburg2_pioneer_360")
    parser.add_argument('--batch-size', type=int, help='batch size', default=10)
    parser.add_argument('--epoch', type=int, help='epoch', default=200)

    args = parser.parse_args()
    dataset_dir = args.data_dir
    batch_size = args.batch_size

    rgb_txt = dataset_dir + "/rgb.txt"
    depth_txt = dataset_dir + "/depth.txt"
    rgb_dir = dataset_dir 
    depth_dir = dataset_dir

    # load dataset
    # dataset = DepthDatasetKitti(rgb_img_txt=rgb_txt, depth_img_txt=depth_txt, rgb_img_dir=rgb_dir, depth_img_dir=depth_dir)
    # train-test split
    # dataset_len = dataset.__len__()
    # test_split = int(0.2 * dataset_len)
    # lengths = [(dataset_len - test_split), test_split ] # [train ratio, test ratio]
    # train_dataset, test_dataset = random_split(dataset, lengths)


    train_dataset = DepthDatasetKitti(split="train", data_dir = "/home/arjun/Desktop/vlr_project/data/dump")
    test_dataset = DepthDatasetKitti(split="val", data_dir = "/home/arjun/Desktop/vlr_project/data/dump")
    train_dataloader =  DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=10, pin_memory=True)
    test_dataloader =  DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=10, pin_memory=True)

    # model = DepthPosePredictor()
    model = UNet(n_channels=1, n_classes=1)
    trainer = Trainer(model, train_dataloader, test_dataloader, batch_size=batch_size, num_epochs= args.epoch)
    trainer.train()
    trainer.eval()