import torch
import torchvision
import torchgeometry as tgm
import argparse
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from dataloader import DepthDataset, DepthDatasetKitti, DepthPoseDatasetKitti
# from model import DepthPosePredictor
from unet.unet_model import UNet
import matplotlib.pyplot as plt
import numpy as np
from utils import *
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
        loss_val = loss_fn(model_out, depth_imgs_gt)
        # mask = (depth_imgs_gt != 0)
        # loss_val = loss_fn(model_out[mask], depth_imgs_gt[mask])
        return loss_val
    
    def reproj_loss(self, source_imgs, target_imgs, pose_out, depth_out, intrinsics):
        warped_imgs, valid_pts = inverse_warp(source_imgs, pose_out, depth_out, intrinsics)
        reproj_loss = (target_imgs - warped_imgs) * valid_pts.unsqueeze(1).float()
        reproj_loss = reproj_loss.abs().mean(dim=(-2, -1))
        # average over batch later - so sum now
        reproj_loss = reproj_loss.sum()

        return reproj_loss
    
    def pose_loss(self, pose_gt, pose_out):

        R_gt, trans_gt = pose_gt[:, :, :-1], pose_gt[:, :, -1]
        trans_out, quat_out = pose_out[:, :3], pose_out[:, 3:]
        R_out = quat_to_rotMat(quat_out)
        R_out_T_R_gt = torch.transpose(R_out, dim0=1, dim1=2) @ R_gt

        R_norm = torch.linalg.norm((R_out_T_R_gt - torch.eye(3, device=pose_gt.device).unsqueeze(0)), dim=(-2, -1))
        trans_norm = torch.linalg.norm((trans_gt - trans_out), dim=-1)

        # average over batch later - so sum now
        loss = R_norm.abs().mean() + trans_norm.abs().mean()

        return loss

    def depth_smoothness_loss(self, depth_img, rgb_img):
        smooth = tgm.losses.InverseDepthSmoothnessLoss()
        loss = smooth(depth_img, rgb_img)
        return loss

    def train(self):

        self.model.to(self.device)
        for i in range(self.num_epochs):
            train_loss = 0
            for rgb_imgs_t, rgb_imgs_tPlus1, depth_imgs_gt, pose_gt, intrinsics in self.train_dataloader:

                b, c, h, w = rgb_imgs_t.shape

                rgb_imgs_t = rgb_imgs_t.to(self.device)
                rgb_imgs_tPlus1 = rgb_imgs_tPlus1.to(self.device)
                depth_imgs_gt = depth_imgs_gt.to(self.device)
                intrinsics = intrinsics.to(self.device)
                pose_gt = pose_gt.to(self.device)

                model_out = self.model(xt=rgb_imgs_t, xt1 = rgb_imgs_tPlus1)
                depth_out = model_out[:, :-7]
                depth_out = depth_out.reshape(b, c, h, w)
                pose_out = model_out[:, -7:]

                mse_loss = self.loss(depth_out, depth_imgs_gt)
                depth_loss = self.depth_smoothness_loss(depth_out, rgb_imgs_t)
                # reproj_loss = self.reproj_loss(source_imgs=rgb_imgs_t,
                #                                target_imgs=rgb_imgs_tPlus1,
                #                                pose_out=pose_out,
                #                                depth_out=depth_out,
                #                                intrinsics=intrinsics)
                pose_loss = self.pose_loss(pose_gt, pose_out)

                loss = pose_loss + depth_loss + mse_loss # + reproj_loss

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                train_loss += loss.item()


            train_loss /= len(self.train_dataloader)

            print(f"EPOCH {i + 1} [TRAIN LOSS] {train_loss}")
            writer.add_scalar('Loss/train', train_loss, i)

            if (i + 1) % self.val_every == 0:
                self.eval(i + 1)

            if self.scheduler is not None:
                self.scheduler.step()

        return train_loss
    
    def eval(self,i = None):
        val_loss = 0
        self.model.eval()

        with torch.no_grad():
            for rgb_imgs_t, rgb_imgs_tPlus1, depth_imgs_gt, pose_gt, intrinsics in self.test_dataloader:

                b, c, h, w = rgb_imgs_t.shape

                rgb_imgs_t = rgb_imgs_t.to(self.device)
                rgb_imgs_tPlus1 = rgb_imgs_tPlus1.to(self.device)
                depth_imgs_gt = depth_imgs_gt.to(self.device)
                intrinsics = intrinsics.to(self.device)
                pose_gt = pose_gt.to(self.device)

                model_out = self.model(xt=rgb_imgs_t, xt1 = rgb_imgs_tPlus1)
                depth_out = model_out[:, :-7]
                depth_out = depth_out.reshape(b, c, h, w)
                pose_out = model_out[:, -7:]

                mse_loss = self.loss(depth_out, depth_imgs_gt)
                depth_loss = self.depth_smoothness_loss(depth_out, rgb_imgs_t)
                # reproj_loss = self.reproj_loss(source_imgs=rgb_imgs_t,
                #                                target_imgs=rgb_imgs_tPlus1,
                #                                pose_out=pose_out,
                #                                depth_out=depth_out,
                #                                intrinsics=intrinsics)
                pose_loss = self.pose_loss(pose_gt, pose_out)

                val_loss += (pose_loss.item() + depth_loss.item() + mse_loss.item()) # + reproj_loss.item()

            val_loss /= len(self.test_dataloader)

            print(f"[VAL LOSS] {val_loss}")
            writer.add_scalar('Loss/val', val_loss, i)

            torch.save(model.state_dict(), "best.pth")

        images, images_t1, depth, pose_gt, intrinsics = iter(self.test_dataloader).next()
        self.show_image(torchvision.utils.make_grid(images[1:10],10,1), torchvision.utils.make_grid(depth[1:10],10,1))
        # plt.show()
        # plt.figure()

        self.visualise_output(self.model, images, images_t1, i)

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


    def visualise_output(self, model, images, images_t1, i = None):

        with torch.no_grad():

            images = images.to(self.device)
            images_t1 = images_t1.to(self.device)

            predicted = model(images, images_t1)
            predicted = predicted.cpu()

            pred_depth = predicted[:, :-7]
            pred_pose = predicted[:, -7:]

            trans_out, quat_out = pred_pose[:, :3], pred_pose[:, 3:]
            R_out = quat_to_rotMat(quat_out)
            pred_pose = torch.cat((R_out, trans_out.unsqueeze(-1)), dim=-1) # (B x 3 x 4)

            pred_depth = pred_depth.view(images.shape)
            np_imagegrid = torchvision.utils.make_grid(pred_depth[1:10], 10, 1).numpy()
            # if i == None: idx = 0
            # else: idx = i
            print(f"[INFO] Saving pred plot")
            np_imagegrid /= np.max(np_imagegrid)
            plt.imsave("pred_depth.png", np.transpose(np_imagegrid, (1, 2, 0)))

            #TODO: Log pred_pose in a .txt for comparison

            # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset directory
    parser.add_argument('--data-dir', type=str, help='path to dataset directory', default="/home/arjun/Desktop/spring23/vlr/project/DPE/data/rgbd_dataset_freiburg2_pioneer_360")
    parser.add_argument('--batch-size', type=int, help='batch size', default=3)
    parser.add_argument('--epoch', type=int, help='epoch', default=2)

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


    train_dataset = DepthPoseDatasetKitti(split="train", data_dir = "/home/arjun/Desktop/vlr_project/data/dump")
    test_dataset = DepthPoseDatasetKitti(split="val", data_dir = "/home/arjun/Desktop/vlr_project/data/dump")
    train_dataloader =  DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=10, pin_memory=True)
    test_dataloader =  DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=10, pin_memory=True)

    # model = DepthPosePredictor()
    model = UNet(n_channels=1, n_classes=1)
    model = torch.nn.DataParallel(model)
    trainer = Trainer(model, train_dataloader, test_dataloader, batch_size=batch_size, num_epochs= args.epoch)
    trainer.train()
    trainer.eval()