import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from unet.unet_model import UNet
from dataloader import DepthPoseDatasetKitti
from utils import *

batch_size = 3

model = UNet(n_channels=1, n_classes=1)
model = torch.nn.DataParallel(model) # did this since server trained on all GPUs
test_dataset = DepthPoseDatasetKitti(split = "val", data_dir = "/home/arjun/Desktop/vlr_project/data/dump")
test_dataloader =  torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=10, pin_memory=True)

checkpoint = torch.load("best_server_combined.pth")
model.load_state_dict(checkpoint)

val_loss = 0
model.eval()
device = 'cuda'

def show_image(img, depth = None):
    # img = self.to_img(img)
    npimg = img.numpy()
    print(f"[INFO] Saving gt img plot")
    plt.imsave("img.png", np.transpose(npimg, (1, 2, 0)))

    npdepth = depth.numpy()
    npdepth /= np.max(npdepth)
    print(f"[INFO] Saving gt plot")
    plt.imsave("gt_depth.png", np.transpose(npdepth, (1, 2, 0)))

def visualise_output(model,images, images_t = None,  i = None):

    with torch.no_grad():
        images = images.to(device)
        predicted = model(images, images_t)
        predicted = predicted.cpu()
        pred_depth = predicted[:, :-7]
        pred_pose = predicted[:, -7:]

        pred_depth = pred_depth.view(images.shape)
        np_imagegrid = torchvision.utils.make_grid(pred_depth[1:10], 10, 1).numpy()
        if i == None: idx = 0 
        else: idx = i
        print(f"[INFO] Saving pred plot")
        np_imagegrid /= np.max(np_imagegrid)
        plt.imsave("pred_depth.png", np.transpose(np_imagegrid, (1, 2, 0)))

def loss_mse(model_out, depth_imgs_gt):
    loss_fn = torch.nn.MSELoss()
    # loss_fn = torch.nn.CrossEntropyLoss()
    loss_val = loss_fn(model_out, depth_imgs_gt)

    # mask = (depth_imgs_gt != 0)
    # loss_val = loss_fn(model_out[mask], depth_imgs_gt[mask])

    return loss_val

def reproj_loss(source_imgs, target_imgs, pose_out, depth_out):
    warped_imgs, valid_pts = inverse_warp(source_imgs, pose_out, depth_out, self.intrinsic_mat)
    reproj_loss = (target_imgs - warped_imgs) * valid_pts.unsqueeze(1).float()
    reproj_loss = reproj_loss.abs().mean()
    return reproj_loss

f = open("pred_poses.txt", 'a')
with torch.no_grad():
    for rgb_imgs_t, rgb_imgs_tPlus1, depth_imgs_gt, pose_gt, cam_intrinsics in test_dataloader:
        b, c, h, w = rgb_imgs_t.shape
        rgb_imgs_t = rgb_imgs_t.to(device)
        rgb_imgs_tPlus1 = rgb_imgs_tPlus1.to(device)
        depth_imgs_gt = depth_imgs_gt.to(device)
        model_out = model(xt=rgb_imgs_t, xt1 = rgb_imgs_tPlus1)
        depth_out = model_out[:, :-7]
        depth_out = depth_out.reshape(b, c, h, w)
        pose_out = model_out[:, -7:]

        np.savetxt(f, pose_out.detach().cpu().numpy())
        loss = loss_mse(depth_out, depth_imgs_gt)
        #reproj_loss = reproj_loss(source_imgs=rgb_imgs_t, target_imgs=rgb_imgs_tPlus1, pose_out=pose_out, depth_out=depth_out)
        val_loss += (loss.item())# + reproj_loss.item())
    val_loss /= len(test_dataloader)

    print(f"[VAL LOSS] {val_loss}")


images, images_t, depth, pose = iter(test_dataloader).next()
show_image(torchvision.utils.make_grid(images[1:10],10,1), torchvision.utils.make_grid(depth[1:10],10,1))
# plt.show()
# plt.figure()

visualise_output(model, images, images_t)