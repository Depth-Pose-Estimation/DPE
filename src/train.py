import torch
import argparse
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from dataloader import DepthDataset
from model import DepthPredictor

class Trainer:
    def __init__(self, model, train_dataloader, test_dataloader, learning_rate = 0.001, batch_size = 100, 
                    num_epochs = 10, scheduler = None, device = 'cuda'):
      
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
        loss_val = loss_fn(model_out, depth_imgs_gt)
        return loss_val
    
    def train(self):
        self.model.to(self.device)
        for i in range(self.num_epochs):

            for rgb_imgs, depth_imgs_gt in self.train_dataloader:
                rgb_imgs = rgb_imgs.to(self.device)
                depth_imgs_gt = depth_imgs_gt.to(self.device)
                model_out = self.model(rgb_imgs)

                loss = self.loss(model_out, depth_imgs_gt)
                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
            if self.scheduler is not None:
                self.scheduler.step()
    
    def eval(self, test_dataloader):
        self.model.eval()

        for rgb_imgs, depth_imgs_gt in test_dataloader:
            pass
        
        self.model.train()
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset directory
    parser.add_argument('--data-dir', type=str, help='path to dataset directory', default="/home/arjun/Desktop/spring23/vlr/project/DPE/data/rgbd_dataset_freiburg2_pioneer_360")
    parser.add_argument('--batch-size', type=str, help='batch size', default=32)

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

    train_dataloader =  DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader =  DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = DepthPredictor()
    trainer = Trainer(model, train_dataloader, test_dataloader, batch_size=batch_size)
    trainer.train()