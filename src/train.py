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
        pass
    
    def train(self):
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
    parser.add_argument('--data-dir', required=True, type=str, help='path to dataset directory')
    parser.add_argument('--batch-size', required=True, type=str, help='batch size')

    args = parser.parse_args()
    dataset_dir = args.data_dir
    batch_size = args.batch_size

    rgb_txt = dataset_dir + "/rgb.txt"
    depth_txt = dataset_dir + "/depth.txt"
    rgb_dir = dataset_dir + "/rgb/"
    depth_dir = dataset_dir + "/depth/"

    # load dataset
    dataset = DepthDataset(rgb_img_txt=rgb_txt, depth_img_txt=depth_txt, rgb_img_dir=rgb_dir, depth_img_dir=depth_dir)
    # train-test split
    test_split = 0.2
    lengths = [1 - test_split, test_split] # [train ratio, test ratio]
    train_dataset, test_dataset = random_split(dataset, lengths)

    train_dataloader =  DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader =  DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = DepthPredictor()
    trainer = Trainer(model, train_dataloader, test_dataloader, batch_size=batch_size)
    # trainer.train()