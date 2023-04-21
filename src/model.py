import torch

class UNetFwd(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super (UNetFwd, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels = in_channels,  out_channels = out_channels, kernel_size = 3, stride = 1, padding = 'same')
        self.conv2 = torch.nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 'same')
        self.act1  = torch.nn.ReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act1(x)

        return x

class DepthPredictor(torch.nn.Module):
    def __init__(self):
        super(DepthPredictor, self).__init__()

        self.channels = [3, 64, 128, 256, 512, 1024]
        self.down_block1 = UNetFwd(in_channels= self.channels[0], out_channels = self.channels[1])
        self.down_block2 = UNetFwd(in_channels= self.channels[1], out_channels = self.channels[2])
        self.down_block3 = UNetFwd(in_channels= self.channels[2], out_channels = self.channels[3])
        self.down_block4 = UNetFwd(in_channels= self.channels[3], out_channels = self.channels[4])
        self.down_block5 = UNetFwd(in_channels= self.channels[4], out_channels = self.channels[5])

        self.depth_block1 = UNetFwd(in_channels= self.channels[4], out_channels = self.channels[3])
        self.depth_block2 = UNetFwd(in_channels= self.channels[3], out_channels = self.channels[2])
        self.depth_block3 = UNetFwd(in_channels= self.channels[2], out_channels = self.channels[1])
        self.depth_block4 = UNetFwd(in_channels= self.channels[1], out_channels = 1)

        self.pose_block1  = UNetFwd(in_channels= self.channels[4], out_channels = self.channels[3])
        self.pose_block2  = UNetFwd(in_channels= self.channels[3], out_channels = self.channels[2])
        self.pose_block3  = UNetFwd(in_channels= self.channels[2], out_channels = self.channels[1])
        self.pose_block4  = UNetFwd(in_channels= self.channels[1], out_channels = 1)
        self.pose_mp_pose = torch.nn.AdaptiveMaxPool2d(6)

        self.mp1    = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.mp2    = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.depth_uconv1 = torch.nn.ConvTranspose2d(in_channels =  self.channels[-1], out_channels = self.channels[-2], kernel_size = 2, stride = 2) 
        self.depth_uconv2 = torch.nn.ConvTranspose2d(in_channels =  self.channels[-2], out_channels = self.channels[-3], kernel_size = 2, stride = 2)
        self.depth_uconv3 = torch.nn.ConvTranspose2d(in_channels =  self.channels[-3], out_channels = self.channels[-4], kernel_size = 2, stride = 2)
        # self.depth_uconv4 = torch.nn.ConvTranspose2d(in_channels =  self.channels[-4], out_channels = self.channels[-5], kernel_size = 2, stride = 2)

        self.pose_uconv1 = torch.nn.ConvTranspose2d(in_channels =  self.channels[-1], out_channels = self.channels[-2], kernel_size = 2, stride = 2) 
        self.pose_uconv2 = torch.nn.ConvTranspose2d(in_channels =  self.channels[-2], out_channels = self.channels[-3], kernel_size = 2, stride = 2)
        self.pose_uconv3 = torch.nn.ConvTranspose2d(in_channels =  self.channels[-3], out_channels = self.channels[-4], kernel_size = 2, stride = 2)
        # self.pose_uconv4 = torch.nn.ConvTranspose2d(in_channels =  self.channels[-4], out_channels = self.channels[-5], kernel_size = 2, stride = 2)

        self.act1   = torch.nn.ReLU()
        self.act2   = torch.nn.Sigmoid()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):

        # Currently an AE network
        # IN PROGRESS : UNet-esque network architecture

        f1 = self.down_block1(x)
        f1_down = self.mp1(f1)

        f2 = self.down_block2(f1_down)
        f2_down = self.mp1(f2)

        f3 = self.down_block2(f2_down)
        f3_down = self.mp1(f3)

        f4 = self.down_block2(f3_down)
        f4_down = self.mp1(f4)

        f5 = self.down_block5(f4_down)
        f5_up = self.uconv1(f5)

        # Depth Branch
        d1 = self.depth_block1(torch.cat((f5_up, f4), dim = 1))
        d1_up = self.depth_uconv1(d1)

        d2 = self.depth_block2(torch.cat(f3, d1_up), dim = 1)
        d2_up = self.depth_uconv2(d2)

        d3 = self.depth_block3(torch.cat(f2, d2_up), dim = 1)
        d3_up = self.depth_uconv2(d3)

        d4 = self.depth_block4(torch.cat(f1, d3_up), dim = 1)
        # d4_up = self.depth_uconv2(d4)


        return None

############################# TEST CODE ###########################################
# https://thanos.charisoudis.gr/blog/a-simple-conv2d-dimensions-calculator-logger
# x = torch.rand(12, 3, 640, 320)
# model = DepthPredictor()
# y = model(x)
# print(y.shape)