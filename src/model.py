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

class DepthPosePredictor(torch.nn.Module):
    def __init__(self):
        super(DepthPosePredictor, self).__init__()

        self.channels = [3, 64, 128, 256, 512, 1024]
        self.down_block1 = UNetFwd(in_channels= self.channels[0], out_channels = self.channels[1])
        self.down_block2 = UNetFwd(in_channels= self.channels[1], out_channels = self.channels[2])
        self.down_block3 = UNetFwd(in_channels= self.channels[2], out_channels = self.channels[3])
        self.down_block4 = UNetFwd(in_channels= self.channels[3], out_channels = self.channels[4])
        self.down_block5 = UNetFwd(in_channels= self.channels[4], out_channels = self.channels[5])

        self.depth_block1 = UNetFwd(in_channels= self.channels[5], out_channels = self.channels[4])
        self.depth_block2 = UNetFwd(in_channels= self.channels[4], out_channels = self.channels[3])
        self.depth_block3 = UNetFwd(in_channels= self.channels[3], out_channels = self.channels[2])
        self.depth_block4 = UNetFwd(in_channels= self.channels[2], out_channels = 1)

        self.pose_block1  = UNetFwd(in_channels= self.channels[5], out_channels = self.channels[4])
        self.pose_block2  = UNetFwd(in_channels= self.channels[4], out_channels = self.channels[3])
        self.pose_block3  = UNetFwd(in_channels= self.channels[3], out_channels = self.channels[2])
        self.pose_block4  = UNetFwd(in_channels= self.channels[2], out_channels = 6)
        self.pose_mp_pose = torch.nn.AdaptiveMaxPool2d(1)
        # self.pose_mp_pose = torch.nn.AvgPool2d(6)


        self.mp1    = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.mp2    = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.depth_uconv1 = torch.nn.ConvTranspose2d(in_channels =  self.channels[-1], out_channels = self.channels[-2], kernel_size = 2, stride = 2) 
        self.depth_uconv2 = torch.nn.ConvTranspose2d(in_channels =  self.channels[-2], out_channels = self.channels[-3], kernel_size = 2, stride = 2)
        self.depth_uconv3 = torch.nn.ConvTranspose2d(in_channels =  self.channels[-3], out_channels = self.channels[-4], kernel_size = 2, stride = 2)
        self.depth_uconv4 = torch.nn.ConvTranspose2d(in_channels =  self.channels[-4], out_channels = self.channels[-5], kernel_size = 2, stride = 2)

        self.pose_uconv1 = torch.nn.ConvTranspose2d(in_channels =  self.channels[-1], out_channels = self.channels[-2], kernel_size = 2, stride = 2) 
        self.pose_uconv2 = torch.nn.ConvTranspose2d(in_channels =  self.channels[-2], out_channels = self.channels[-3], kernel_size = 2, stride = 2)
        self.pose_uconv3 = torch.nn.ConvTranspose2d(in_channels =  self.channels[-3], out_channels = self.channels[-4], kernel_size = 2, stride = 2)
        self.pose_uconv4 = torch.nn.ConvTranspose2d(in_channels =  self.channels[-4], out_channels = self.channels[-5], kernel_size = 2, stride = 2)

        self.act1   = torch.nn.ReLU()
        self.act2   = torch.nn.Sigmoid()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):

        # Multi head UNet Architecture for depth and pose estimation

        ################### ENCODER ############################################
        # X : [ B X 3 X 640 X 320]
        f1 = self.down_block1(x)            # B x 64 X 640 X 320
        f1_down = self.mp1(f1)              # b x 64 x 320 x 160

        f2 = self.down_block2(f1_down)      # B x 128 X 320 X 160
        f2_down = self.mp1(f2)              # B x 128 X 160 X 80

        f3 = self.down_block3(f2_down)      # B x 256 X 160 X 80
        f3_down = self.mp1(f3)              # B x 256 X 80 X 40

        f4 = self.down_block4(f3_down)      # B x 512 X 80 X 40
        f4_down = self.mp1(f4)              # B x 512 X 40 X 20

        f5 = self.down_block5(f4_down)      # B x 1024 X 40 X 20

        ################### DEPTH BRANCH #########################################
        f5_depth_up = self.depth_uconv1(f5)                           # B x 512 X 80 X 40
        d1 = self.depth_block1(torch.cat((f5_depth_up, f4), dim = 1)) # B x 512 X 80 X 40
        d1_up = self.depth_uconv2(d1)                                 # B x 512 X 160 X 80

        d2 = self.depth_block2(torch.cat((f3, d1_up), dim = 1))       # B x 256 x 160 x 80
        d2_up = self.depth_uconv3(d2)                                 # B x 128 x 320 x 160

        d3 = self.depth_block3(torch.cat((f2, d2_up), dim = 1))       # B x 128 x 320 x 160
        d3_up = self.depth_uconv4(d3)                                 # B x 64 x 640 x 320

        d4 = self.depth_block4(torch.cat((f1, d3_up), dim = 1))       # B x 1 x 640 x 320

        ################### POSE BRANCH #########################################

        f5_pose_up = self.pose_uconv1(f5)                           # B x 512 X 80 X 40
        p1 = self.pose_block1(torch.cat((f5_pose_up, f4), dim = 1)) # B x 512 X 80 X 40
        p1_up = self.pose_uconv2(p1)                                 # B x 512 X 160 X 80

        p2 = self.pose_block2(torch.cat((f3, p1_up), dim = 1))       # B x 256 x 160 x 80
        p2_up = self.depth_uconv3(p2)                                # B x 128 x 320 x 160

        p3 = self.pose_block3(torch.cat((f2, p2_up), dim = 1))       # B x 128 x 320 x 160
        p3_up = self.depth_uconv4(p3)                                 # B x 64 x 640 x 320

        p4 = self.pose_block4(torch.cat((f1, p3_up), dim = 1))       # B x 1 x 640 x 320
        p4 = self.pose_mp_pose(p4)                                   # B x 6 x 1 x 1
        p4 = p4.view(-1, 6)                                          # B x 6

        return d4, p4

############################# TEST CODE ###########################################
# https://thanos.charisoudis.gr/blog/a-simple-conv2d-dimensions-calculator-logger
x = torch.rand(12, 3, 640, 320)
model = DepthPosePredictor()
depth, pose = model(x)
print(f"depth : {depth.shape}, pose : {pose.shape}")