""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))

        self.up1_depth = (Up(1024, 512 // factor, bilinear))
        self.up2_depth = (Up(512, 256 // factor, bilinear))
        self.up3_depth = (Up(256, 128 // factor, bilinear))
        self.up4_depth = (Up(128, 64, bilinear))
        self.outc_depth = (OutConv(64, n_classes))

        self.up1_pose = (Up(2048, 512 // factor, bilinear))
        self.up2_pose = (Up(512, 256 // factor, bilinear))
        self.up3_pose = (Up(256, 128 // factor, bilinear))
        self.up4_pose = (Up(128, 64, bilinear))
        self.outc_pose = (OutConv(64, 7))
        self.mp = torch.nn.AdaptiveAvgPool2d(1)

        self.act1 = torch.nn.ReLU()

    def forward(self, xt, xt1 = None):
        x1 = self.inc(xt)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if xt1 is not None:
            x1t = self.inc(xt1)
            x2t = self.down1(x1t)
            x3t = self.down2(x2t)
            x4t = self.down3(x3t)
            x5t = self.down4(x4t)
        
        x_depth = self.up1_depth(x5, x4)
        x_depth = self.up2_depth(x_depth, x3)
        x_depth = self.up3_depth(x_depth, x2)
        x_depth = self.up4_depth(x_depth, x1)
        logits_depth = self.outc_depth(x_depth)
        logits_depth = self.act1(logits_depth)  # B x 1 x h x w

        if xt1 is not None:
            x_pose = self.up1_pose(torch.cat((x5, x5t), dim = 1), torch.cat((x4, x4t), dim = 1))
            x_pose = self.up2_pose(x_pose, (x3 + x3t))
            x_pose = self.up3_pose(x_pose, (x2 + x2t))
            x_pose = self.up4_pose(x_pose, (x1 + x1t))
            logits_pose = self.outc_pose(x_pose)
            logits_pose = self.act1(logits_pose)
            logits_pose = self.mp(logits_pose)     #  B x 6 x 1 x 1

            logits = torch.cat((logits_depth.view(logits_depth.shape[0], -1), logits_pose.view(logits_pose.shape[0], -1)), dim = -1) 
        else :
            logits = logits_depth.view(logits_depth.shape[0], -1)

        # x_pose = self.up1_pose(x5, x4)
        # x_pose = self.up2_pose(x_pose, x3)
        # x_pose = self.up3_pose(x_pose, x2)
        # x_pose = self.up4_pose(x_pose, x1)
        # logits_pose = self.outc_pose(x_pose)
        # logits_pose = self.act1(logits_pose)  # B x 1 x h x w
        # logits = torch.cat((logits_depth.view(logits_depth.shape[0], -1), logits_pose.view(logits_pose.shape[0], -1)), dim = -1) 

        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

# from torchviz import make_dot
# model = UNet(1, 1)
# x = torch.rand(23, 1, 128, 128)
# y = model(x, x)
# dot = make_dot(y.mean(), params=dict(model.named_parameters()))
# dot.format = 'png'
# dot.render('combined')

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter("torchlogs/")
# writer.add_graph(model, x)
# writer.close()