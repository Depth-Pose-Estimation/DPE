import torch

class DepthPredictor(torch.nn.Module):
    def __init__(self):
        super(DepthPredictor, self).__init__()

        self.dconv1 = torch.nn.Conv2d(in_channels = 3  , out_channels = 64, kernel_size = 3, stride = 2, padding = 1)
        self.dconv2 = torch.nn.Conv2d(in_channels = 64 , out_channels = 128, kernel_size = 3, stride = 2, padding = 1)
        self.dconv3 = torch.nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 2, padding = 1)
        self.dconv4 = torch.nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 2, padding = 1)

        self.uconv1 = torch.nn.ConvTranspose2d(in_channels =  512, out_channels = 256, kernel_size = 2, stride = 2)#, padding = 1) 
        self.uconv2 = torch.nn.ConvTranspose2d(in_channels =  256, out_channels = 128, kernel_size = 2, stride = 2)#, padding = 1)
        self.uconv3 = torch.nn.ConvTranspose2d(in_channels =  128, out_channels = 64, kernel_size = 2, stride = 2)#, padding = 1)
        self.uconv4 = torch.nn.ConvTranspose2d(in_channels =  64, out_channels = 1, kernel_size = 2, stride = 2)#, padding = 1)

        self.act1   = torch.nn.ReLU()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):

        # Currently an AE network
        # TODO : UNet-esque network architecture
        # TODO : Take inspiration from the depth N/W

        l1 = self.dconv1(x)
        l1 = self.act1(l1)

        l2 = self.dconv2(l1)
        l2 = self.act1(l2)

        l3 = self.dconv3(l2)
        l3 = self.act1(l3)

        l4 = self.dconv4(l3)
        l4 = self.act1(l4)

        l5 = self.uconv1(l4)
        l5 = self.act1(l5)

        l6 = self.uconv2(l5)
        l6 = self.act1(l6)

        l7 = self.uconv3(l6)
        l7 = self.act1(l7)

        l8 = self.uconv4(l7)
        l8 = self.act1(l8)

        return l8

############################# TEST CODE ###########################################
# https://thanos.charisoudis.gr/blog/a-simple-conv2d-dimensions-calculator-logger
# x = torch.rand(12, 3, 640, 320)
# model = DepthPredictor()
# y = model(x)
# print(y.shape)