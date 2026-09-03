import torch
import torch.nn as nn
from . import block as B
def make_model(args, parent=False):
    model = LiquidSR()
    return model


class LiquidSR(nn.Module):
    def __init__(self, in_nc=3, nf=55, num_modules=6, out_nc=3, upscale=3):
        super(LiquidSR, self).__init__()
        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)

        self.B1 = B.DSLNBB(in_channels=nf)
        self.B2 = B.DSLNBB(in_channels=nf)
        self.B3 = B.DSLNBB(in_channels=nf)
        self.B4 = B.DSLNBB(in_channels=nf)
        self.B5 = B.DSLNBB(in_channels=nf)
        self.B6 = B.DSLNBB(in_channels=nf)
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')
        self.c1 = B.conv_block(nf * 2, nf, kernel_size=1, act_type='lrelu')
        self.c2 = B.conv_block(nf * 3, nf, kernel_size=1, act_type='lrelu')
        self.c3 = B.conv_block(nf * 4, nf, kernel_size=1, act_type='lrelu')
        self.c4 = B.conv_block(nf * 5, nf, kernel_size=1, act_type='lrelu')
        self.c5 = B.conv_block(nf * 6, nf, kernel_size=1, act_type='lrelu')
        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)

        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)
        self.scale_idx = 0


    def forward(self, input):
        out_fea = self.fea_conv(input)
        
        out_B1 = self.B1(out_fea)
        out_B10 = torch.cat([out_fea , out_B1], dim=1)
        out_B11 = self.c1(out_B10)
        
        out_B2 = self.B2(out_B11)
        out_B20 = torch.cat([out_B10 , out_B2], dim=1)
        out_B22 = self.c2(out_B20)
        
        out_B3 = self.B3(out_B22)
        out_B30 = torch.cat([out_B20 , out_B3], dim=1)
        out_B33 = self.c3(out_B30)
        
        out_B4 = self.B4(out_B33)
        out_B40 = torch.cat([out_B30 , out_B4], dim=1)
        out_B44 = self.c4(out_B40)
        
        out_B5 = self.B5(out_B44)
        out_B50 = torch.cat([out_B40 , out_B5], dim=1)
        out_B55 = self.c5(out_B50)
        
        out_B6 = self.B6(out_B55)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx