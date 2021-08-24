import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from .frnet import conv,block




   

class FRNet_OOS(nn.Module):
    def __init__(self,  num_classes=1,num_channels=1, feature_scale=2,  dropout=0, out_ave=True):
        super(FRNet_OOS, self).__init__()
        filters = int(64 / feature_scale)
        self.out_ave = out_ave
        self.res1 = conv(num_channels,filters, dropout)
        self.res2 = conv(filters,filters, dropout)
        self.res3 = conv(filters,filters, dropout)
        self.res4 = conv(filters,filters, dropout)
        self.res5 = conv(filters,filters, dropout)
        self.res6 = conv(filters,filters, dropout)
        self.res7 = conv(filters,filters, dropout)
        
        self.final = nn.Conv2d(
            filters, num_classes, kernel_size=1, padding=0, bias=True)
     
    def forward(self, x):
        x1 = self.res1(x)
        x2 = self.res2(x1)
        x3 = self.res3(x2)
        x4 = self.res4(x3)
        x5 = self.res5(x4)
        x6 = self.res6(x5)
        x7 = self.res7(x6)

        # if self.out_ave == True:
        #     output = (self.final1(x1)+self.final2(x2)+self.final3(x3)+self.final4(x4)+self.final5(x5)+self.final6(x6)+self.final7(x7))/7
        # else:
        output = self.final(x7)

        return output

class FRNet_NOS(nn.Module):
    def __init__(self,  num_classes=1,num_channels=1, feature_scale=2,  dropout=0, fuse = False):
        super(FRNet_NOS, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / feature_scale) for x in filters]
        self.block1_3 = block(
            num_channels, filters[0],  dp=dropout, is_up=False, is_down=True, fuse = fuse)
        self.block13 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=False, fuse = fuse)

        self.block2_2 = block(
            filters[1], filters[1],  dp=dropout, is_up=False, is_down=True, fuse = fuse)
        self.block2_1 = block(
            filters[1], filters[1],  dp=dropout, is_up=False, is_down=True, fuse = fuse)
        self.block20 = block(
            filters[1]*2, filters[1],  dp=dropout, is_up=False, is_down=True, fuse = fuse)
        self.block21 = block(
            filters[1]*2, filters[1],  dp=dropout, is_up=False, is_down=False, fuse = fuse)
        self.block22 = block(
            filters[1]*2, filters[1],  dp=dropout, is_up=True, is_down=False, fuse = fuse)

        self.block3_1 = block(
            filters[2], filters[2],  dp=dropout, is_up=True, is_down=True, fuse = fuse)
        self.block30 = block(
            filters[2]*2, filters[2],  dp=dropout, is_up=True, is_down=False, fuse = fuse)
        self.block31 = block(
            filters[2]*3, filters[2],  dp=dropout, is_up=True, is_down=False, fuse = fuse)

        self.block40 = block(filters[3], filters[3],
                             dp=dropout, is_up=True, is_down=False, fuse = fuse)

 
        self.final = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x1_3, x_down1_3 = self.block1_3(x)

        x2_2, x_down2_2 = self.block2_2(x_down1_3)

        
        x2_1, x_down2_1 = self.block2_1(x2_2)
        x3_1, x_up3_1, x_down3_1 = self.block3_1(x_down2_2)

        
        x20,  x_down20 = self.block20(
            torch.cat([x2_1, x_up3_1], dim=1))
        x30, x_up30 = self.block30(torch.cat([x_down2_1, x3_1], dim=1))
        _, x_up40 = self.block40(x_down3_1)

        
        x21 = self.block21(torch.cat([x20, x_up30], dim=1))
        _, x_up31 = self.block31(torch.cat([x_down20, x30, x_up40], dim=1))

        _, x_up22 = self.block22(torch.cat([ x21, x_up31], dim=1))

        x13 = self.block13(torch.cat([x1_3, x_up22], dim=1))
      
        output = self.final(x13)

        return output



