import torch
from torch._C import TupleType
import torch.nn as nn
from torch.nn.utils import weight_norm

from .frnet import block 

class FRNet_P1(nn.Module):
    def __init__(self,  num_classes=1,num_channels=1, feature_scale=2,  dropout=0.2, out_ave=True, fuse = True):
        super(FRNet_P1, self).__init__()
        self.out_ave = out_ave
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / feature_scale) for x in filters]
        self.block1_3 = block(
            num_channels, filters[0],  dp=dropout, is_up=False, is_down=True, fuse = fuse)
        self.block1_2 = block(
            filters[0], filters[0],  dp=dropout, is_up=False, is_down=True, fuse = fuse)
        self.block1_1 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=True, fuse = fuse)
        self.block10 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=True, fuse = fuse)
        self.block11 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=False, fuse = fuse)
        self.block12 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=False, fuse = fuse)
      

        self.block2_2 = block(
            filters[1], filters[1],  dp=dropout, is_up=True, is_down=True, fuse = fuse)
        self.block2_1 = block(
            filters[1]*2, filters[1],  dp=dropout, is_up=True, is_down=True, fuse = fuse)
        self.block20 = block(
            filters[1]*3, filters[1],  dp=dropout, is_up=True, is_down=False, fuse = fuse)
        self.block21 = block(
            filters[1]*3, filters[1],  dp=dropout, is_up=True, is_down=False, fuse = fuse)
        
        self.block3_1 = block(
            filters[2], filters[2],  dp=dropout, is_up=True, is_down=False, fuse = fuse)
        self.block30 = block(
            filters[2]*2, filters[2],  dp=dropout, is_up=True, is_down=False, fuse = fuse)
       
        
      

        self.final1 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final2 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final3 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final4 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)

       
   

    def forward(self, x):
        x1_3, x_down1_3 = self.block1_3(x)

        x1_2, x_down1_2 = self.block1_2(x1_3)
        x2_2, x_up2_2, x_down2_2 = self.block2_2(x_down1_3)

        x1_1, x_down1_1 = self.block1_1(torch.cat([x1_2, x_up2_2], dim=1))
        x2_1, x_up2_1, x_down2_1 = self.block2_1(
            torch.cat([x_down1_2, x2_2], dim=1))
        x3_1, x_up3_1 = self.block3_1(x_down2_2)

        x10, x_down10 = self.block10(torch.cat([x1_1, x_up2_1], dim=1))
        x20, x_up20 = self.block20(
            torch.cat([x_down1_1, x2_1, x_up3_1], dim=1))
        _, x_up30 = self.block30(torch.cat([x_down2_1, x3_1], dim=1))
        

        x11 = self.block11(torch.cat([x10, x_up20], dim=1))
        _, x_up21 = self.block21(torch.cat([x_down10, x20, x_up30], dim=1))
       
        x12 = self.block12(torch.cat([x11, x_up21], dim=1))
        
       

        
        if self.out_ave == True:
            output = (self.final1(x1_1) + self.final2(x10) + self.final3(x11) + self.final4(x12))/4
        else:
            output = self.final4(x12)
        return output


class FRNet_P2(nn.Module):
    def __init__(self,  num_classes=1,num_channels=1, feature_scale=2,  dropout=0.2, out_ave=True, fuse = True):
        super(FRNet_P2, self).__init__()
        self.out_ave = out_ave
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / feature_scale) for x in filters]
        self.block1_3 = block(
            num_channels, filters[0],  dp=dropout, is_up=False, is_down=True, fuse = fuse)
        self.block1_2 = block(
            filters[0], filters[0],  dp=dropout, is_up=False, is_down=True, fuse = fuse)
        self.block1_1 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=True, fuse = fuse)
        self.block10 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=False, fuse = fuse)
        self.block11 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=False, fuse = fuse)
   
      

        self.block2_2 = block(
            filters[1], filters[1],  dp=dropout, is_up=True, is_down=True, fuse = fuse)
        self.block2_1 = block(
            filters[1]*2, filters[1],  dp=dropout, is_up=True, is_down=False, fuse = fuse)
        self.block20 = block(
            filters[1]*3, filters[1],  dp=dropout, is_up=True, is_down=False, fuse = fuse)
        
        self.block3_1 = block(
            filters[2], filters[2],  dp=dropout, is_up=True, is_down=False, fuse = fuse)
        
        
      

        self.final1 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final2 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final3 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        
   

    def forward(self, x):
        x1_3, x_down1_3 = self.block1_3(x)

        x1_2, x_down1_2 = self.block1_2(x1_3)
        x2_2, x_up2_2, x_down2_2 = self.block2_2(x_down1_3)

        x1_1, x_down1_1 = self.block1_1(torch.cat([x1_2, x_up2_2], dim=1))
        x2_1, x_up2_1 = self.block2_1(
            torch.cat([x_down1_2, x2_2], dim=1))
        _, x_up3_1 = self.block3_1(x_down2_2)

        x10 = self.block10(torch.cat([x1_1, x_up2_1], dim=1))
        _, x_up20 = self.block20(
            torch.cat([x_down1_1, x2_1, x_up3_1], dim=1))
       
        x11 = self.block11(torch.cat([x10, x_up20], dim=1))
        
        if self.out_ave == True:
            output = (self.final1(x1_1) + self.final2(x10) + self.final3(x11))/3
        else:
            output = self.final3(x11)

        return output

            
       


class FRNet_P3(nn.Module):
    def __init__(self,  num_classes=1,num_channels=1, feature_scale=2,  dropout=0.2, out_ave=True, fuse = True):
        super(FRNet_P3, self).__init__()
        self.out_ave = out_ave
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / feature_scale) for x in filters]
        self.block1_3 = block(
            num_channels, filters[0],  dp=dropout, is_up=False, is_down=True, fuse = fuse)
        self.block1_2 = block(
            filters[0], filters[0],  dp=dropout, is_up=False, is_down=True, fuse = fuse)
        self.block1_1 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=False, fuse = fuse)
        self.block10 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=False, fuse = fuse)
        
   
      

        self.block2_2 = block(
            filters[1], filters[1],  dp=dropout, is_up=True, is_down=False, fuse = fuse)
        self.block2_1 = block(
            filters[1]*2, filters[1],  dp=dropout, is_up=True, is_down=False, fuse = fuse)
      
        
        
        
      

        self.final1 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final2 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
      
        
   

    def forward(self, x):
        x1_3, x_down1_3 = self.block1_3(x)

        x1_2, x_down1_2 = self.block1_2(x1_3)
        x2_2, x_up2_2 = self.block2_2(x_down1_3)

        x1_1 = self.block1_1(torch.cat([x1_2, x_up2_2], dim=1))
        _, x_up2_1 = self.block2_1(
            torch.cat([x_down1_2, x2_2], dim=1))
        

        x10 = self.block10(torch.cat([x1_1, x_up2_1], dim=1))
    
       
        
        if self.out_ave == True:
            output = (self.final1(x1_1) + self.final2(x10))/3
        else:
            output = self.final3(x10)

        return output


class FRNet_P4(nn.Module):
    def __init__(self,  num_classes=1,num_channels=1, feature_scale=2,  dropout=0.2, out_ave=True, fuse = True):
        super(FRNet_P4, self).__init__()
        self.out_ave = out_ave
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / feature_scale) for x in filters]
        self.block1_3 = block(
            num_channels, filters[0],  dp=dropout, is_up=False, is_down=True, fuse = fuse)
        self.block1_2 = block(
            filters[0], filters[0],  dp=dropout, is_up=False, is_down=False, fuse = fuse)
        self.block1_1 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=False, fuse = fuse)
       
   
      

        self.block2_2 = block(
            filters[1], filters[1],  dp=dropout, is_up=True, is_down=False, fuse = fuse)
        
        self.final1 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
      
        
   

    def forward(self, x):
        x1_3, x_down1_3 = self.block1_3(x)

        x1_2  = self.block1_2(x1_3)
        _, x_up2_2 = self.block2_2(x_down1_3)

        x1_1 = self.block1_1(torch.cat([x1_2, x_up2_2], dim=1))
        
       
        
        output = self.final1(x1_1)

        return output