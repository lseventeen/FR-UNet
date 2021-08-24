import torch
import torch.nn as nn
from .utils import initialize_weights
class conv(nn.Module):
    def __init__(self, in_c, out_c):
        super(conv, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.conv = nn.Sequential(

            nn.Conv2d(out_c, out_c, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(out_c, out_c, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True),

        )    
            
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        if self.in_c != self.out_c:
            self.diminsh_c = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1,
                          padding=0, stride=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.1, inplace=True)
                
            )
       


    def forward(self, x):
        if self.in_c != self.out_c:
            x = self.diminsh_c(x)
        res = x
        x = self.conv(x)
        out = x + res
        out = self.relu(out)
        return x

    


    
       
  
class up(nn.Module):
    def __init__(self, in_c, out_c):
        super(up, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2,
                               padding=0, stride=2, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=False),
            
        )

    def forward(self, x):
        x = self.up(x)
        return x


class down(nn.Module):
    def __init__(self, in_c, out_c):
        super(down, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=2,
                      padding=0, stride=2, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True)
           
        )

    def forward(self, x):
        x = self.down(x)
        return x



class UNet_RES(nn.Module):

    def __init__(self, num_channels=1, num_classes=1, feature_scale=2):
        super(UNet_RES, self).__init__()
        self.in_channels = num_channels
        self.feature_scale = feature_scale
    

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.enconv1 = conv(self.in_channels, filters[0])
        self.down1 = down(filters[0],filters[1])
        self.enconv2 = conv(filters[1], filters[1])
        self.down2 = down(filters[1],filters[2])
        self.enconv3 = conv(filters[2], filters[2])
        self.down3 = down(filters[2],filters[3])

        self.center = conv(filters[3], filters[3])
     
        # upsampling
        self.up3 = up(filters[3],filters[2])
        self.deconv3 = conv(filters[2]*2, filters[2])
        self.up2 = up(filters[2],filters[1])
        self.deconv2 = conv(filters[1]*2, filters[1])
        self.up1 = up(filters[1],filters[0])
        self.deconv1 = conv(filters[0]*2, filters[0])

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], num_classes, 1)
        # initialise weights
        initialize_weights(self)

    def forward(self, inputs):
        enconv1 = self.enconv1(inputs)          
        down1 =   self.down1(enconv1)      
        
        enconv2 = self.enconv2(down1)          
        down2 =   self.down2(enconv2)         

        enconv3 = self.enconv3(down2)          
        down3 =   self.down3(enconv3)       

        conv4 = self.center(down3)         
          

       
        up3 = self.up3(conv4)  
        deconv3 = self.deconv3(torch.cat([enconv3, up3], dim=1)) 

        up2 = self.up2(deconv3)  
        deconv2 = self.deconv2(torch.cat([enconv2, up2], dim=1))     

        up1 = self.up1(deconv2)  
        deconv3 = self.deconv1(torch.cat([enconv1, up1], dim=1))   

        final = self.final(up1)

        return final


