import torch.nn as nn
from models.SDENetPointGrad.MultiTask.ImageEncoder.SDEBlock.SDEBlock import SDEBlock

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        num_groups = max(1, out_channels // 8)
        while out_channels % num_groups != 0:
            num_groups -= 1
        self.convBlock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.convBlock(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        skip = self.conv(x)
        out = self.maxpool(skip)
        return skip, out
        
class ImageEncoder(nn.Module):
    FEATURE_NAMES = ("skip_l3", "skip_l2", "skip_l1", "bottleneck")
    
    def __init__(self, in_channels=3, out_channels=(64, 128, 128, 256), r=3):
        super().__init__()

        self.sde_block = SDEBlock(in_channels=in_channels, dec_in_channels=in_channels, r=r, sde_block="Full", input=True)
        sde_out_channels = self.sde_block.calculate_C_sd()
        
        blocks = {}
        blocks[self.FEATURE_NAMES[0]] = EncoderBlock(in_channels=sde_out_channels, out_channels = out_channels[0])
        blocks[self.FEATURE_NAMES[1]] = EncoderBlock(in_channels=out_channels[0], out_channels = out_channels[1])
        blocks[self.FEATURE_NAMES[2]] = EncoderBlock(in_channels=out_channels[1], out_channels = out_channels[2])
        blocks[self.FEATURE_NAMES[3]] = ConvBlock(in_channels=out_channels[2], out_channels = out_channels[3], kernel_size=1, padding=0)
        self.image_encoder_blocks = nn.ModuleDict(blocks) 
        
        
    def forward(self, x):
        image_features = {}

        x_gray = 0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
        x_gray = x_gray.unsqueeze(1)
        x = self.sde_block(U_enc=x_gray, U_dec=x)
        
        for name, block in self.image_encoder_blocks.items():
            if name == "bottleneck":
                x = block(x)
                image_features[name] = x
            else:
                skip, x = block(x)
                image_features[name] = skip
        return image_features
