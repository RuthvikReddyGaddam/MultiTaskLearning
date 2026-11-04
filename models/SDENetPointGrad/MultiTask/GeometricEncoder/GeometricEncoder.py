import torch
import torch.nn as nn
import torch.nn.functional as F
from models.SDENetPointGrad.MultiTask.ImageEncoder.ImageEncoder import ConvBlock


class GradientConv(nn.Module):
    def __init__(self):
        super().__init__()

        kernels = [
            [
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, 0]
            ],
            [
                [0, 1, 0],
                [0, -1, 0],
                [0, 0, 0]
            ],
            [
                [0, 0, 1],
                [0, -1, 0],
                [0, 0, 0]
            ],
            [
                [0, 0, 0],
                [1, -1, 0],
                [0, 0, 0]
            ],
            [
                [0, 0, 0],
                [0, -1, 1],
                [0, 0, 0]
            ],
            [
                [0, 0, 0],
                [0, -1, 0],
                [1, 0, 0]
            ],
            [
                [0, 0, 0],
                [0, -1, 0],
                [0, 1, 0]
            ],
            [
                [0, 0, 0],
                [0, -1, 0],
                [0, 0, 1]
            ],
        ]

        # Shape: [8, 1, 3, 3]
        kernels = torch.tensor(kernels, dtype=torch.float32).unsqueeze(1)

        # Frozen, but follows model device and is included in state_dict
        self.register_buffer("kernels", kernels)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]

        Returns:
            out: [B, C * 8, H, W]
        """

        B, C, H, W = x.shape

        # Repeat the 8 kernels for each input channel
        # [8, 1, 3, 3] -> [C*8, 1, 3, 3]
        weight = self.kernels.repeat(C, 1, 1, 1)

        # groups=C means each input channel is processed independently.
        out = F.conv2d(
            x,
            weight,
            bias=None,
            stride=1,
            padding=1,
            groups=C
        )

        return out


class PointGrad1(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()

        self.gradConv = GradientConv()

        # Original C channels + 8 gradient channels per input channel
        self.conv_in_channels = 9 * in_channels

        num_groups = max(1, out_channels // 8)
        while out_channels % num_groups != 0:
            num_groups -= 1
        self.convBlock = nn.Sequential(
            nn.Conv2d(
                in_channels=self.conv_in_channels,
                out_channels=out_channels,
                kernel_size=1
            ),
            nn.GroupNorm(num_groups, out_channels),
            nn.LeakyReLU()
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x_grad = self.gradConv(x)

        # [B, C, H, W] + [B, 8C, H, W]
        # -> [B, 9C, H, W]
        x = torch.cat([x, x_grad], dim=1)

        x_skip = self.convBlock(x)
        x = self.maxpool(x_skip)
        return x_skip, x

class ConvPG(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()

        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=out_channels)

        self.conv2 = ConvBlock(in_channels=out_channels, out_channels=out_channels)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x_skip = self.conv2(x)
        x = self.maxpool(x_skip)
        return x_skip, x


class GeometricEncoder(nn.Module):
    FEATURE_NAMES = ("skip_l3", "skip_l2", "skip_l1", "bottleneck")

    def __init__(self, in_channels=3, out_channels=(64, 128, 128, 256)):
        super().__init__()

        blocks = {}

        blocks[self.FEATURE_NAMES[0]] = PointGrad1(
            in_channels=in_channels,
            out_channels=out_channels[0]
        )

        for index, name in enumerate(self.FEATURE_NAMES[1:], start=1):
            blocks[name] = PointGrad1(
                in_channels=out_channels[index - 1],
                out_channels=out_channels[index]
            )

        # self.geometric_encoder_blocks = nn.ModuleDict(blocks)
        # blocks[self.FEATURE_NAMES[0]] = PointGrad1(in_channels=in_channels, out_channels=out_channels[0])
        # blocks[self.FEATURE_NAMES[1]] = PointGrad1(in_channels=out_channels[0], out_channels=out_channels[1])
        # blocks[self.FEATURE_NAMES[2]] = ConvPG(in_channels=out_channels[1], out_channels=out_channels[2])
        # blocks[self.FEATURE_NAMES[3]] = ConvPG(in_channels=out_channels[2], out_channels=out_channels[3])
        
        self.geometric_encoder_blocks = nn.ModuleDict(blocks)

    def forward(self, x):
        geometric_features = {}

        for name, block in self.geometric_encoder_blocks.items():

            x_skip, x = block(x)

            geometric_features[name] = x_skip

        return geometric_features
