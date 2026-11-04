import torch
import torch.nn as nn
import numpy as np


class ConvLBP(nn.Conv2d):
    """
    Depthwise Local Binary Pattern (LBP) Convolution.
    Convolves each input channel independently with the 8 fixed 3x3 difference kernels.
    Output channels: 8 * in_channels.
    """
    def __init__(self, in_channels: int = 1, stride: int = 1):
        super(ConvLBP, self).__init__(
            in_channels=in_channels,
            out_channels=8 * in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        base_kernels = torch.zeros((8, 1, 3, 3), dtype=torch.float32)
        base_kernels[0, 0, :, :] = torch.from_numpy(np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, 0]]))
        base_kernels[1, 0, :, :] = torch.from_numpy(np.asarray([[0, 1, 0], [0, -1, 0], [0, 0, 0]]))
        base_kernels[2, 0, :, :] = torch.from_numpy(np.asarray([[0, 0, 1], [0, -1, 0], [0, 0, 0]]))
        base_kernels[3, 0, :, :] = torch.from_numpy(np.asarray([[0, 0, 0], [1, -1, 0], [0, 0, 0]]))
        base_kernels[4, 0, :, :] = torch.from_numpy(np.asarray([[0, 0, 0], [0, -1, 1], [0, 0, 0]]))
        base_kernels[5, 0, :, :] = torch.from_numpy(np.asarray([[0, 0, 0], [0, -1, 0], [1, 0, 0]]))
        base_kernels[6, 0, :, :] = torch.from_numpy(np.asarray([[0, 0, 0], [0, -1, 0], [0, 1, 0]]))
        base_kernels[7, 0, :, :] = torch.from_numpy(np.asarray([[0, 0, 0], [0, -1, 0], [0, 0, 1]]))

        self.register_buffer("base_kernels", base_kernels)
        self.weight.data = base_kernels.repeat(in_channels, 1, 1, 1)
        self.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.size(1)
        if c == self.in_channels:
            return super().forward(x)
        # Dynamically adapt depthwise groups for arbitrary input channel counts
        kernels = self.base_kernels.repeat(c, 1, 1, 1).to(device=x.device, dtype=x.dtype)
        return torch.nn.functional.conv2d(x, kernels, stride=self.stride, padding=1, groups=c)

