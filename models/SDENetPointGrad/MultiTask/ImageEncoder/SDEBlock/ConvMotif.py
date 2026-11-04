import torch
import torch.nn as nn
import numpy as np


class ConvMotif(nn.Conv2d):
    """
    Depthwise Peano Motif Convolution.
    Convolves each input channel independently with the 36 fixed 2x2 difference kernels (12 motifs x 3 differences).
    Output channels: 36 * in_channels.
    """
    def __init__(self, in_channels: int = 1, stride: int = 1):
        super(ConvMotif, self).__init__(
            in_channels=in_channels,
            out_channels=36 * in_channels,
            kernel_size=2,
            stride=stride,
            padding=0,
            groups=in_channels,
            bias=False,
        )
        base_kernels = torch.zeros((36, 1, 2, 2), dtype=torch.float32)

        # Motifs 1 to 6 (r = 1..6)
        base_kernels[0, 0, :, :] = torch.from_numpy(np.asarray([[1, -1], [0, 0]]))   # z
        base_kernels[1, 0, :, :] = torch.from_numpy(np.asarray([[0, 1], [-1, 0]]))
        base_kernels[2, 0, :, :] = torch.from_numpy(np.asarray([[0, 0], [1, -1]]))

        base_kernels[3, 0, :, :] = torch.from_numpy(np.asarray([[1, -1], [0, 0]]))   # c
        base_kernels[4, 0, :, :] = torch.from_numpy(np.asarray([[0, 1], [0, -1]]))
        base_kernels[5, 0, :, :] = torch.from_numpy(np.asarray([[0, 0], [-1, 1]]))

        base_kernels[6, 0, :, :] = torch.from_numpy(np.asarray([[1, 0], [-1, 0]]))   # N
        base_kernels[7, 0, :, :] = torch.from_numpy(np.asarray([[0, -1], [1, 0]]))
        base_kernels[8, 0, :, :] = torch.from_numpy(np.asarray([[0, 1], [0, -1]]))

        base_kernels[9, 0, :, :] = torch.from_numpy(np.asarray([[1, 0], [-1, 0]]))   # U
        base_kernels[10, 0, :, :] = torch.from_numpy(np.asarray([[0, 0], [1, -1]]))
        base_kernels[11, 0, :, :] = torch.from_numpy(np.asarray([[0, -1], [0, 1]]))

        base_kernels[12, 0, :, :] = torch.from_numpy(np.asarray([[1, 0], [0, -1]]))  # alpha
        base_kernels[13, 0, :, :] = torch.from_numpy(np.asarray([[0, -1], [0, 1]]))
        base_kernels[14, 0, :, :] = torch.from_numpy(np.asarray([[0, 1], [-1, 0]]))

        base_kernels[15, 0, :, :] = torch.from_numpy(np.asarray([[1, 0], [0, -1]]))  # gamma
        base_kernels[16, 0, :, :] = torch.from_numpy(np.asarray([[0, 0], [-1, 1]]))
        base_kernels[17, 0, :, :] = torch.from_numpy(np.asarray([[0, -1], [1, 0]]))

        # Motifs 7 to 12 (r = 7..12)
        n = 18
        base_kernels[0 + n, 0, :, :] = torch.from_numpy(np.asarray([[-1, 1], [0, 0]]))  # r=7
        base_kernels[1 + n, 0, :, :] = torch.from_numpy(np.asarray([[1, 0], [0, -1]]))
        base_kernels[2 + n, 0, :, :] = torch.from_numpy(np.asarray([[0, 0], [-1, 1]]))

        base_kernels[3 + n, 0, :, :] = torch.from_numpy(np.asarray([[-1, 1], [0, 0]]))  # r=8
        base_kernels[4 + n, 0, :, :] = torch.from_numpy(np.asarray([[1, 0], [-1, 0]]))
        base_kernels[5 + n, 0, :, :] = torch.from_numpy(np.asarray([[0, 0], [1, -1]]))

        base_kernels[6 + n, 0, :, :] = torch.from_numpy(np.asarray([[0, 1], [0, -1]]))  # r=9
        base_kernels[7 + n, 0, :, :] = torch.from_numpy(np.asarray([[-1, 0], [0, 1]]))
        base_kernels[8 + n, 0, :, :] = torch.from_numpy(np.asarray([[1, 0], [-1, 0]]))

        base_kernels[9 + n, 0, :, :] = torch.from_numpy(np.asarray([[0, -1], [0, 1]]))  # r=10
        base_kernels[10 + n, 0, :, :] = torch.from_numpy(np.asarray([[-1, 1], [0, 0]]))
        base_kernels[11 + n, 0, :, :] = torch.from_numpy(np.asarray([[1, 0], [-1, 0]]))

        base_kernels[12 + n, 0, :, :] = torch.from_numpy(np.asarray([[0, 1], [-1, 0]]))  # r=11
        base_kernels[13 + n, 0, :, :] = torch.from_numpy(np.asarray([[-1, 0], [1, 0]]))
        base_kernels[14 + n, 0, :, :] = torch.from_numpy(np.asarray([[1, 0], [0, -1]]))

        base_kernels[15 + n, 0, :, :] = torch.from_numpy(np.asarray([[0, -1], [1, 0]]))  # r=12
        base_kernels[16 + n, 0, :, :] = torch.from_numpy(np.asarray([[-1, 1], [0, 0]]))
        base_kernels[17 + n, 0, :, :] = torch.from_numpy(np.asarray([[1, 0], [0, -1]]))

        self.register_buffer("base_kernels", base_kernels)
        self.weight.data = base_kernels.repeat(in_channels, 1, 1, 1)
        self.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.size(1)
        if c == self.in_channels:
            return super().forward(x)
        # Dynamically adapt depthwise groups for arbitrary input channel counts
        kernels = self.base_kernels.repeat(c, 1, 1, 1).to(device=x.device, dtype=x.dtype)
        return torch.nn.functional.conv2d(x, kernels, stride=self.stride, padding=0, groups=c)



