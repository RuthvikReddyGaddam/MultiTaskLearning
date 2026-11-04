import torch
import torch.nn as nn
from models.SDENetPointGrad.MultiTask.ImageEncoder.SDEBlock.ConvLBP import ConvLBP
from models.SDENetPointGrad.MultiTask.ImageEncoder.SDEBlock.ConvMotif import ConvMotif


class Hybrid(nn.Module):
    """
    Depthwise Hybrid feature extractor combining Local Binary Patterns (LBP)
    and Peano Motifs across all input channels independently.
    For an input with C channels:
      - ConvLBP outputs 8 * C channels
      - ConvMotif outputs 36 * C channels (12 motifs x 3 differences per channel)
      - motifs sum over differences: 12 * C channels
      - motif statistics (max, min, median per channel): 3 * C channels
      - Total output: (8 + 12 + 3) * C = 23 * C channels
    """
    def __init__(self, in_channels: int = 1, stride: int = 1):
        super(Hybrid, self).__init__()
        self.in_channels = in_channels
        self.conv_lbp = ConvLBP(in_channels=in_channels, stride=stride)
        self.motif = ConvMotif(in_channels=in_channels, stride=stride)

        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        xlbp = self.activation(self.conv_lbp(x))
        x_motif = self.activation(self.motif(x)) / 3.0

        H_out, W_out = xlbp.shape[2:]
        if x_motif.shape[2:] != (H_out, W_out):
            x_motif = nn.Upsample(size=(H_out, W_out), mode='bilinear', align_corners=True)(x_motif)

        # Depthwise motif processing:
        # x_motif has shape (B, 36 * C, H_out, W_out)
        # Reshape to (B, C, 12, 3, H_out, W_out) and sum over the 3 difference filters
        motifs = x_motif.view(B, C, 12, 3, H_out, W_out).sum(dim=3)  # (B, C, 12, H_out, W_out)

        # Statistics per channel across the 12 motifs
        motif_max, _ = motifs.max(dim=2)        # (B, C, H_out, W_out)
        motif_min, _ = motifs.min(dim=2)        # (B, C, H_out, W_out)
        motif_median, _ = motifs.median(dim=2)  # (B, C, H_out, W_out)

        motifs_flat = motifs.view(B, 12 * C, H_out, W_out)
        motif_value = torch.cat([motif_max, motif_min, motif_median], dim=1)  # (B, 3 * C, H_out, W_out)

        return torch.cat([xlbp, motifs_flat, motif_value], dim=1)


class SDOperator(nn.Module):
    def __init__(self, r: int = 3, in_channels: int = 3, dec_in_channels: int = 3,
                 k_size: int = 3, padding: int = 1, stride: int = 1, input: bool = True):
        super().__init__()
        self.input = input
        self.C_enc = in_channels
        self.C_dec = dec_in_channels
        self.r = r
        self.C_sq = max(1, self.C_enc // self.r)

        self.conv1 = nn.Conv2d(in_channels=self.C_enc, out_channels=self.C_sq, kernel_size=k_size, padding=padding, stride=stride) if not self.input else nn.Identity()
        self.hybrid = Hybrid(in_channels=self.C_sq, stride=stride)

        if not self.input:
            self.upsample = nn.ConvTranspose2d(in_channels=self.C_dec, out_channels=self.C_dec,
                                               kernel_size=2, padding=0, stride=2)

    def forward(self, U_enc: torch.Tensor, U_dec: torch.Tensor) -> torch.Tensor:
        U_enc = self.conv1(U_enc)
        U_de = self.hybrid(U_enc)

        if not self.input:
            U_dec = self.upsample(U_dec)
        U_sd = torch.cat([U_de, U_dec], dim=1)

        return U_sd