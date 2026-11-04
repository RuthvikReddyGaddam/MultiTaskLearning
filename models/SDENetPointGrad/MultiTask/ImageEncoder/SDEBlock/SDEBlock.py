from typing import Optional
import torch
import torch.nn as nn
from models.SDENetPointGrad.MultiTask.ImageEncoder.SDEBlock.SDOperator import SDOperator
from models.SDENetPointGrad.MultiTask.ImageEncoder.SDEBlock.SEOperator import SEOperator


class SDEBlock(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        dec_in_channels: int = 3,
        sde_block: Optional[str] = "Full",
        r: int = 3,
        k_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        input: bool = True,
    ):
        super().__init__()
        self.C_enc = in_channels
        self.C_dec = dec_in_channels
        self.r = r
        self.sde_block = sde_block
        self.input = input
        self.C_sd = self.calculate_C_sd()

        if sde_block =="Full":
            self.sd_operator = SDOperator(r=self.r, in_channels=in_channels, dec_in_channels=dec_in_channels, k_size=k_size, padding=padding, stride=stride, input=input)
            self.se_operator = SEOperator(C_sd = self.C_sd)
        elif sde_block =="SE_only":
            self.sd_operator = None
            self.se_operator = SEOperator(C_sd = in_channels + dec_in_channels)
        else:
            self.sd_operator = None
            self.se_operator = None
        if not input:
            self.upsample = nn.ConvTranspose2d(self.C_dec, self.C_dec, kernel_size=2, stride=2)


    def calculate_C_sd(self):
        if self.sde_block == "Full":
            C_sq = self.C_enc // self.r
            C_lbp = 8 * C_sq
            C_motif = 15 * C_sq
            channels = C_lbp + C_motif + self.C_dec
        else:
            channels = self.C_enc + self.C_dec
        return channels
    
    def forward(self, U_enc, U_dec):
        if self.sde_block == "Full":    
            U_sd = self.sd_operator(U_enc, U_dec)
            U_ex = self.se_operator(U_sd)
        elif self.sde_block == "SE_only":
            if not self.input:
                U_dec = self.upsample(U_dec)
            U_ex = self.se_operator(torch.concat([U_enc, U_dec], dim=1))
        else:
            U_ex = torch.concat([U_enc, U_dec], dim=1)
        return U_ex