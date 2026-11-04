import torch
import torch.nn as nn

class SEOperator(nn.Module):
    def __init__(self, C_sd):
        super().__init__()
        self.C_sd = C_sd
        
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = nn.Sequential(
            nn.Linear(in_features=self.C_sd, out_features=self.C_sd//2),
            nn.ReLU(),
            nn.Linear(in_features=self.C_sd//2, out_features=self.C_sd),
            nn.Sigmoid()
        )
        
    def forward(self, U_sd):
        B, C, _, _ = U_sd.shape
        U_pool = self.pool1(U_sd)
        U_flat  = torch.flatten(U_pool, 1)
        S_sd = self.dense(U_flat)
        U_ex = U_sd * S_sd.view(B, C, 1, 1)
        return U_ex
        