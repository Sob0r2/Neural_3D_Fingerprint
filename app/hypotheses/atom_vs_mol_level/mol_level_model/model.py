import torch
import torch.nn as nn
import torch.nn.functional as F

class DescriptorContrastiveModel(nn.Module):
    def __init__(self, input_dim=18, embedding_dim=1024):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),

            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = F.normalize(x, p=2, dim=1)
        return x
