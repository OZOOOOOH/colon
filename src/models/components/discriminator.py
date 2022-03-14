import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class Discriminator(nn.Module):
    def __init__(self,x):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(x, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 3),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity
