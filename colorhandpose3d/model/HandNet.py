# HandSegNet.py
# Wonjun Jeong (wonjun.jg@gmail.com)
# Model definition for the hand binary classification net.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg11


class HandNet(nn.Module):
    def __init__(self, pretraiend=vgg11(pretrained=True)):
        super(HandNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(35840, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.activation = {}
        self.pretraiend = pretraiend.features[:]
        if self.pretraiend:
            for param in self.pretraiend.parameters():
                param.requires_grad = False
            self.pretraiend.eval()

    def forward(self, x):
        self.pretraiend[20].register_forward_hook(self.get_activation("max_pool_2d"))
        _ = self.pretraiend(x)
        X = self.activation["max_pool_2d"]  # (32, 512, 7, 7)
        x = torch.flatten(X, start_dim=1)  # (B, 25088)
        out = self.fc(x)

        return out

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()

        return hook

