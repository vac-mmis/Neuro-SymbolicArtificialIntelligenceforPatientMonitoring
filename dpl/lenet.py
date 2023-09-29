import torch
from torch import nn as nn

class LeNet(nn.Module):
    def __init__(self, n=10, with_softmax=True):
        super(LeNet, self).__init__()
        self.with_softmax = with_softmax
        self.batch_size = 1
        if with_softmax:
            if n == 1:
                self.softmax = nn.Sigmoid()
            else:
                self.softmax = nn.Softmax(1)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.Tanh(),
            nn.AvgPool2d(2, 2),  # 16 8 8 -> 16 4 4
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 84),
            nn.Tanh(),
            nn.Linear(84, n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if self.with_softmax:
            x = self.softmax(x)
        return x