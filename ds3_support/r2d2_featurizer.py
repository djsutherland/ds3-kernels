import torch
from torch import nn


class R2D2Featurizer(nn.Module):
    "Featurizer network from https://arxiv.org/abs/1805.08136"
    def __init__(self, in_channels=1, dropout_p=0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, 96, 3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(96, affine=True),
                    nn.MaxPool2d(2, stride=2),
                    nn.LeakyReLU(0.1),
                ),
                nn.Sequential(
                    nn.Conv2d(96, 192, 3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(192, affine=True),
                    nn.MaxPool2d(2, stride=2),
                    nn.LeakyReLU(0.1),
                ),
                nn.Sequential(
                    nn.Conv2d(192, 384, 3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(384, affine=True),
                    nn.MaxPool2d(2, stride=2),
                    nn.LeakyReLU(0.1),
                    nn.Dropout(p=dropout_p),
                ),
                nn.Sequential(
                    nn.Conv2d(384, 512, 3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(512, affine=True),
                    nn.MaxPool2d(2, stride=1),
                    nn.LeakyReLU(0.1),
                    nn.Dropout(p=dropout_p),
                ),
            ]
        )

        # pooling used for featurizer: in their code, not mentioned in paper :/
        self.pool3 = nn.MaxPool2d(2, stride=1)

    def forward(self, x):
        x = self.blocks[0](x)
        x = self.blocks[1](x)
        x = self.blocks[2](x)
        p1 = self.pool3(x).view(x.size(0), -1)
        x = self.blocks[3](x)
        p2 = x.view(x.size(0), -1)
        return torch.cat([p1, p2], 1)
