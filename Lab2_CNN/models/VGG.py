import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

class VGG(nn.Module):
    def __init__(self, config, output_dim):
        super().__init__()

        self.features = self.get_vgg_layers(config)

        self.avgpool = nn.AdaptiveAvgPool2d(7) # allow for different image input sizes

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x, h


    def get_vgg_layers(self, config):

        layers = []
        in_channels = 3

        for c in config:
            assert c == 'M' or isinstance(c, int)
            if c == 'M':
                layers += [nn.MaxPool2d(kernel_size=2)]
            else:
                conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
                in_channels = c

        return nn.Sequential(*layers)

