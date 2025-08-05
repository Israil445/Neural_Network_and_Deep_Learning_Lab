import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOv1(nn.Module):
    def __init__(self, grid_size=7, num_boxes=2, num_classes=1):
        super(YOLOv1, self).__init__()
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes

        # Backbone CNN inspired by VGG/YOLOv1
        self.features = nn.Sequential(
            # Conv Block 1
            self._conv_block(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(2, 2),

            # Conv Block 2
            self._conv_block(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),

            # Conv Block 3
            self._conv_block(192, 128, kernel_size=1, stride=1, padding=0),
            self._conv_block(128, 256, kernel_size=3, stride=1, padding=1),
            self._conv_block(256, 256, kernel_size=1, stride=1, padding=0),
            self._conv_block(256, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),

            # Conv Block 4 (repeat pattern)
            *[block for _ in range(4) for block in (
                self._conv_block(512, 256, kernel_size=1, stride=1, padding=0),
                self._conv_block(256, 512, kernel_size=3, stride=1, padding=1),
            )],
            self._conv_block(512, 512, kernel_size=1, stride=1, padding=0),
            self._conv_block(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),

            # Conv Block 5
            self._conv_block(1024, 512, kernel_size=1, stride=1, padding=0),
            self._conv_block(512, 1024, kernel_size=3, stride=1, padding=1),
            self._conv_block(1024, 512, kernel_size=1, stride=1, padding=0),
            self._conv_block(512, 1024, kernel_size=3, stride=1, padding=1),
            self._conv_block(1024, 1024, kernel_size=3, stride=1, padding=1),
            self._conv_block(1024, 1024, kernel_size=3, stride=2, padding=1),

            # Final Conv
            self._conv_block(1024, 1024, kernel_size=3, stride=1, padding=1),
            self._conv_block(1024, 1024, kernel_size=3, stride=1, padding=1),
        )

        # Detection head
        self.detector = nn.Sequential(
            nn.AdaptiveAvgPool2d((grid_size, grid_size)),
            nn.Flatten(),
            nn.Linear(1024 * grid_size * grid_size, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(4096, grid_size * grid_size * (num_boxes * 5 + num_classes)),
        )

        self._initialize_weights()

    def _conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.detector(x)
        batch_size = x.size(0)
        output_dim = self.num_boxes * 5 + self.num_classes
        return x.view(batch_size, self.grid_size, self.grid_size, output_dim)
