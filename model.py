import torch.nn as nn


class MyCnn(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),  # (28-3+2*1)/1 + 1= 28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (28-2)/2 + 1 = 14

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # (14-3+2*1)/1 +1= 14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (14-2)/2 +1=7
        )

        self.fc = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
