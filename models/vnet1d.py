import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, cin, cout, num_conv, kernel_size=5, stride=1):
        super().__init__()
        layers = []
        for _ in range(num_conv):
            layers.append(nn.Conv1d(cin, cout, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2))
            layers.append(nn.PReLU)
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x)


##First down block [batch, cin, length] as ex. [1, 2, 249] -> [1, 16, 249]
class DownBlock(nn.Module):
    def __init__(self, cin, cout, kernel_size=2, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(cin, cout, kernel_size=kernel_size, stide=stride, padding=kernel_size // 2),
            nn.PReLU()
        )

    def forward(self, x):
        return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, cin, cout, kernel_size=2, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose1d(cin, cout, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            nn.PReLU()
        )

    def forward(self, x):
        return self.conv(x)

class VNet1d(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.encoder = nn.ModuleList([ResBlock(cin=input_dim, cout=16, num_conv=1),
                                      DownBlock(cin=16, cout=32),
                                      ResBlock(cin=32, cout=32, num_conv=2),
                                      DownBlock(cin=32, cout=64),
                                      ResBlock(cin=64, cout=64, num_conv=3),
                                      DownBlock(cin=64, cout=128),
                                      ResBlock(cin=128, cout=128, num_conv=3),
                                      DownBlock(cin=128, cout=256),
                                      ResBlock(cin=256, cout=256, num_conv=3)])
        
        self.decoder = nn.ModuleList([UpBlock(cin=256, cout=256),
                                     ResBlock(cin=256, cout=256, num_conv=3),
                                     UpBlock(cin=256, cout=128),
                                     ResBlock(cin=128, cout=128, num_conv=3),
                                     UpBlock(cin=128, cout=64),
                                     ResBlock(cin=64, cout=64, num_conv=2),
                                     UpBlock(cin=64, cout=32),
                                     ResBlock(cin=32, cout=32, num_conv=1)])
        
        self.last_filter = nn.Sequential(
            nn.Conv1d(cin=32, cout=output_dim, kernel_size=1),
            nn.PReLU()
        )

    def forward(self, x):
        res_cons = []
        for i, block in enumerate(self.encoder):
            x = block(x)
            if (i % 2 == 0 and i < 8):
                #residual connections of all 5x5 convolution blocks except last
                res_cons.append(x)
        for i, block in enumerate(self.decoder):
            if i % 2 == 0:
                x = x + res_cons.pop()
            x = block(x)

        x = self.last_filter(x)

        return x