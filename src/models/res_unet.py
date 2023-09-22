import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=input_dim,
                out_channels=output_dim,
                kernel_size=3,
                stride=stride,
                padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=output_dim,
                out_channels=output_dim,
                kernel_size=3,
                padding=1
            )
        )

        self.conv_skip = nn.Sequential(
            nn.Conv2d(
                in_channels=input_dim,
                out_channels=output_dim,
                kernel_size=3,
                stride=stride,
                padding=1
            ),
            nn.BatchNorm2d(output_dim)
        )

    def forward(self, x):
        x1 = self.conv_block(x)
        x2 = self.conv_skip(x)

        return x1 + x2


class UpSample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(UpSample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=kernel,
            stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class ResUNet(nn.Module):
    def __init__(self, channel, filters=[64, 128, 256, 512]):
        super(ResUNet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=channel,
                out_channels=filters[0],
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=filters[0],
                out_channels=filters[0],
                kernel_size=3,
                padding=1
            )
        )

        self.input_skip = nn.Sequential(
            nn.Conv2d(
                in_channels=channel,
                out_channels=filters[0],
                kernel_size=3,
                padding=1
            )
        )

        self.residual_conv1 = ResidualConv(
            filters[0], filters[1], stride=2, padding=1)
        self.residual_conv2 = ResidualConv(
            filters[1], filters[2], stride=2, padding=1)

        self.bridge = ResidualConv(filters[2], filters[3], stride=2, padding=1)

        self.upsample1 = UpSample(filters[3], filters[3], kernel=2, stride=2)
        self.up_residual_conv1 = ResidualConv(
            filters[3] + filters[2], filters[2], stride=1, padding=1)

        self.upsample2 = UpSample(filters[2], filters[2], kernel=2, stride=2)
        self.up_residual_conv2 = ResidualConv(
            filters[2] + filters[1], filters[1], stride=1, padding=1)

        self.upsample3 = UpSample(filters[1], filters[1], kernel=2, stride=2)
        self.up_residual_conv3 = ResidualConv(
            filters[1] + filters[0], filters[0], stride=1, padding=1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=filters[0],
                out_channels=1,
                kernel_size=1
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Input shape: [batch_size, 3, 128, 128]
        # Encoder
        # [batch_size, 64, 128, 128]
        x1 = self.input_layer(x) + self.input_skip(x)
        # [batch_size, 128, 64, 64]
        x2 = self.residual_conv1(x1)
        # [batch_size, 256, 32, 32]
        x3 = self.residual_conv2(x2)

        # Bridge
        x4 = self.bridge(x3)  # [batch_size, 512, 16, 16]

        # Decoder
        x4 = self.upsample1(x4)  # [batch_size, 512, 32, 32]

        x5 = torch.cat([x4, x3], dim=1)  # [batch_size, 768, 32, 32]

        x6 = self.up_residual_conv1(x5)  # [batch_size, 256, 32, 32]

        x6 = self.upsample2(x6)  # [batch_size, 256, 64, 64]
        x7 = torch.cat([x6, x2], dim=1)  # [batch_size, 384, 64, 64]

        x8 = self.up_residual_conv2(x7)  # [batch_size, 128, 64, 64]

        x8 = self.upsample3(x8)  # [batch_size, 128, 128, 128]
        x9 = torch.cat([x8, x1], dim=1)  # [batch_size, 192, 128, 128]

        x10 = self.up_residual_conv3(x9)  # [batch_size, 64, 128, 128]

        # Output
        x11 = self.output_layer(x10)  # [batch_size, 1, 128, 128]

        return x11

    def __str__(self):
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return super(ResUNet, self).__str__() + f"\nTrainable parameters: {n_params}"
