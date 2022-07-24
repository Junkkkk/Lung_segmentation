import torch
from torch import nn
import torch.nn.functional as F


__all__ = ['UNet', 'NestedUNet']


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        return out

class UpConv(nn.Module):
        """
        A helper Module that performs 2 convolutions and 1 UpConvolution.
        A ReLU activation follows each convolution.
        """
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels

            self.upconv = nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size=2, stride=2)

            self.conv1 = nn.Conv2d(self.in_channels, self.out_channels,  kernel_size=3, padding=0)
            self.conv2 = nn.Conv2d(self.out_channels, self.out_channels,  kernel_size=3, padding=0)

        def forward(self, from_down, from_up):
            """ Forward pass
            Arguments:
                from_down: tensor from the encoder pathway
                from_up: upconv'd tensor from the decoder pathway
            """
            from_up = self.upconv(from_up)

            lower = int((from_down.shape[2] - from_up.shape[2]) / 2)
            upper = int(from_down.shape[2] - lower)
            from_down = from_down[:, :, lower:upper, lower:upper]

            x = torch.cat([from_up, from_down], dim=1)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))

            return x


class UNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, **kwargs):
        super().__init__()

        nb_filter = [64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.conv0_0 = DownConv(in_channels, nb_filter[0])
        self.conv1_0 = DownConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DownConv(nb_filter[1], nb_filter[2])
        self.conv3_0 = DownConv(nb_filter[2], nb_filter[3])
        self.conv4_0 = DownConv(nb_filter[3], nb_filter[4])

        self.conv3_1 = UpConv(nb_filter[4], nb_filter[3])
        self.conv2_2 = UpConv(nb_filter[3], nb_filter[2])
        self.conv1_3 = UpConv(nb_filter[2], nb_filter[1])
        self.conv0_4 = UpConv(nb_filter[1], nb_filter[0])

        self.conv_final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(x3_0, x4_0)
        x2_2 = self.conv2_2(x2_0, x3_1)
        x1_3 = self.conv1_3(x1_0, x2_2)
        x0_4 = self.conv0_4(x0_0, x1_3)

        output = self.conv_final(x0_4)

        return output


class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = DownConv(input_channels, nb_filter[0])
        self.conv1_0 = DownConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DownConv(nb_filter[1], nb_filter[2])
        self.conv3_0 = DownConv(nb_filter[2], nb_filter[3])
        self.conv4_0 = DownConv(nb_filter[3], nb_filter[4])

        self.conv0_1 = UpConv(nb_filter[1], nb_filter[0])
        self.conv1_1 = UpConv(nb_filter[2], nb_filter[1])
        self.conv0_2 = UpConv(nb_filter[3], nb_filter[2])
        self.conv3_1 = UpConv(nb_filter[4], nb_filter[3])

        self.conv0_2 = UpConv(nb_filter[1], nb_filter[0])
        self.conv1_2 = UpConv(nb_filter[2], nb_filter[1])
        self.conv0_2 = UpConv(nb_filter[3], nb_filter[2])
        self.conv3_1 = UpConv(nb_filter[4], nb_filter[3])

        self.conv0_2 = UpConv(nb_filter[1], nb_filter[0])
        self.conv1_2 = UpConv(nb_filter[2], nb_filter[1])
        self.conv2_2 = UpConv(nb_filter[3], nb_filter[2])

        self.conv0_3 = UpConv(nb_filter[1], nb_filter[0])
        self.conv1_3 = UpConv(nb_filter[2], nb_filter[1])

        self.conv0_4 = UpConv(nb_filter[1], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(x3_0, x4_0)
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1], 1 ), x3_1)
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2],1), x2_2)
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3], 1), x1_3)

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output