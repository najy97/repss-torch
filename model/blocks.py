import torch.nn as nn


class ConvBlock3D(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding):
        super(ConvBlock3D, self).__init__()
        self.convblock3d = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size, stride, padding),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock3d(x)


class DeConvBlock3D(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding):
        super(DeConvBlock3D, self).__init__()
        self.deconvblock3d = nn.Sequential(
            nn.ConvTranspose3d(ch_in, ch_out, kernel_size, stride, padding),
            nn.BatchNorm3d(ch_out),
            nn.ELU()
        )

    def forward(self, x):
        return self.deconvblock3d(x)
