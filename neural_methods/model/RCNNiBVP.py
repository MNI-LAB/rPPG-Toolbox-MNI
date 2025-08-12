"""RCNNiBVP - Residual CNN + iBVPNet architecture.
Combining ResidualCNN preprocessing with iBVPNet 3D convolutions for depth + green/NIR processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        """
        A simple residual CNN block for processing depth images.
        """
        super(ResidualCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, depth):
        return self.net(depth)


class ConvBlock3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ConvBlock3D, self).__init__()
        self.conv_block_3d = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding),
            nn.Tanh(),
            nn.InstanceNorm3d(out_channel),
        )

    def forward(self, x):
        return self.conv_block_3d(x)


class DeConvBlock3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(DeConvBlock3D, self).__init__()
        self.deconv_block_3d = nn.Sequential(
            nn.ConvTranspose3d(in_channel, out_channel, kernel_size, stride, padding),
            nn.Tanh(),
            nn.InstanceNorm3d(out_channel),
        )

    def forward(self, x):
        return self.deconv_block_3d(x)


# num_filters
nf = [8, 16, 24, 40, 64]


class encoder_block(nn.Module):
    def __init__(self, in_channel, debug=False):
        super(encoder_block, self).__init__()
        # in_channel, out_channel, kernel_size, stride, padding

        self.debug = debug
        self.spatio_temporal_encoder = nn.Sequential(
            ConvBlock3D(in_channel, nf[0], [1, 3, 3], [1, 1, 1], [0, 1, 1]),
            ConvBlock3D(nf[0], nf[1], [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
            ConvBlock3D(nf[1], nf[2], [1, 3, 3], [1, 1, 1], [0, 1, 1]),
            ConvBlock3D(nf[2], nf[3], [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
            ConvBlock3D(nf[3], nf[4], [1, 3, 3], [1, 1, 1], [0, 1, 1]),
            ConvBlock3D(nf[4], nf[4], [3, 3, 3], [1, 1, 1], [1, 1, 1]),
        )

        self.temporal_encoder = nn.Sequential(
            ConvBlock3D(nf[4], nf[4], [11, 1, 1], [1, 1, 1], [5, 0, 0]),
            ConvBlock3D(nf[4], nf[4], [11, 3, 3], [1, 1, 1], [5, 1, 1]),
            nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),
            ConvBlock3D(nf[4], nf[4], [11, 1, 1], [1, 1, 1], [5, 0, 0]),
            ConvBlock3D(nf[4], nf[4], [11, 3, 3], [1, 1, 1], [5, 1, 1]),
            nn.MaxPool3d((2, 2, 2), stride=(2, 1, 1)),
            ConvBlock3D(nf[4], nf[4], [7, 1, 1], [1, 1, 1], [3, 0, 0]),
            ConvBlock3D(nf[4], nf[4], [7, 3, 3], [1, 1, 1], [3, 1, 1])
        )

    def forward(self, x):
        if self.debug:
            print("Encoder")
            print("x.shape", x.shape)
        st_x = self.spatio_temporal_encoder(x)
        if self.debug:
            print("st_x.shape", st_x.shape)
        t_x = self.temporal_encoder(st_x)
        if self.debug:
            print("t_x.shape", t_x.shape)
        return t_x


class decoder_block(nn.Module):
    def __init__(self, debug=False):
        super(decoder_block, self).__init__()
        self.debug = debug
        self.decoder_block = nn.Sequential(
            DeConvBlock3D(nf[4], nf[3], [7, 3, 3], [2, 2, 2], [2, 1, 1]),
            DeConvBlock3D(nf[3], nf[2], [7, 3, 3], [2, 2, 2], [2, 1, 1])
        )

    def forward(self, x):
        if self.debug:
            print("Decoder")
            print("x.shape", x.shape)
        x = self.decoder_block(x)
        if self.debug:
            print("After decoder x.shape", x.shape)
        return x


class RCNNiBVP(nn.Module):
    def __init__(self, frames=128, in_channels=1, depth_channels=1, debug=False):
        super(RCNNiBVP, self).__init__()
        self.debug = debug
        self.frames = frames
        self.in_channels = in_channels  # green/NIR channel
        self.depth_channels = depth_channels
        
        # Residual CNN for depth preprocessing
        self.RCNN = ResidualCNN(self.depth_channels, self.in_channels)

        # iBVPNet components
        if self.in_channels == 1:
            self.norm = nn.InstanceNorm3d(self.in_channels)
        else:
            print("RCNNiBVP: Unsupported input channels, expected 1 for green/NIR")

        self.ibvpnet = nn.Sequential(
            encoder_block(in_channels, debug),
            decoder_block(debug),
            # spatial adaptive pooling - use frames-1 since we apply diff operation
            nn.AdaptiveMaxPool3d((frames-1, 1, 1)),
            nn.Conv3d(nf[2], 1, [1, 1, 1], stride=1, padding=0)  # nf[2] = 24, from decoder output
        )

    def forward(self, green_nir, depth): 
        """
        Args:
            green_nir: [batch, 1, frames, width, height] - green or NIR channel
            depth: [batch, 1, frames, width, height] - depth channel
        """
        [batch, channel, length, width, height] = green_nir.shape

        if self.debug:
            print("Input green_nir.shape", green_nir.shape)
            print("Input depth.shape", depth.shape)
        
        # Process each frame through ResidualCNN
        # Reshape to process all frames at once: [batch*frames, channels, height, width]
        depth_reshaped = depth.permute(0, 2, 1, 3, 4).contiguous().view(batch * length, self.depth_channels, height, width)
        green_nir_reshaped = green_nir.permute(0, 2, 1, 3, 4).contiguous().view(batch * length, self.in_channels, height, width)
        
        # Apply ResidualCNN to depth
        residual_depth = self.RCNN(depth_reshaped)
        
        # Subtract residual depth from green/NIR
        compensated_green_nir = green_nir_reshaped - residual_depth
        
        # Reshape back to 3D: [batch, channels, frames, height, width]
        compensated_green_nir = compensated_green_nir.view(batch, length, self.in_channels, height, width).permute(0, 2, 1, 3, 4)
        
        if self.debug:
            print("Compensated green_nir.shape", compensated_green_nir.shape)

        # Apply temporal differencing
        x = torch.diff(compensated_green_nir, dim=2)

        if self.debug:
            print("After diff.shape", x.shape)

        # Normalize
        x = self.norm(x)

        if self.debug:
            print("Diff Normalized shape", x.shape)

        # Process through iBVPNet
        feats = self.ibvpnet(x)
        if self.debug:
            print("feats.shape", feats.shape)
        
        # Reshape properly - feats should be [batch, 1, frames-1, 1, 1] after adaptive pooling
        rPPG = feats.view(feats.size(0), -1)  # [batch, frames-1]
        return rPPG


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RCNNiBVP(frames=128, in_channels=1, depth_channels=1, debug=True).to(device)
    
    # Create dummy input
    green_nir = torch.randn(2, 1, 128, 72, 72).to(device)  # batch=2, channels=1, frames=128, height=72, width=72
    depth = torch.randn(2, 1, 128, 72, 72).to(device)      # batch=2, channels=1, frames=128, height=72, width=72
    
    output = model(green_nir, depth)
    print(f"Output shape: {output.shape}")
