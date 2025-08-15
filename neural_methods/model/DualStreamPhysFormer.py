"""This file implements a dual-stream PhysFormer model that processes RGB and depth videos in parallel,
   then fuses the features before tokenization. Based on the original PhysFormer implementation:
   https://github.com/ZitongYu/PhysFormer

   The dual-stream approach involves:
   1. Parallel RGB and Depth stems
   2. Feature fusion (concatenation) before tokenization
   3. Standard transformer processing on the fused features
"""

import numpy as np
from typing import Optional
import torch
from torch import nn
from torch import Tensor 
from torch.nn import functional as F
import math

def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)

'''
Temporal Center-difference based Convolutional layer (3D version)
theta: control the percentage of original convolution and centeral-difference convolution
'''
class CDC_T(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.6):

        super(CDC_T, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            [C_out, C_in, t, kernel_size, kernel_size] = self.conv.weight.shape

            # only CD works on temporal kernel size>1
            if self.conv.weight.shape[2] > 1:
                kernel_diff = self.conv.weight[:, :, 0, :, :].sum(2).sum(2) + self.conv.weight[:, :, 2, :, :].sum(
                    2).sum(2)
                kernel_diff = kernel_diff[:, :, None, None, None]
                out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                                    padding=0, dilation=self.conv.dilation, groups=self.conv.groups)
                return out_normal - self.theta * out_diff

            else:
                return out_normal


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

class MultiHeadedSelfAttention_TDC_gra_sharp(nn.Module):
    """Multi-Headed Dot Product Attention with depth-wise Conv3d"""
    def __init__(self, dim, num_heads, dropout, theta):
        super().__init__()
        
        self.proj_q = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),  
            nn.BatchNorm3d(dim),
        )
        self.proj_k = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),  
            nn.BatchNorm3d(dim),
        )
        self.proj_v = nn.Sequential(
            nn.Conv3d(dim, dim, 1, stride=1, padding=0, groups=1, bias=False),
        )
        
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None # for visualization

    def forward(self, x, gra_sharp):    # [B, 4*4*40, 128]
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        
        [B, P, C]=x.shape
        x = x.transpose(1, 2).view(B, C, P//16, 4, 4)      # [B, dim, 40, 4, 4]
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q = q.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        k = k.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        v = v.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / gra_sharp

        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h, scores

class PositionWiseFeedForward_ST(nn.Module):
    """FeedForward Neural Networks for each position"""
    def __init__(self, dim, ff_dim):
        super().__init__()
        
        self.fc1 = nn.Sequential(
            nn.Conv3d(dim, ff_dim, 1, stride=1, padding=0, bias=False),  
            nn.BatchNorm3d(ff_dim),
            nn.ELU(),
        )
        
        self.STConv = nn.Sequential(
            nn.Conv3d(ff_dim, ff_dim, 3, stride=1, padding=1, groups=ff_dim, bias=False),  
            nn.BatchNorm3d(ff_dim),
            nn.ELU(),
        )
        
        self.fc2 = nn.Sequential(
            nn.Conv3d(ff_dim, dim, 1, stride=1, padding=0, bias=False),  
            nn.BatchNorm3d(dim),
        )

    def forward(self, x):    # [B, 4*4*40, 128]
        [B, P, C]=x.shape
        x = x.transpose(1, 2).view(B, C, P//16, 4, 4)      # [B, dim, 40, 4, 4]
        x = self.fc1(x)		              # x [B, ff_dim, 40, 4, 4]
        x = self.STConv(x)		          # x [B, ff_dim, 40, 4, 4]
        x = self.fc2(x)		              # x [B, dim, 40, 4, 4]
        x = x.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        
        return x

class Block_ST_TDC_gra_sharp(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, ff_dim, dropout, theta):
        super().__init__()
        self.attn = MultiHeadedSelfAttention_TDC_gra_sharp(dim, num_heads, dropout, theta)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward_ST(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, gra_sharp):
        Atten, Score = self.attn(self.norm1(x), gra_sharp)
        h = self.drop(self.proj(Atten))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        return x, Score

class Transformer_ST_TDC_gra_sharp(nn.Module):
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout, theta):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block_ST_TDC_gra_sharp(dim, num_heads, ff_dim, dropout, theta) for _ in range(num_layers)])

    def forward(self, x, gra_sharp):
        for block in self.blocks:
            x, Score = block(x, gra_sharp)
        return x, Score

class DualStreamPhysFormer(nn.Module):
    """Dual-stream PhysFormer with RGB and Depth stems, feature fusion, and transformer processing"""

    def __init__(
        self, 
        name: Optional[str] = None, 
        pretrained: bool = False, 
        patches: int = 16,
        dim: int = 768,
        ff_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        attention_dropout_rate: float = 0.0,
        dropout_rate: float = 0.2,
        representation_size: Optional[int] = None,
        load_repr_layer: bool = False,
        classifier: str = 'token',
        in_channels_rgb: int = 3, 
        in_channels_depth: int = 1,
        frame: int = 580,
        theta: float = 0.2,
        image_size: Optional[int] = None,
        rgb_stem_channels: Optional[list] = None,
        depth_stem_channels: Optional[list] = None,
    ):
        super().__init__()

        self.image_size = image_size  
        self.frame = frame  
        self.dim = dim
        self.patch_size = patches              

        # Image and patch sizes
        t, h, w = as_tuple(image_size)  # tube sizes
        ft, fh, fw = as_tuple(patches)  # patch sizes, ft = 4 ==> 160/4=40
        gt, gh, gw = t//ft, h // fh, w // fw  # number of patches
        seq_len = gh * gw * gt

        # Set default stem channel configurations if not provided
        if rgb_stem_channels is None:
            rgb_stem_channels = [dim//4, dim//2, dim]
        if depth_stem_channels is None:
            depth_stem_channels = [dim//8, dim//4, dim//2]

        # RGB Stem (original PhysFormer stem)
        self.rgb_stem0 = nn.Sequential(
            nn.Conv3d(in_channels_rgb, rgb_stem_channels[0], [1, 5, 5], stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(rgb_stem_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        
        self.rgb_stem1 = nn.Sequential(
            nn.Conv3d(rgb_stem_channels[0], rgb_stem_channels[1], [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(rgb_stem_channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        
        self.rgb_stem2 = nn.Sequential(
            nn.Conv3d(rgb_stem_channels[1], rgb_stem_channels[2], [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(rgb_stem_channels[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )

        # Depth Stem (specialized for depth data)
        self.depth_stem0 = nn.Sequential(
            nn.Conv3d(in_channels_depth, depth_stem_channels[0], [1, 5, 5], stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(depth_stem_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        
        self.depth_stem1 = nn.Sequential(
            nn.Conv3d(depth_stem_channels[0], depth_stem_channels[1], [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(depth_stem_channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        
        self.depth_stem2 = nn.Sequential(
            nn.Conv3d(depth_stem_channels[1], depth_stem_channels[2], [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(depth_stem_channels[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )

        # Feature fusion layer - combines RGB and depth features
        total_channels = rgb_stem_channels[2] + depth_stem_channels[2]
        self.feature_fusion = nn.Sequential(
            nn.Conv3d(total_channels, dim, 1, stride=1, padding=0),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True),
        )
        
        # Patch embedding for the fused features
        self.patch_embedding = nn.Conv3d(dim, dim, kernel_size=(ft, fh, fw), stride=(ft, fh, fw))
        
        # Transformer blocks (same as original PhysFormer)
        self.transformer1 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers//3, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        self.transformer2 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers//3, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        self.transformer3 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers//3, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        
        # Upsampling and final layers
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(dim, dim, [3, 1, 1], stride=1, padding=(1,0,0)),   
            nn.BatchNorm3d(dim),
            nn.ELU(),
        )
        
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(dim, dim//2, [3, 1, 1], stride=1, padding=(1,0,0)),   
            nn.BatchNorm3d(dim//2),
            nn.ELU(),
        )
 
        self.ConvBlockLast = nn.Conv1d(dim//2, 1, 1, stride=1, padding=0)
        
        # Initialize weights
        self.init_weights()
        
    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
        self.apply(_init)

    def forward(self, rgb_data, depth_data, gra_sharp):
        """
        Forward pass for dual-stream PhysFormer
        Args:
            rgb_data: RGB video data [B, 3, T, H, W]
            depth_data: Depth video data [B, 1, T, H, W]
            gra_sharp: Gradient sharpening parameter
        """
        self.frame = rgb_data.shape[1] # Handles variable frame length inputs
        b, c, t, h, w = rgb_data.shape # batch size, # channels, t frames, height, width
        
        
        # Process RGB stream
        rgb_features = self.rgb_stem0(rgb_data)
        rgb_features = self.rgb_stem1(rgb_features)
        rgb_features = self.rgb_stem2(rgb_features)  # [B, rgb_channels, T, H, W]
        
        # print(f"rgb_features shape: {rgb_features.shape}") # [1, 96, 580, 16, 16]
        
        # Process depth stream
        depth_features = self.depth_stem0(depth_data)
        depth_features = self.depth_stem1(depth_features)
        depth_features = self.depth_stem2(depth_features)  # [B, depth_channels, T, H, W]
        
        # print(f"depth_features shape: {depth_features.shape}") # [1, 48, 580, 16, 16]
        
        # Feature fusion: concatenate along channel dimension
        fused_features = torch.cat([rgb_features, depth_features], dim=1) 
        
        # Apply feature fusion layer to get unified representation
        # combines rgb + depth into single feature space
        fused_features = self.feature_fusion(fused_features)  # [B, dim, T, H, W]
        
        # print(f"fused_features shape: {fused_features.shape}") # [1, 96, 580, 16, 16] = 222720
        
        # save fused feature shape for later upsampling
        f_b, f_dim, f_t, f_h, f_w = fused_features.shape
        
        # Patch embedding
        x = self.patch_embedding(fused_features)
        x = x.flatten(2).transpose(1, 2)
        
        # print(f"x shape: {x.shape}") # [1, 2320, 96] = 222720
        
        # Transformer processing
        Trans_features, Score1 = self.transformer1(x, gra_sharp)
        Trans_features2, Score2 = self.transformer2(Trans_features, gra_sharp)
        Trans_features3, Score3 = self.transformer3(Trans_features2, gra_sharp)
        
        # print(f"Trans_features3 shape: {Trans_features3.shape}") # [1, 2320, 96] = 222720

        # Convert back to fused features dimension
        # Revert transformer output back to spatial format
        # Reverse the operations: x = x.flatten(2).transpose(1, 2)
        # First transpose back: (1, 2) -> (2, 1)
        # Then unflatten: restore spatial dimensions
        features_last = Trans_features3.transpose(1, 2)  # [B, dim, seq_len]
        features_last = features_last.view(f_b, f_dim, f_t//4, f_h//4, f_w//4)  # [B, dim, T//4, H//4, W//4]
        
        features_last = self.upsample(features_last)
        features_last = self.upsample2(features_last)
        
        features_last = torch.mean(features_last, 3)     # Average over spatial dimensions
        features_last = torch.mean(features_last, 3)     # Average over spatial dimensions
        rPPG = self.ConvBlockLast(features_last)        # Final 1D convolution
        
        rPPG = rPPG.squeeze(1)

        return rPPG, Score1, Score2, Score3
