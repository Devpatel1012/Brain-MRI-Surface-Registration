import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm3D(nn.Module):
    """ Channels-first LayerNorm for 3D inputs """
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
    
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x

class ConvNeXt3DBlock(nn.Module):
    def __init__(self, dim):
        super(ConvNeXt3DBlock, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm3D(dim)

        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)

        x = x.permute(0, 2, 3, 4, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        x = x.permute(0, 4, 1, 2, 3)
        return shortcut + x

class FeatureVolumeEncoder(nn.Module):
    def __init__(self, in_channel=1, embed_dim=62):
        super(FeatureVolumeEncoder, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channel, embed_dim, kernel_size=4, stride=4, padding=1),
            LayerNorm3D(embed_dim)
        )

        self.stages = nn.Sequential(
            ConvNeXt3DBlock(embed_dim),
            ConvNeXt3DBlock(embed_dim),
            ConvNeXt3DBlock(embed_dim)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return x
    
class TrilinearSampler(nn.Module):
    def __init__(self, image_shape):
        super(TrilinearSampler, self).__init__()
        self.register_buffer('image_shape', torch.tensor(image_shape, dtype=torch.float32))

    def forward(self, feature_volume, mesh_coords):
        B, N, _ = mesh_coords.shape

        half_size = (self.image_shape - 1) / 2.0
        norm_coords = (mesh_coords - half_size) / half_size

        norm_coords = norm_coords.view(B, 1, 1, N, 3)

        sampled_features = F.grid_sample(
            feature_volume, 
            norm_coords, 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=True
        )

        sampled_features = sampled_features.squeeze(2).squeeze(2).transpose(1, 2)
        return sampled_features