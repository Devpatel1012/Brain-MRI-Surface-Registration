import torch
import torch.nn as nn
from image_encoder import FeatureVolumeEncoder, TrilinearSampler
from Encoder import Encoder

class AttentionBasedEncoder(nn.Module):
    def __init__(self, image_shape=(96, 144, 192), embed_dim=62):
        super().__init__()
        self.cnn = FeatureVolumeEncoder(in_channel=1, embed_dim=embed_dim)
        self.sampler = TrilinearSampler(image_shape=image_shape)
        
        self.mesh_encoder = Encoder(in_channels=embed_dim, embed_dim=embed_dim, pool_config=[])

    def forward(self, mri_volume, stacked_template_vertices):
  
        dense_volume = self.cnn(mri_volume) 
        
        vertex_tokens = self.sampler(dense_volume, stacked_template_vertices) 
        
        output_features, final_mesh = self.mesh_encoder(vertex_tokens, mesh=None) 
        
        return output_features, final_mesh