import torch
import torch.nn as nn
import torch.nn.functional as F

class DeformationGraphDecoder(nn.Module):
    def __init__(self, embed_dim=62, sigma=5.0):
        super(DeformationGraphDecoder,self).__init__()
        self.sigma = sigma

        self.displacement_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim //2 ,3)
        )

    def get_interpolation_weights(self, dense_verts, control_verts): # Fixed spelling here!
        dist_matrix  = torch.cdist(dense_verts, control_verts)
        weights = torch.exp(-(dist_matrix ** 2) / (2 * self.sigma ** 2))
        weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-8)
        return weights
    
    def forward(self, control_features, template_dense_verts, template_control_verts):
        sparse_displacement = self.displacement_predictor(control_features)

        weights = self.get_interpolation_weights(template_dense_verts, template_control_verts)

        dense_displacements = torch.bmm(weights, sparse_displacement)

        deformed_mesh = template_dense_verts + dense_displacements

        return deformed_mesh, sparse_displacement
    
    