import torch
import torch.nn as nn
from image_encoder import FeatureVolumeEncoder, TrilinearSampler
from Encoder import Encoder
from Decoder import DeformationGraphDecoder # <-- Import your new Decoder!

class AttentionBasedEncoder(nn.Module):
    def __init__(self, image_shape=(96, 144, 192), embed_dim=62):
        super().__init__()
        self.cnn = FeatureVolumeEncoder(in_channel=1, embed_dim=embed_dim)
        self.sampler = TrilinearSampler(image_shape=image_shape)
        self.mesh_encoder = Encoder(in_channels=embed_dim, embed_dim=embed_dim, pool_config=[])
        
        # Initialize the Decoder
        self.decoder = DeformationGraphDecoder(embed_dim=embed_dim, sigma=5.0)

    def vertices_to_edges(self, vertex_tokens, mesh_list):
        """ Maps (B, Vertices, Channels) -> (B, Channels, Edges) for MeshCNN """
        batch_edges = []
        for b in range(len(mesh_list)):
            edges = torch.tensor(mesh_list[b].edges, dtype=torch.long, device=vertex_tokens.device)
            v1_feat = vertex_tokens[b, edges[:, 0], :] 
            v2_feat = vertex_tokens[b, edges[:, 1], :] 
            edge_feat = (v1_feat + v2_feat) / 2.0 
            batch_edges.append(edge_feat)
        edge_tensor = torch.stack(batch_edges, dim=0)
        return edge_tensor.permute(0, 2, 1)

    def forward(self, mri_volume, template_vertices, mesh_topology):
        # 1. Image Features & Sampling
        dense_volume = self.cnn(mri_volume) 
        vertex_tokens = self.sampler(dense_volume, template_vertices) 
        
        # 2. Convert to Edges & Run Encoder
        edge_tokens = self.vertices_to_edges(vertex_tokens, mesh_topology)
        
        # Notice we are now unpacking THREE variables here!
        dense_features, control_features, final_mesh = self.mesh_encoder(edge_tokens, mesh=mesh_topology) 
        
        # 3. Prepare Decoder Inputs (Slicing the template the exact same way)
        template_control_vertices = template_vertices[:, :642, :]
        
        # 4. Run the Deformation Graph Decoder!
        deformed_mesh, sparse_displacements = self.decoder(
            control_features=control_features, 
            template_dense_verts=template_vertices, 
            template_control_verts=template_control_vertices
        )
        
        # Return the physical 3D coordinates of your newly deformed brain mesh!
        return deformed_mesh