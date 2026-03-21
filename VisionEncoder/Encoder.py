import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from topofit.topofit import ico,utils
from layers.mesh_conv import MeshConv
from layers.mesh_pool import MeshPool

class MeshCNNBlock(nn.Module):
    def __init__(self,in_channel,out_channel, pool_ratio = None):
        super(MeshCNNBlock,self).__init__()

        # meshconvoluation layer
        self.mesh_conv = MeshConv(in_channel,out_channel)

        # BatchNorm Normalization layer
        self.batch_norm = nn.BatchNorm1d(out_channel)

        # Relu Activation layer
        self.relu = nn.ReLU(inplace = True)

        # Edge Pooling layer
        if pool_ratio is not None:
            self.pool = MeshPool(pool_ratio)
        else:
            self.pool = None
    
    def forward(self, x, mesh):
        if not isinstance(mesh, list):
            mesh_list = [mesh]
        else:
            mesh_list = mesh
        x = self.mesh_conv(x,mesh_list)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.batch_norm(x)
        x = self.relu(x)

        if self.pool is not None:
            x, mesh_list = self.pool(x, mesh_list)
            if not isinstance(mesh, list):
                return x, mesh_list[0]
            return x, mesh_list

        return x, mesh


class LinearProjection (nn.Module):
    def __init__(self, in_channels, embed_dim):
        super(LinearProjection, self).__init__()
        self.proj = nn.Linear(in_channels, embed_dim)
    
    def forward(self, x):
        x = x.permute(0,2,1).contiguous() #(B,N,C)
        x = self.proj(x) #(B,N,embed_dim)
        return x

class TokenFeatureProjection(nn.Module):
    def __init__ (self, embed_dim):
        super(TokenFeatureProjection,self).__init__()
        self.embedding = nn.Linear(embed_dim,embed_dim)
    
    def forward(self, x):
        x = self.embedding(x) #(B,N,embed_dim)
        return x

class TokenLayerNorm(nn.Module):
    def __init__(self,embed_dim):
        super(TokenLayerNorm,self).__init__()
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self,x):
        return self.norm(x) #(B,N,embed_dim)

class MeshTokenEmbeddingBlock(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super(MeshTokenEmbeddingBlock,self).__init__()
        self.linear_projection = LinearProjection(in_channels,embed_dim)
        self.token_feature_proj = TokenFeatureProjection(embed_dim)
        self.Layer_norm = TokenLayerNorm(embed_dim)

    def forward(self,x):
        x = self.linear_projection(x)
        x = self.token_feature_proj(x)
        x = self.Layer_norm(x)

        return x
    
# class GeodesicWindowPartition(nn.Module):
#     # Partition mesh tokens into geodesic windows using ico-mapping.
#     # def __init__(self, high_order=6, window_order=2):
#     #     super(GeodesicWindowPartition, self).__init__()
        
#     #     # Load the mapping from high-res vertices to low-res 'window' faces
#     #     # This is the "genealogy" strategy discussed
#     #     mapping = ico.get_mapping(high_order, window_order)
#     #     window_ids = torch.from_numpy(mapping).long()
        
#     #     # register_buffer ensures the map moves to GPU with the model
#     #     self.register_buffer('window_ids', window_ids)
#     #     self.num_windows = len(torch.unique(window_ids))

#     def __init__(self, high_order=6, window_order=2):
#         super(GeodesicWindowPartition, self).__init__()
        
#         base_vertices = ico.get_ico_data(f'ico-{window_order}-vertices')
#         num_parent_vertices = base_vertices.shape[0] # Should be 162
#         current_map = np.arange(num_parent_vertices)
        
#         for order in range(window_order, high_order):
#             mapping_indices = ico.get_ico_data(f'mapping-{order}-to-{order+1}-indices')
            
#             mapping_indices = mapping_indices.astype(int) - 1
            
#             mapping_indices = np.clip(mapping_indices, 0, num_parent_vertices - 1)
            
#             current_map = current_map[mapping_indices]
            
#             num_parent_vertices = current_map.shape[0]
            
#         window_ids = torch.as_tensor(current_map, dtype=torch.long)        
#         self.register_buffer('window_ids', window_ids)
#         self.num_windows = len(torch.unique(window_ids))
    
#     def forward(self, x):
#         """
#         Groups vertices into windows, ensuring indices are within valid bounds.
#         """
#         B, N, C = x.shape
#         device = x.device
        
#         window_ids = self.window_ids.to(device).view(-1)
        
#         if window_ids.shape[0] > N:
#             window_ids = window_ids[:N]
#         elif window_ids.shape[0] < N:
#             padding = torch.zeros(N - window_ids.shape[0], dtype=torch.long, device=device)
#             window_ids = torch.cat([window_ids, padding])

#         indices = torch.argsort(window_ids)
#         x_sorted = x[:, indices, :]
        
#         counts = torch.bincount(window_ids, minlength=self.num_windows)
#         max_win_size = int(counts.max()) 
        
#         x_windows = torch.zeros(B, self.num_windows, max_win_size, C, device=device)
        
#         curr = 0
#         for i in range(self.num_windows):
#             count = int(counts[i])
#             if count > 0:
#                 x_windows[:, i, :count, :] = x_sorted[:, curr:curr+count, :]
#             curr += count
            
#         windows = x_windows.view(-1, max_win_size, C)
        
#         return windows, indices, counts
    
#     def reverse(self, windows, indices, counts, B, N):
#         """
#         Restores windows back to the original Order 6 mesh layout.
#         """
#         C = windows.shape[-1]
#         max_win_size = windows.shape[1]
        
#         # Reshape to (Batch, 162, 256, Channels)
#         x_windows = windows.view(B, self.num_windows, max_win_size, C)
        
#         # Reconstruct the sorted tensor while removing the padding
#         x_sorted = torch.zeros(B, N, C, device=windows.device)
#         curr = 0
#         for i in range(self.num_windows):
#             count = int(counts[i])
#             x_sorted[:, curr:curr+count, :] = x_windows[:, i, :count, :]
#             curr += count
            
#         # Invert the sorting permutation
#         inv_indices = torch.argsort(indices)
#         return x_sorted[:, inv_indices, :]

class GeodesicWindowPartition(nn.Module):
    def __init__(self, high_order=6, window_order=2):
        super(GeodesicWindowPartition, self).__init__()
        
        # 1. Load the 3D coordinates of both mesh orders
        base_vertices = ico.get_ico_data(f'ico-{window_order}-vertices') # Shape: (162, 3)
        high_res_vertices = ico.get_ico_data(f'ico-{high_order}-vertices') # Shape: (40962, 3)
        
        # 2. Convert to PyTorch tensors
        centers = torch.tensor(base_vertices, dtype=torch.float32)
        vertices = torch.tensor(high_res_vertices, dtype=torch.float32)
        
        # 3. Calculate distance from every high-res vertex to every window center
        # This guarantees perfect, balanced, geographically local windows!
        distances = torch.cdist(vertices, centers) 
        
        # 4. Assign each vertex to its closest center (Voronoi Partitioning)
        window_ids = torch.argmin(distances, dim=1) # Shape: (40962,)
        
        self.register_buffer('window_ids', window_ids)
        self.num_windows = len(torch.unique(window_ids))
    
    def forward(self, x):
        """ Groups vertices into windows, ensuring indices are within valid bounds. """
        B, N, C = x.shape
        device = x.device
        
        window_ids = self.window_ids.to(device).view(-1)
        
        if window_ids.shape[0] > N:
            window_ids = window_ids[:N]
        elif window_ids.shape[0] < N:
            padding = torch.zeros(N - window_ids.shape[0], dtype=torch.long, device=device)
            window_ids = torch.cat([window_ids, padding])

        indices = torch.argsort(window_ids)
        x_sorted = x[:, indices, :]
        
        counts = torch.bincount(window_ids, minlength=self.num_windows)
        max_win_size = int(counts.max()) 
        
        x_windows = torch.zeros(B, self.num_windows, max_win_size, C, device=device)
        
        curr = 0
        for i in range(self.num_windows):
            count = int(counts[i])
            if count > 0:
                x_windows[:, i, :count, :] = x_sorted[:, curr:curr+count, :]
            curr += count
            
        windows = x_windows.view(-1, max_win_size, C)
        
        return windows, indices, counts
    
    def reverse(self, windows, indices, counts, B, N):
        """ Restores windows back to the original mesh layout. """
        C = windows.shape[-1]
        max_win_size = windows.shape[1]
        
        # Reshape back to windows
        x_windows = windows.view(B, self.num_windows, max_win_size, C)
        
        # Reconstruct the sorted tensor while removing the padding
        x_sorted = torch.zeros(B, N, C, device=windows.device)
        curr = 0
        for i in range(self.num_windows):
            count = int(counts[i])
            if count > 0:
                x_sorted[:, curr:curr+count, :] = x_windows[:, i, :count, :]
            curr += count
            
        # Invert the sorting permutation
        inv_indices = torch.argsort(indices)
        return x_sorted[:, inv_indices, :]

class QKVProjection(nn.Module):
    def __init__(self,embed_dim,num_heads):
        super(QKVProjection,self).__init__()

        self.num_heads = num_heads
        self.head_dim = embed_dim//num_heads

        self.qkv = nn.Linear(embed_dim,embed_dim*3)
    
    def forward(self,x):
        B,N,C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(B,N,3,self.num_heads,self.head_dim)

        qkv = qkv.permute(2,0,3,1,4)
        q,k,v =qkv[0],qkv[1],qkv[2]

        return q, k, v
    
class RelativeGeometryEncoding(nn.Module):
    def __init__(self, num_heads):
        super(RelativeGeometryEncoding,self).__init__()

        self.num_heads = num_heads
        self.relative_bias = nn.Parameter(torch.zeros(num_heads,1,1))

    def forward(self,attn):
        attn = attn+self.relative_bias
        return attn
    
class WindowAttention(nn.Module):
    def __init__(self,embed_dim,num_heads):
        super(WindowAttention,self).__init__()
        
        self.num_heads = num_heads
        self.scale = (embed_dim//num_heads) ** -0.5

        self.qkv_proj = QKVProjection(embed_dim,num_heads)
        self.relative_encoding = RelativeGeometryEncoding(num_heads)

    def forward(self,x):
        q,k,v = self.qkv_proj(x)

        attn  = (q @ k.transpose(-2,-1)) * self.scale

        attn = self.relative_encoding(attn)
        attn = F.softmax(attn,dim = -1) 
        out = attn @ v
        out = out.transpose(1,2).reshape(x.shape[0],x.shape[1],-1)

        return out

class AttentionOutputProjection(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionOutputProjection,self).__init__()
        self.proj = nn.Linear(embed_dim,embed_dim)

    def forward(self, x):
        return self.proj(x)
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim,hidden_dim):
        super(FeedForwardNetwork,self).__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim,hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim,embed_dim)
        )
    
    def forward(self,x):
        return self.net(x)


class WindowBasedGraphAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        window_size=None, 
        mlp_ratio=4,
        high_order=6,
        window_order=2
    ):
        super(WindowBasedGraphAttentionBlock, self).__init__()

        # Updated to use hierarchical partitioning
        self.partition = GeodesicWindowPartition(high_order = 6, window_order = 2)
        self.attention = WindowAttention(embed_dim, num_heads)
        self.proj = AttentionOutputProjection(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim, embed_dim * mlp_ratio)

    # def forward(self, x, mesh):
    #     shortcut = x
    #     B, N, C = x.shape

    #     # FIX: Correctly unpack the 3 values returned by the updated partitioner
    #     # windows: (B*162, 256, C), indices: (N,), counts: (162,)
    #     windows, indices, counts = self.partition(x)

    #     # Apply Window Attention
    #     attn_windows = self.attention(self.norm1(windows))
    #     attn_windows = self.proj(attn_windows)

    #     # FIX: Pass 'indices' and 'counts' to the reverse method for correct reconstruction
    #     x_recovered = self.partition.reverse(attn_windows, indices, counts, B, N)

    #     x = shortcut + x_recovered
    #     x = x + self.ffn(self.norm2(x))

    #     return x, mesh
    def forward(self, x, mesh):
        shortcut = x
        B, N, C = x.shape

        # Standard forward pass instead of checkpoint
        windows, indices, counts = self.partition(x)
        attn_windows = self.attention(self.norm1(windows))
        attn_windows = self.proj(attn_windows)
        
        x_recovered = self.partition.reverse(
            attn_windows, 
            indices, 
            counts, 
            B, 
            N
        )

        x = shortcut + x_recovered
        x = x + self.ffn(self.norm2(x))
        return x, mesh

    
class GraphPooling(nn.Module):
    def __init__(self, mesh_info, current_order):
        super(GraphPooling, self).__init__()
        self.mesh_info = mesh_info
        self.current_order = current_order
    
    def forward(self, x, mesh=None):

        B, N, C = x.shape
        pooled_list = []

        for b in range(B):
            pooled = utils.pool(x[b], self.mesh_info)  # (N,C) expected
            pooled_list.append(pooled)

        x = torch.stack(pooled_list, dim=0)

        new_mesh = {
            'order': self.current_order - 1,
            'adjacency': ico.adjancency_indices(self.current_order - 1)
        }

        return x, new_mesh
    
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, nums_heads, mlp_ratio = 4):
        super(TransformerBlock,self).__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim,nums_heads,batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim,embed_dim * mlp_ratio)
    
    def forward(self,x,mesh):

        shortcut = x
        x_norm = self.norm1(x)
        attn_out,_ = self.attention(x_norm,x_norm,x_norm)
        x = shortcut+attn_out
        
        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = shortcut+x

        return x,mesh
    
class Encoder(nn.Module):
    def __init__(self, in_channels, embed_dim, pool_config):
        super(Encoder, self).__init__()

        # MeshCNN Blocks - 3
        self.mesh_cnn = nn.ModuleList([
            MeshCNNBlock(in_channels, embed_dim, pool_ratio=None),
            MeshCNNBlock(embed_dim, embed_dim, pool_ratio=None),
            MeshCNNBlock(embed_dim, embed_dim, pool_ratio=None)
        ])
        
        # Token embedding block
        self.embedding = MeshTokenEmbeddingBlock(embed_dim, embed_dim)

        # Window Based Graph Attention Blocks - 4
        self.win_gat = nn.ModuleList([
            WindowBasedGraphAttentionBlock(embed_dim, num_heads=2)
            for _ in range(4)
        ])

        self.pooling = None
        if pool_config and 'mesh_info' in pool_config[0]:
            mapping_keys = ['pooling_a', 'pooling_b', 'pooling_shape_a', 'pooling_shape_b']
            if all(key in pool_config[0]['mesh_info'] for key in mapping_keys):
                self.pooling = GraphPooling(
                    pool_config[0]['mesh_info'], 
                    current_order=pool_config[0].get('order', 6)
                )
            else:
                print("Warning: Missing pooling keys. GraphPooling layer disabled.")

        # Transformer Blocks - 3
        self.global_control_transformer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=4, 
            dim_feedforward=embed_dim * 4, 
            batch_first=True
        )

    def edges_to_vertices(self, edge_features, mesh_list, num_vertices=40962):
        """ Maps (B, Channels, Edges) -> (B, Channels, Vertices) """
        B, C, E = edge_features.shape
        device = edge_features.device

        batch_vertices = []
        for b in range(B):
            # Get the raw edge indices (E, 2)
            edges = torch.tensor(mesh_list[b].edges, dtype=torch.long, device=device)
            
            # Tensors to hold vertex accumulations and counts
            v_feats = torch.zeros((C, num_vertices), device=device)
            counts = torch.zeros(num_vertices, device=device)
            
            src = edge_features[b] # Shape: (C, E)
            
            # Expand indices so we can use PyTorch's ultra-fast scatter_add
            idx1 = edges[:, 0].unsqueeze(0).expand(C, -1)
            idx2 = edges[:, 1].unsqueeze(0).expand(C, -1)
            
            # Add edge features to both vertices that make up the edge
            v_feats.scatter_add_(1, idx1, src)
            v_feats.scatter_add_(1, idx2, src)
            
            # Count how many edges touch each vertex to calculate the average
            counts.scatter_add_(0, edges[:, 0], torch.ones(E, device=device))
            counts.scatter_add_(0, edges[:, 1], torch.ones(E, device=device))
            
            # Divide by counts to get the average feature per vertex
            v_feats = v_feats / counts.unsqueeze(0).clamp(min=1)
            batch_vertices.append(v_feats)
            
        return torch.stack(batch_vertices, dim=0) # Shape: (B, C, V)

    def forward(self, x, mesh):
        # MeshCNN expects a list
        if not isinstance(mesh, list):
            mesh_list = [mesh]
        else:
            mesh_list = mesh

        # 1. MeshCNN Stages (Processes EDGES)
        for block in self.mesh_cnn:
            x, mesh_list = block(x, mesh_list)

        # --- THE REVERSE BRIDGE ---
        # Convert the processed Edges back to 40,962 Vertices!
        x = self.edges_to_vertices(x, mesh_list)

        # Mesh is no longer a list for the rest of the pipeline
        mesh_obj = mesh_list[0]

        # 2. Embedding Stage (Processes VERTICES)
        x = self.embedding(x)

        # 3. Window Attention Stages (Processes VERTICES safely in small chunks!)
        for block in self.win_gat:
            x, mesh_obj = block(x, mesh_obj)

        # 4. Global Transformer Stages 
        # (TEMPORARILY DISABLED to prevent 50GB OOM explosion. 
        # Turn on pooling first if you want to use these!)
        
        # for block in self.transformer:
        #     x, mesh = block(x, mesh)

        dense_features = x # Shape: (B, 40962, 62)
        
        # Slicing the first 642 vertices gives us the perfect Order 3 control points!
        control_features = x[:, :642, :] # Shape: (B, 642, 62)

        control_features = self.global_control_transformer(control_features)

        # Return both feature sets
        return dense_features, control_features, mesh_obj