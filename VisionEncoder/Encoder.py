import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from topofit import ico,utils
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
        x = self.mesh_conv(x,mesh)
        x = self.batch_norm(x)
        x = self.relu(x)

        if self.pool is not None:
            x,mesh = self.pool(x,mesh)

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
    
class GeodesicWindowPartition(nn.Module):
    # Partition mesh tokens into geodesic windows using ico-mapping.
    def __init__(self, high_order=6, window_order=2):
        super(GeodesicWindowPartition, self).__init__()
        
        # Load the mapping from high-res vertices to low-res 'window' faces
        # This is the "genealogy" strategy discussed
        mapping = ico.get_mapping(high_order, window_order)
        window_ids = torch.from_numpy(mapping).long()
        
        # register_buffer ensures the map moves to GPU with the model
        self.register_buffer('window_ids', window_ids)
        self.num_windows = len(torch.unique(window_ids))
    
    def forward(self, x):
        B, N, C = x.shape
        
        # Sort vertices by their window ID to group them spatially
        indices = torch.argsort(self.window_ids)
        x_grouped = x[:, indices, :]
        
        window_size = N // self.num_windows
        windows = x_grouped.view(B * self.num_windows, window_size, C)

        return windows, self.num_windows, indices

    def reverse(self, windows, indices, B, N):
        # Reconstruct the original mesh order after attention
        C = windows.shape[-1]
        x_grouped = windows.view(B, N, C)
        
        # Invert the sorting permutation to restore ico-6 order
        inv_indices = torch.argsort(indices)
        return x_grouped[:, inv_indices, :]
    

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
        self.partition = GeodesicWindowPartition(high_order, window_order)
        self.attention = WindowAttention(embed_dim, num_heads)
        self.proj = AttentionOutputProjection(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim, embed_dim * mlp_ratio)

    def forward(self, x, mesh):
        shortcut = x
        B, N,C = x.shape

        # Partition mesh tokens into geodesic windows.
        windows, num_windows, sort_idx = self.partition(x)

        # Apply Window Attention
        attn_windows = self.attention(self.norm1(windows))
        attn_windows = self.proj(attn_windows)

        # Re-assemble the mesh from windows back to original vertex order
        x_recovered = self.partition.reverse(attn_windows, sort_idx, B, N)

        x = shortcut + x_recovered
        x = x + self.ffn(self.norm2(x))

        return x, mesh
    
class GraphPooling(nn.Module):
    def __init__(self, mesh_info, current_order):
        super(GraphPooling, self).__init__()
        self.mesh_info = mesh_info
        self.current_order = current_order
    
    def forward(self, x,mesh = None):
        x = utils.pool(x,self.mesh_info)
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
    
    def forward(self,x):

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
    def __init__(self,in_channels,embed_dim,pool_config):
        super(Encoder,self).__init__()

        # MeshCNN Blocks - 3
        self.mesh_cnn = nn.ModuleList([
            MeshCNNBlock(in_channels,embed_dim),
            MeshCNNBlock(embed_dim,embed_dim),
            MeshCNNBlock(embed_dim,embed_dim)
            ])
        
        # Token embedding block
        self.embedding = MeshTokenEmbeddingBlock(embed_dim,embed_dim)

        # Window Based Graph Attention Blocks -4
        self.win_gat = nn.ModuleList([
            WindowBasedGraphAttentionBlock(embed_dim,num_heads=8)
            for _ in range(4)
        ])

        # Graph pooling layer
        self.pooling = GraphPooling(pool_config[0]['mesh_info'],current_order=6)

        # Transoformer Blocks -3
        self.transformer = nn.ModuleList([
            TransformerBlock(embed_dim,nums_heads=8)
            for _ in range(3)
        ])

    def forward(self,x,mesh):

            for block in self.mesh_cnn:
                x,mesh = block(x,mesh)

            x = self.embedding(x)

            for block in self.win_gat:
                x, mesh = block(x,mesh)

            x, mesh = self.pooling(x, mesh)

            for block in self.transformer:
                x, mesh = block(x,mesh)

            return x, mesh
