import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureWhitening(nn.Module):
    def __init__ (self, eps = 1e -5):
        super(FeatureWhitening,self).__init__()
        self.eps = eps
     
    def forward(self,x):
        mean = x.mean(dim = 1, keepdim = True)
        std = x.std(dim = 1,keepdim = True) + self.eps
        return (x-mean)/std

class SpectralProj(nn.Module):
    def forward(self,features, eigenvectors):
        return torch.einsum('nk,bnc-->bkc', eigenvectors , features)


class crossAttentionMatching(nn.Module):
    def __init__(self,dim,heads = 4):
        super(crossAttentionMatching,self).__init__()
        assert dim % heads  == 0
        self.heads = heads
        self.scale = (dim//heads) ** -0.5

        self.to_q = nn.Linear(dim,dim)
        self.to_k = nn.Linear(dim,dim)
        self.to_v = nn.Linear(dim,dim)

    def forward(self, src, tgt):
        Q = self.to_q(src)
        K = self.to_k(tgt)
        V = self.to_v(tgt)

        sim = torch.matmul(Q,k.transpose(-2,-1)) * self.scale
        attn = F.softmax(sim,dim =1)

        out  = torch.matmul(attn,V)
        return out, sim.mean(dim = 1)

class SinkornNormalization(nn.Module):
    def __init__(self, iteration =10):
        super(SinkornNormalization,self).__init__()
        self.iterations = iteration
    
    def forward(self, sim_matrix):
        sim_matrix = sim_matrix - sim_matrix.max(dim = -1, keepdim = True)[0].max(dim = -2, keepdim = True)[0]
        P = torch.exp(sim_matrix)

        for _ in range(self,self.iterations):
            P = P/(P.sum(dim = 2 , keepdim = True)+ 1e-8)
            P = P/(P.sum(dim = 1 , keepdim = True)+ 1e-8)
            return P
        

class SeptralMatchingPipeline(nn.Module):
    def __init__(self,features_dim = 64 ,spectral_dim = 32,laplacian_path='laplacians.npz'):
        super(SeptralMatchingPipeline,self).__init__()
        self.whiten = FeatureWhitening()
        self.feature_reduce = nn.Linear(features_dim,spectral_dim)
        self.spectral_proj = SpectralProj()
        self.cross_attention = crossAttentionMatching(spectral_dim,heads=4)
        self.sinkhorn = SinkornNormalization(iteration=10)

        if os.path.exists(laplacian_path):
            self.laplacian_data = np.load(laplacian_path)
        else:
            raise FileNotFoundError(f"Run precompute_laplacians.py first!")

    def get_eigenvectors(self, order, device='cuda'):
        """Helper to fetch vectors for a specific order"""
        vecs = self.laplacian_data[f'ico-{order}-eigenvectors']
        return torch.from_numpy(vecs).float().to(device)

    def forward(self, Fs, Ft, order_s, order_t):
        # Retrieve Phi_s and Phi_t based on order
        Phi_s = self.get_eigenvectors(order_s)
        Phi_t = self.get_eigenvectors(order_t)

        Fs,Ft = self.whiten(Fs),self.whiten(Ft)
        Fs,Ft = self.feature_reduce(Fs),self.feature_reduce(Ft)

        Fs_spec, Ft_spec = self.spectral_proj(Fs, Phi_s), self.spectral_proj(Ft, Phi_t)

        _, sim_matrix = self.cross_attention(Fs_spec, Ft_spec)
        C = self.sinkhorn(sim_matrix)

        P_vertex = torch.matmul(Phi_s, torch.matmul(C, Phi_t.transpose(-2, -1)))

        P_vertex = F.relu(P_vertex)
        P_vertex = P_vertex / (P_vertex.sum(dim=-1, keepdim=True) + 1e-8)

        return P_vertex, C