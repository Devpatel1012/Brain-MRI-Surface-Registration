import torch
import torch.nn as nn

class BrainRegistrationLoss(nn.Module):
    def __init__(self, edges, w_chamfer=1.0, w_edge=0.1):
        super().__init__()
        self.w_chamfer = w_chamfer
        self.w_edge = w_edge
        
        # We only need the edges from the Order 6 template, NOT the faces
        self.register_buffer('edges', torch.tensor(edges, dtype=torch.long))

    def chamfer_distance(self, predicted_verts, target_verts):
        """
        Calculates the physical distance between two meshes, even if they have 
        different numbers of vertices!
        """
        # Calculate distance matrix: (Batch, 40962, 107418)
        # To avoid OOM errors on large distance matrices, we calculate it in chunks if necessary,
        # but for batch size 1, cdist usually handles it fine on 12GB+ VRAM.
        dist_matrix = torch.cdist(predicted_verts, target_verts)
        
        # Find the closest target vertex for each predicted vertex
        min_dist_pred_to_target, _ = torch.min(dist_matrix, dim=2)
        
        # Find the closest predicted vertex for each target vertex
        min_dist_target_to_pred, _ = torch.min(dist_matrix, dim=1)
        
        # Average the distances
        chamfer_loss = torch.mean(min_dist_pred_to_target) + torch.mean(min_dist_target_to_pred)
        return chamfer_loss

    def edge_length_regularization(self, predicted_verts):
        """
        Prevents the mesh from creating sharp spikes.
        Calculated ONLY on the predicted 40962 mesh.
        """
        B = predicted_verts.shape[0]
        loss = 0.0
        
        for b in range(B):
            verts = predicted_verts[b]
            v1 = verts[self.edges[:, 0]]
            v2 = verts[self.edges[:, 1]]
            edge_lengths = torch.norm(v1 - v2, dim=1)
            loss += torch.var(edge_lengths)
            
        return loss / B

    def forward(self, pred_verts, target_verts):
        
        # 1. Chamfer Distance (Point Cloud Alignment)
        c_loss = self.chamfer_distance(pred_verts, target_verts)
        
        # 2. Edge Strain (Smoothness)
        e_loss = self.edge_length_regularization(pred_verts)
        
        total_loss = (self.w_chamfer * c_loss) + (self.w_edge * e_loss)
        
        return total_loss, c_loss, e_loss