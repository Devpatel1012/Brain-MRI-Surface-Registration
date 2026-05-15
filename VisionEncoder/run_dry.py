import sys
import os
import torch

sys.path.append(os.path.abspath(os.getcwd()))

from Encoder import Encoder
from topofit.topofit import ico, io, utils 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
utils.set_device(device)


mesh_topology = ico.load_topology(order=6)
pool_config = [{'mesh_info': mesh_topology}]

#  Load Mock Data
subj_dir = 'IXI_Mock_Subj'
data = io.load_subject_data(subj_dir, hemi='lh', ground_truth=True)
input_vertices = torch.from_numpy(data['input_vertices']).float().to(device).unsqueeze(0)

#  Initialize Encoder
model = Encoder(in_channels=3, embed_dim=64, pool_config=pool_config).to(device)

#  Forward Pass (Passing the topology object instead of None)
with torch.no_grad():
    output, mesh = model(input_vertices, mesh_topology)
    print("Encoder output shape:", output.shape)
