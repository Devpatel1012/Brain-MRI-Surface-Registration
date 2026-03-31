import sys
import os
import torch

# 1. Path Management (Setting root to Visionencoder/)
sys.path.append(os.path.abspath(os.getcwd()))

# 2. Imports (Adjusted for nested folder structure)
from Encoder import Encoder
from topofit.topofit import ico, io, utils # Note the nested import

# 3. Rest of your script...# 2. Setup Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
utils.set_device(device)

# 3. Setup Topology and Pool Config
# This is the 'map' the model uses to understand the surface
mesh_topology = ico.load_topology(order=6)
pool_config = [{'mesh_info': mesh_topology}]

# 4. Load Mock Data
subj_dir = 'IXI_Mock_Subj'
data = io.load_subject_data(subj_dir, hemi='lh', ground_truth=True)
input_vertices = torch.from_numpy(data['input_vertices']).float().to(device).unsqueeze(0)

# 5. Initialize Encoder
model = Encoder(in_channels=3, embed_dim=64, pool_config=pool_config).to(device)

# 6. Forward Pass (Passing the topology object instead of None)
with torch.no_grad():
    output, mesh = model(input_vertices, mesh_topology)
    print("Encoder output shape:", output.shape)