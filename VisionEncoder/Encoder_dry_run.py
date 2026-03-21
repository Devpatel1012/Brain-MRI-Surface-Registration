import sys
import os
import torch
import numpy as np
import surfa as sf
import gc
from torch.amp import autocast

# 1. Setup paths
project_path = '/content/drive/MyDrive/Brain MRI Surface Registration'
if project_path not in sys.path:
    sys.path.append(project_path)

from VisionEncoder.Encoder import Encoder
from VisionEncoder.topofit.topofit import ico

# 2. Clear previous memory
gc.collect()
torch.cuda.empty_cache()

# 3. Define Mock Mesh
class MockMesh:
    def __init__(self, order, vertices):
        self.vertices = vertices
        # Use smaller order for dry run
        raw_edges = ico.get_ico_data(f'ico-{order}-edges-mapping')
        num_edges = raw_edges.shape[0] // 4
        # Using as_tensor to prevent UserWarning
        self.gemm_edges = torch.as_tensor(raw_edges.reshape(num_edges, 4), dtype=torch.long)
        self.edges_count = self.gemm_edges.shape[0]

# 4. Load smaller mesh (Order 3)
surf_path = 'IXI_Mock_Subj/surf/lh.white.ico.surf'
surf = sf.load_mesh(surf_path)
# Use a subset for dry run to keep memory low
vertices = torch.from_numpy(surf.vertices).float()[:642]
mesh_obj = MockMesh(order=3, vertices=vertices)

B, N, C = 1, vertices.shape[0], 3
dummy_input = torch.randn(B, C, N).to('cuda')

# 5. Prepare REAL pooling configuration
# The GraphPooling layer expects these keys to exist inmapping-2-to-3-shape mesh_info
pool_info = {
    'pooling_shape_a': ico.get_ico_data('mapping-2-to-3-shape'),
    'pooling_shape_b': ico.get_ico_data('mapping-2-to-3-shape'),
    'pooling_a': ico.get_ico_data('mapping-2-to-3-indices'),
    'pooling_b': ico.get_ico_data('mapping-2-to-3-values')}

pool_config = [{'mesh_info': {}}]

# 6. Run Optimized Dry Run
print("Initializing Encoder...")
# Ensure num_heads=2 in Encoder.py for this test
encoder = Encoder(in_channels=C, embed_dim=64, pool_config=pool_config).to('cuda')
encoder.eval()

print(f"Running forward pass with N={N} vertices...")
try:
    with torch.no_grad():
        # Using updated torch.amp.autocast syntax
        with autocast(device_type='cuda', dtype=torch.float16):
            output, updated_mesh = encoder(dummy_input, [mesh_obj])

    print("Dry run successful!")
    print("Output feature shape:", output.shape)
except Exception as e:
    print("Dry run failed!")
    import traceback
    traceback.print_exc()
finally:
    gc.collect()
    torch.cuda.empty_cache()