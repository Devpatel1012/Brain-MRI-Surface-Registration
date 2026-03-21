import torch
from torch_scatter import scatter_max
import numpy as np


#  a way to track the current torch device globally
global_config = {
    'device': None,
}


def get_device():
    """
    Return the global torch device
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_device(device):
    """
    Set the global torch device
    """
    global_config['device'] = device


def read_file_list(filename, prefix=None, suffix=None):
    """
    Reads a list of files from a line-seperated text file
    """
    with open(filename, 'r') as file:
        content = file.readlines()
    filelist = [x.strip() for x in content if x.strip()]
    if prefix is not None:
        filelist = [prefix + f for f in filelist]
    if suffix is not None:
        filelist = [f + suffix for f in filelist]
    return filelist


def cross(vector1, vector2, dim=-1):
    """
    Cross product of two Nx3 vector arrays
    """
    v1_x, v1_y, v1_z = torch.unbind(vector1, dim=dim)
    v2_x, v2_y, v2_z = torch.unbind(vector2, dim=dim)
    n_x = v1_y * v2_z - v1_z * v2_y
    n_y = v1_z * v2_x - v1_x * v2_z
    n_z = v1_x * v2_y - v1_y * v2_x
    return torch.stack([n_x, n_y, n_z], dim=dim)


def point_sample(coords, features, image_size, normed=False):
    """
    Sample image features from a set of coordinates
    """
    if not normed:
        half_size = (image_size - 1) / 2
        coords = (coords - half_size) / half_size
    coords = torch.reshape(coords, (1, coords.shape[-2], 1, 1, coords.shape[-1]))
    point_features = torch.nn.functional.grid_sample(features.unsqueeze(0).swapaxes(-1, -3), coords, align_corners=True, mode='bilinear')
    point_features = point_features.squeeze(0).squeeze(-1).squeeze(-1).swapaxes(-1, -2)
    return point_features


def face_normals(face_coords, clockwise=False, normalize=True):
    """
    Compute face normals
    """
    v0, v1, v2 = torch.unbind(face_coords, -2)
    normal_vector = cross(v1 - v0, v2 - v0, dim=-1)
    if not clockwise:
        normal_vector = -1.0 * normal_vector
    if normalize:
        normal_vector = torch.nn.functional.normalize(normal_vector, p=2, dim=-1)
    return normal_vector


def compute_normals(coords, face_indices):
    """
    Compute vertex normals as an averge of face normals
    """
    face_coords = coords[face_indices]
    mesh_face_normals = face_normals(face_coords, clockwise=False, normalize=False)

    unnorm_vertex_normals = torch.zeros(coords.shape, dtype=torch.float32, device=get_device())
    for i in range(3):
        unnorm_vertex_normals = unnorm_vertex_normals.scatter_add(-2, face_indices[..., i:i + 1].expand(-1, 3), mesh_face_normals)

    vector_norms = torch.sqrt(torch.sum(unnorm_vertex_normals ** 2, dim=-1, keepdims=True))
    return unnorm_vertex_normals / vector_norms


def pool(features, mesh_info):
    """
    Pooling of an icosphere order
    """
    pooled = gather_vertex_features(features,
        mesh_info['pooling_shape_a'],
        mesh_info['pooling_b'],
        mesh_info['pooling_a'],)
    return pooled


def unpool(features, mesh_info):
    """
    Unpooling of an icosphere order
    """
    unpooled = gather_vertex_features(features,
        mesh_info['pooling_shape_b'],
        mesh_info['pooling_a'],
        mesh_info['pooling_b'],)
    return unpooled


def gather_vertex_features(features, size, sources, targets):
    """
    Gather vertex features across any array with robust type handling
    """
    # 1. Ensure inputs are Tensors
    if not isinstance(sources, torch.Tensor):
        sources = torch.as_tensor(sources, device=get_device(), dtype=torch.long)
    if not isinstance(targets, torch.Tensor):
        targets = torch.as_tensor(targets, device=get_device(), dtype=torch.long)
        
    nb_features = features.shape[-1]
    features = features.to(get_device())
    gathered_features = features[sources]
    
    # 2. Define size_val here, before it is used
    if isinstance(size, (torch.Tensor, np.ndarray)):
        size_val = int(size.flatten()[0]) 
    elif hasattr(size, 'item'):
        size_val = int(size.item())
    elif isinstance(size, (list, tuple)):
        size_val = int(size[0])
    else:
        size_val = int(size)
        
    # 3. Now initialize the output buffer using size_val
    out = torch.zeros((size_val, int(nb_features)), dtype=torch.float32, device=get_device()) - 1000
    
    # 4. Perform scatter operation
    out, _ = scatter_max(gathered_features, targets, -2, out=out)
    
    return out