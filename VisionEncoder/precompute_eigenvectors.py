import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import sys
import os

sys.path.append('/content/drive/MyDrive/Brain MRI Surface Registration/VisionEncoder/')
from topofit.topofit import ico

def cotangent(a, b, c):
    ba, bc = a - b, c - b
    cos_angle = np.dot(ba, bc)
    sin_angle = np.linalg.norm(np.cross(ba, bc))
    return cos_angle / (sin_angle + 1e-8)

def compute_laplacian_eigenvectors(vertices, faces, k=32):
    n = vertices.shape[0]
    target_k = k
    k = min(k, n - 1) 
    I, J, W = [], [], []
    mass = np.zeros(n)
    for tri in faces:
        i, j, kf = tri
        vi, vj, vk = vertices[i], vertices[j], vertices[kf]
        cot_i, cot_j, cot_k = cotangent(vj, vi, vk), cotangent(vi, vj, vk), cotangent(vi, vk, vj)
        for (u, v, w) in [(i, j, cot_k), (j, kf, cot_i), (kf, i, cot_j)]:
            I.extend([u, v]); J.extend([v, u]); W.extend([w/2, w/2])
        area = np.linalg.norm(np.cross(vj - vi, vk - vi)) / 2
        mass[i] += area / 3; mass[j] += area / 3; mass[kf] += area / 3
    W = sp.coo_matrix((W, (I, J)), shape=(n, n))
    D = sp.diags(np.array(W.sum(axis=1)).flatten())
    L, M = D - W, sp.diags(mass)
    
    eigvals, eigvecs = eigsh(L.astype(float), k=k, M=M.astype(float), sigma=0, which='SM', tol=1e-3, maxiter=10000)
    if eigvecs.shape[1] < target_k:
        eigvecs = np.pad(eigvecs, ((0,0), (0, target_k - eigvecs.shape[1])), mode='constant')
    return eigvecs

master_path = '/content/drive/MyDrive/Brain MRI Surface Registration/VisionEncoder/laplacians.npz'
data = dict(np.load(master_path)) if os.path.exists(master_path) else {}

for order in range(8):
    key = f'ico-{order}-eigenvectors'
    if key in data: continue
    
    print(f"Computing Order {order}...")
    verts = ico.get_ico_data(f'ico-{order}-vertices')
    faces = ico.get_ico_data(f'ico-{order}-faces')
    data[key] = compute_laplacian_eigenvectors(verts, faces, k=32)
    np.savez(master_path, **data)
    print(f"Saved Order {order}")