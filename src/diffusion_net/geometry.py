import scipy
import scipy.sparse.linalg as sla
# ^^^ we NEED to import scipy before torch, or it crashes :(
# (observed on Ubuntu 20.04 w/ torch 1.6.0 and scipy 1.5.2 installed via conda)

import os.path
import sys
import random
from multiprocessing import Pool

import numpy as np
import scipy.spatial
import torch
from torch.distributions.categorical import Categorical
import sklearn.neighbors
from itertools import combinations

import robust_laplacian
import potpourri3d as pp3d

import diffusion_net.utils as utils
from .utils import toNP

import diffusion_net_cuda as dnc

def norm(x, highdim=False):
    """
    Computes norm of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
    """
    return torch.norm(x, dim=len(x.shape) - 1)


def norm2(x, highdim=False):
    """
    Computes norm^2 of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
    """
    return dot(x, x)


def normalize(x, divide_eps=1e-6, highdim=False):
    """
    Computes norm^2 of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
    """
    if(len(x.shape) == 1):
        raise ValueError("called normalize() on single vector of dim " +
                         str(x.shape) + " are you sure?")
    if(not highdim and x.shape[-1] > 4):
        raise ValueError("called normalize() with large last dimension " +
                         str(x.shape) + " are you sure?")
    return x / (norm(x, highdim=highdim) + divide_eps).unsqueeze(-1)


def face_coords(verts, faces):
    coords = verts[faces]
    return coords


def cross(vec_A, vec_B):
    return torch.cross(vec_A, vec_B, dim=-1)


def dot(vec_A, vec_B):
    return torch.sum(vec_A * vec_B, dim=-1)


# Given (..., 3) vectors and normals, projects out any components of vecs
# which lies in the direction of normals. Normals are assumed to be unit.

def project_to_tangent(vecs, unit_normals):
    dots = dot(vecs, unit_normals)
    return vecs - unit_normals * dots.unsqueeze(-1)


def face_area(verts, faces):
    coords = face_coords(verts, faces)
    vec_A = coords[:, 1, :] - coords[:, 0, :]
    vec_B = coords[:, 2, :] - coords[:, 0, :]

    raw_normal = cross(vec_A, vec_B)
    return 0.5 * norm(raw_normal)

def face_normals(verts, faces, normalized=True):
    coords = face_coords(verts, faces)
    vec_A = coords[:, 1, :] - coords[:, 0, :]
    vec_B = coords[:, 2, :] - coords[:, 0, :]

    raw_normal = cross(vec_A, vec_B)

    if normalized:
        return normalize(raw_normal)

    return raw_normal

def neighborhood_normal(points):
    # points: (N, K, 3) array of neighborhood psoitions
    # points should be centered at origin
    # out: (N,3) array of normals
    # numpy in, numpy out
    (u, s, vh) = np.linalg.svd(points, full_matrices=False)
    normal = vh[:,2,:]
    return normal / np.linalg.norm(normal,axis=-1, keepdims=True)

def mesh_vertex_normals(verts, faces):
    # numpy in / out
    face_n = toNP(face_normals(torch.tensor(verts), torch.tensor(faces))) # ugly torch <---> numpy

    vertex_normals = np.zeros(verts.shape)
    for i in range(3):
        np.add.at(vertex_normals, faces[:,i], face_n)

    vertex_normals = vertex_normals / np.linalg.norm(vertex_normals,axis=-1,keepdims=True)

    return vertex_normals


def vertex_normals(verts, faces, n_neighbors_cloud=30):
    verts_np = toNP(verts)

    if faces.numel() == 0: # point cloud
    
        _, neigh_inds = find_knn(verts, verts, n_neighbors_cloud, omit_diagonal=True, method='cpu_kd')
        neigh_points = verts_np[neigh_inds,:]
        neigh_points = neigh_points - verts_np[:,np.newaxis,:]
        normals = neighborhood_normal(neigh_points)

    else: # mesh

        normals = mesh_vertex_normals(verts_np, toNP(faces))

        # if any are NaN, wiggle slightly and recompute
        bad_normals_mask = np.isnan(normals).any(axis=1, keepdims=True)
        if bad_normals_mask.any():
            bbox = np.amax(verts_np, axis=0) - np.amin(verts_np, axis=0)
            scale = np.linalg.norm(bbox) * 1e-4
            wiggle = (np.random.RandomState(seed=777).rand(*verts.shape)-0.5) * scale
            wiggle_verts = verts_np + bad_normals_mask * wiggle
            normals = mesh_vertex_normals(wiggle_verts, toNP(faces))

        # if still NaN assign random normals (probably means unreferenced verts in mesh)
        bad_normals_mask = np.isnan(normals).any(axis=1)
        if bad_normals_mask.any():
            normals[bad_normals_mask,:] = (np.random.RandomState(seed=777).rand(*verts.shape)-0.5)[bad_normals_mask,:]
            normals = normals / np.linalg.norm(normals, axis=-1)[:,np.newaxis]
            

    normals = torch.from_numpy(normals).to(device=verts.device, dtype=verts.dtype)
    normals = normals.to(torch.float32)
        
    if torch.any(torch.isnan(normals)): raise ValueError("NaN normals :(")

    return normals


def build_tangent_frames(verts, faces, vert_normals=None):

    V = verts.shape[0]
    dtype = verts.dtype
    device = verts.device

    # = find an orthogonal basis

    basis_cand1 = torch.tensor([1, 0, 0]).to(device=device, dtype=dtype).expand(V, -1)
    basis_cand2 = torch.tensor([0, 1, 0]).to(device=device, dtype=dtype).expand(V, -1)
    
    basisX = torch.where((torch.abs(dot(vert_normals, basis_cand1))
                          < 0.9).unsqueeze(-1), basis_cand1, basis_cand2)
    basisX = project_to_tangent(basisX, vert_normals)
    basisX = normalize(basisX)
    basisY = cross(vert_normals, basisX)
    frames = torch.stack((basisX, basisY, vert_normals), dim=-2)
    
    if torch.any(torch.isnan(frames)):
        raise ValueError("NaN coordinate frame! Must be very degenerate")

    return frames
        
def build_grad_point_cloud(verts, frames, n_neighbors_cloud=30):

    verts_np = toNP(verts)
    frames_np = toNP(frames)

    _, neigh_inds = find_knn(verts, verts, n_neighbors_cloud, omit_diagonal=True, method='cpu_kd')
    neigh_points = verts_np[neigh_inds,:]
    neigh_vecs = neigh_points - verts_np[:,np.newaxis,:]

    # TODO this could easily be way faster. For instance we could avoid the weird edges format and the corresponding pure-python loop via some numpy broadcasting of the same logic. The way it works right now is just to share code with the mesh version. But its low priority since its preprocessing code.

    edge_inds_from = np.repeat(np.arange(verts.shape[0]), n_neighbors_cloud)
    edges = np.stack((edge_inds_from, neigh_inds.flatten()))

    # Optionally to have this on GPU earlier
    edge_tangent_vecs = edge_tangent_vectors(verts, frames, edges)
    edge_tangent_vecs_cuda = edge_tangent_vecs.to('cuda:0')
    edges_tensor_cuda = torch.tensor(edges, dtype=torch.int32, device='cuda:0')
    verts_cuda = verts.to('cuda:0')

    return build_grad_cuda(verts_cuda, edges_tensor_cuda, edge_tangent_vecs_cuda,n_neighbors_cloud,verts.device)


def edge_tangent_vectors(verts, frames, edges):
    edge_vecs = verts[edges[1, :], :] - verts[edges[0, :], :]
    basisX = frames[edges[0, :], 0, :]
    basisY = frames[edges[0, :], 1, :]

    compX = dot(edge_vecs, basisX)
    compY = dot(edge_vecs, basisY)
    edge_tangent = torch.stack((compX, compY), dim=-1)

    return edge_tangent

# Curvature computateion. Inspired from: https://nl.mathworks.com/matlabcentral/fileexchange/32573-patch-curvature
def get_curvature(vertices,normals,nHoodRange=1):

    nVertices = len(vertices)

    # Compute rortation matrices
    Minv = torch.zeros((3,3,nVertices),device=vertices.device)
    k = torch.rand((nVertices,3),device=vertices.device)
    l = torch.zeros((3,),device=vertices.device)
    kl = torch.zeros((3,),device=vertices.device)

    # Determine nHood
    nHoodRangeSquare = nHoodRange * nHoodRange
    
    # Get Minv
    Minv = dnc.get_minv_matrix(normals)
    
    nHoodList = []
    for v in vertices:
        dists = vertices-v
        dists_length_square = (dists*dists).sum(axis=1)
        nHood = torch.where(dists_length_square<nHoodRangeSquare)
        nHoodList.append(nHood[0])

    vertices_np = vertices.cpu().numpy()

    #From hereon it is faster to use numpy (lots of lin. algebra)
    i1i2 = np.zeros((nVertices,4))
    curvature = np.zeros((nVertices,2))
    for i in range(nVertices):
        We = np.matmul(vertices_np[nHoodList[i].cpu()],Minv[:,:,i].cpu())
        if We.ndim == 1:
            We = np.expand_dims(We,0)
        f = We[:,0]
        x = We[:,1]
        y = We[:,2]
        FM = np.vstack((x**2, y**2, x*y, x, y, np.ones(x.shape))).T
        A = FM
        b = f
        
        num_vars = A.shape[1]
        rank = np.linalg.matrix_rank(A)
        if rank == num_vars:              
            sol = np.linalg.lstsq(A, b)[0]    # not under-determined
        else:
            for nz in combinations(range(num_vars), rank):    # the variables not set to zero
                try: 
                    nz = np.squeeze(nz)
                    sol = np.zeros((num_vars, 1))  
                    sub = np.asarray(np.linalg.solve(A[:, nz], b))
                    if sub.ndim == 1:
                        sub = np.expand_dims(sub,-1)
                    sol[nz, :] = sub
                except np.linalg.LinAlgError:     
                    pass                    # picked bad variables, can't solve
        
        Dxx = 2*sol[0]
        Dxy = sol[2]
        Dyy = 2*sol[1]
        
        tmp = np.sqrt((Dxx - Dyy)** 2 + 4*Dxy**2)
        v2x = 2*Dxy
        v2y = Dyy - Dxx + tmp
        mag = np.sqrt(v2x**2 + v2y**2)
        v2x = v2x / mag
        v2y = v2y / mag
        
        v1x = -v2y
        v1y = v2x
        
        mu1 = (0.5*(Dxx + Dyy + tmp))
        mu2 = (0.5*(Dxx + Dyy - tmp))
        
        if (abs(mu1)<abs(mu2)):
            Lambda1=mu1;
            Lambda2=mu2;
            I2=[v1x, v1y]
            I1=[v2x, v2y]
        else:
            Lambda1=mu2;
            Lambda2=mu1;
            I2=[v2x, v2y]
            I1=[v1x, v1y]
        
        if Lambda1 < Lambda2:
            curvature[i,0] = Lambda1
            curvature[i,1] = Lambda2
            i1i2[i,0] = I1[0]
            i1i2[i,1] = I1[1]
            i1i2[i,2] = I2[0]
            i1i2[i,3] = I2[1]
        else:
            curvature[i,0] = Lambda2
            curvature[i,1] = Lambda1
            i1i2[i,0] = I2[0]
            i1i2[i,1] = I2[1]
            i1i2[i,2] = I1[0]
            i1i2[i,3] = I1[1]

    cMean = (curvature[:,0]+curvature[:,1])/2
    cGauss = (curvature[:,0]*curvature[:,1])

    curvature = torch.from_numpy(curvature).to(torch.float)
    cMean = torch.from_numpy(cMean).to(torch.float)
    cGauss = torch.from_numpy(cGauss).to(torch.float)
    i1i2 = torch.from_numpy(i1i2).to(torch.float)

    M = torch.transpose(Minv,0,1)

    d1l = torch.hstack((torch.zeros((nVertices,1)),i1i2[:,:2])).to(Minv.device)
    d2l = torch.hstack((torch.zeros((nVertices,1)),i1i2[:,2:])).to(Minv.device)

    for i, (d1, d2) in enumerate(zip(d1l,d2l)):
        d1l[i] =  torch.matmul(d1,M[:,:,i])
        d2l[i] =  torch.matmul(d2,M[:,:,i])

    return curvature, cMean, cGauss, i1i2, d1l, d2l

def build_grad_cuda(verts_cuda, edges_tensor_cuda, edge_tangent_vecs, n_neighbors_cloud=12, return_device='cpu'):
    # Grad_data currently gives additional debug information. Will be removed later.
    grad_data = dnc.build_grad(verts_cuda, edges_tensor_cuda, edge_tangent_vecs,n_neighbors_cloud)

    # only supporting full neighbourhoods now (all vertices must have n_neighbors_cloud neighbours)
    #assert(torch.max(grad_data[4]) == torch.min(grad_data[4]))

    sel_rows = torch.where(grad_data[0] >= 0)

    if grad_data[0].device != return_device:
        rows = grad_data[0][sel_rows].to(return_device).type(torch.LongTensor)
        cols = grad_data[1][sel_rows].to(return_device).type(torch.LongTensor)
        data_gradX = grad_data[2][sel_rows].to(return_device)
        data_gradY = grad_data[3][sel_rows].to(return_device)
    else:
        rows = grad_data[0][sel_rows].type(torch.LongTensor)
        cols = grad_data[1][sel_rows].type(torch.LongTensor)
        data_gradX = grad_data[2][sel_rows]
        data_gradY = grad_data[3][sel_rows]

    indices = torch.vstack((rows,cols))
    shape = torch.Size([len(verts_cuda),len(verts_cuda)])

    if indices.get_device() != data_gradX.get_device():
        indices = indices.to(data_gradX.get_device())

    gradX = torch.sparse.FloatTensor(indices, data_gradX, torch.Size(shape)).coalesce()
    gradY = torch.sparse.FloatTensor(indices, data_gradY, torch.Size(shape)).coalesce()

    return gradX, gradY

def compute_operators(verts, faces, k_eig, normals=None, curvature_range=0.0, eigsh_tol=1e-24):
    """
    Builds spectral operators for a mesh/point cloud. Constructs mass matrix, eigenvalues/vectors for Laplacian, and gradient matrix.

    See get_operators() for a similar routine that wraps this one with a layer of caching.

    Torch in / torch out.

    Arguments:
      - vertices: (V,3) vertex positions
      - faces: (F,3) list of triangular faces. If empty, assumed to be a point cloud.
      - k_eig: number of eigenvectors to use
      - curvature_range: (float) the range of which the curvature computation must occur (in vertex unit space). 0 or None = Disabled
      - eigsh_tol: (float) the tolerance used in sla.eigsh (def.)

    Returns:
      - frames: (V,3,3) X/Y/Z coordinate frame at each vertex. Z coordinate is normal (e.g. [:,2,:] for normals)
      - massvec: (V) real diagonal of lumped mass matrix
      - L: (VxV) real sparse matrix of (weak) Laplacian
      - evals: (k) list of eigenvalues of the Laplacian
      - evecs: (V,k) list of eigenvectors of the Laplacian 
      - gradX: (VxV) sparse matrix which gives X-component of gradient in the local basis at the vertex
      - gradY: same as gradX but for Y-component of gradient

    PyTorch doesn't seem to like complex sparse matrices, so we store the "real" and "imaginary" (aka X and Y) gradient matrices separately, rather than as one complex sparse matrix.

    Note: for a generalized eigenvalue problem, the mass matrix matters! The eigenvectors are only othrthonormal with respect to the mass matrix, like v^H M v, so the mass (given as the diagonal vector massvec) needs to be used in projections, etc.
    """

    device = verts.device
    dtype = verts.dtype
    V = verts.shape[0]
    is_cloud = faces.numel() == 0

    eps = 1e-8

    verts_np = toNP(verts).astype(np.float64)
    faces_np = toNP(faces)

    # Compute/get before build_tangent_frames in case we need curvature computaiton
    if normals == None:
        normals = vertex_normals(verts, faces)  # (V,3)

    if curvature_range > 0:
        curvature, cMean, cGauss, i1i2, d1l, d2l  = get_curvature(verts,normals,nHoodRange=curvature_range)

    frames = build_tangent_frames(verts, faces, vert_normals=normals)

    # Build the scalar Laplacian
    if is_cloud:
        L, M = robust_laplacian.point_cloud_laplacian(verts_np)
        massvec_np = M.diagonal()
    else:
        # L, M = robust_laplacian.mesh_laplacian(verts_np, faces_np)
        # massvec_np = M.diagonal()
        L = pp3d.cotan_laplacian(verts_np, faces_np, denom_eps=1e-10)
        massvec_np = pp3d.vertex_areas(verts_np, faces_np)
        massvec_np += eps * np.mean(massvec_np)
    
    if(np.isnan(L.data).any()):
        raise RuntimeError("NaN Laplace matrix")
    if(np.isnan(massvec_np).any()):
        raise RuntimeError("NaN mass matrix")

    # Read off neighbors & rotations from the Laplacian
    L_coo = L.tocoo()
    inds_row = L_coo.row
    inds_col = L_coo.col

    # === Compute the eigenbasis
    if k_eig > 0:

        # Prepare matrices
        L_eigsh = (L + scipy.sparse.identity(L.shape[0])*eps).tocsc()
        massvec_eigsh = massvec_np
        Mmat = scipy.sparse.diags(massvec_eigsh)
        eigs_sigma = eps

        failcount = 0
        while True:
            try:
                # We would be happy here to lower tol or maxiter since we don't need these to be super precise, but for some reason those parameters seem to have no effect
                evals_np, evecs_np = sla.eigsh(L_eigsh, k=k_eig, M=Mmat, sigma=eigs_sigma,tol=eigsh_tol)
            
                # Clip off any eigenvalues that end up slightly negative due to numerical weirdness
                evals_np = np.clip(evals_np, a_min=0., a_max=float('inf'))

                break
            except Exception as e:
                print(e)
                if(failcount > 3):
                    raise ValueError("failed to compute eigendecomp")
                failcount += 1
                print("--- decomp failed; adding eps ===> count: " + str(failcount))
                L_eigsh = L_eigsh + scipy.sparse.identity(L.shape[0]) * (eps * 10**failcount)


    else: #k_eig == 0
        evals_np = np.zeros((0))
        evecs_np = np.zeros((verts.shape[0],0))


    # == Build gradient matrices

    # For meshes, we use the same edges as were used to build the Laplacian. For point clouds, use a whole local neighborhood
    if is_cloud:
        gradX, gradY = build_grad_point_cloud(verts, frames)
    else: # === WARNING DIFFUSION_NET_CUDA UNTESTED START ===
        edges = torch.tensor(np.stack((inds_row, inds_col), axis=0), device=device, dtype=torch.long)
        edge_vecs = edge_tangent_vectors(verts, frames, edges)

        # Optionally to have this on GPU earlier
        #edge_tangent_vecs = edge_tangent_vectors(verts, frames, edges)
        edge_tangent_vecs_cuda = edge_vecs.to('cuda:0')
        edges = edges.to('cuda:0').to(torch.int32) #torch.tensor(edges, dtype=torch.int32, device='cuda:0')
        verts_cuda = verts.to('cuda:0')
        
        gradX, gradY = build_grad_cuda(verts_cuda, edges, edge_tangent_vecs_cuda,return_device=verts.device)
          # === WARNING DIFFUSION_NET_CUDA UNTESTED END ===


    # === Convert back to torch
    massvec = torch.from_numpy(massvec_np).to(device=device, dtype=dtype)
    L = utils.sparse_np_to_torch(L).to(device=device, dtype=dtype)
    evals = torch.from_numpy(evals_np).to(device=device, dtype=dtype)
    evecs = torch.from_numpy(evecs_np).to(device=device, dtype=dtype)

    if curvature_range > 0:
        return frames, massvec, L, evals, evecs, gradX, gradY, curvature, cMean, cGauss, i1i2, d1l, d2l
    else:
        return frames, massvec, L, evals, evecs, gradX, gradY

# Returned the legacy (CPU) version of build_grad
def build_grad(verts, edges, edge_tangent_vectors):
    """
    Build a (V, V) complex sparse matrix grad operator. Given real inputs at vertices, produces a complex (vector value) at vertices giving the gradient. All values pointwise.
    - edges: (2, E)
    """
    
    edges_np = toNP(edges.cpu())
    edge_tangent_vectors = toNP(edge_tangent_vectors.cpu())

    # TODO find a way to do this in pure numpy?

    # Build outgoing neighbor lists
    N = verts.shape[0]
    vert_edge_outgoing = [[] for i in range(N)]
    for iE in range(edges_np.shape[1]):
        tail_ind = edges_np[0, iE]
        tip_ind = edges_np[1, iE]
        if tip_ind != tail_ind:
            vert_edge_outgoing[tail_ind].append(iE)

    # Build local inversion matrix for each vertex
    row_inds = []
    col_inds = []
    data_vals = []
    eps_reg = 1e-5
    for iV in range(N):
        n_neigh = len(vert_edge_outgoing[iV])

        lhs_mat = np.zeros((n_neigh, 2))
        rhs_mat = np.zeros((n_neigh, n_neigh + 1))
        ind_lookup = [iV]
        for i_neigh in range(n_neigh):
            iE = vert_edge_outgoing[iV][i_neigh]
            jV = edges_np[1, iE]
            ind_lookup.append(jV)
    
            edge_vec = edge_tangent_vectors[iE][:]
            w_e = 1.

            lhs_mat[i_neigh][:] = w_e * edge_vec
            rhs_mat[i_neigh][0] = w_e * (-1)
            rhs_mat[i_neigh][i_neigh + 1] = w_e * 1

        lhs_T = lhs_mat.T
        lhs_inv = np.linalg.inv(lhs_T @ lhs_mat + eps_reg * np.identity(2)) @ lhs_T

        sol_mat = lhs_inv @ rhs_mat
        sol_coefs = (sol_mat[0, :] + 1j * sol_mat[1, :]).T

        for i_neigh in range(n_neigh + 1):
            i_glob = ind_lookup[i_neigh]

            row_inds.append(iV)
            col_inds.append(i_glob)
            data_vals.append(sol_coefs[i_neigh])

    # build the sparse matrix
    row_inds = np.array(row_inds)
    col_inds = np.array(col_inds)
    data_vals = np.array(data_vals)
    mat = scipy.sparse.coo_matrix(
        (data_vals, (row_inds, col_inds)), shape=(
            N, N)).tocsc()

    return mat

def get_all_operators(verts_list, faces_list, k_eig, op_cache_dir=None, normals=None,curvature_range=0.0):
    N = len(verts_list)
            
    frames = [None] * N
    massvec = [None] * N
    L = [None] * N
    evals = [None] * N
    evecs = [None] * N
    gradX = [None] * N
    gradY = [None] * N
    if curvature_range > 0:
        curvature = [None] * N
        cMean = [None] * N
        cGauss = [None] * N
        i1i2 = [None] * N
        d1l = [None] * N
        d2l = [None] * N

    inds = [i for i in range(N)]
    # process in random order
    # random.shuffle(inds)
   
    for num, i in enumerate(inds):
        print("get_all_operators() processing {} / {} {:.3f}%".format(num, N, num / N * 100))
        if normals is None:
            outputs = get_operators(verts_list[i], faces_list[i], k_eig, op_cache_dir,curvature_range=curvature_range)
        else:
            outputs = get_operators(verts_list[i], faces_list[i], k_eig, op_cache_dir, normals=normals[i],curvature_range=curvature_range)
        frames[i] = outputs[0]
        massvec[i] = outputs[1]
        L[i] = outputs[2]
        evals[i] = outputs[3]
        evecs[i] = outputs[4]
        gradX[i] = outputs[5]
        gradY[i] = outputs[6]

        if curvature_range > 0:
            curvature[i] = outputs[7]
            cMean[i] = outputs[8]
            cGauss[i] = outputs[9]
            i1i2[i] = outputs[10]
            d1l[i] = outputs[11]
            d2l[i] = outputs[12]

    if curvature_range > 0:
        return frames, massvec, L, evals, evecs, gradX, gradY, curvature, cMean, cGauss, i1i2, d1l, d2l
    else:    
        return frames, massvec, L, evals, evecs, gradX, gradY

def get_operators(verts, faces, k_eig=128, op_cache_dir=None, normals=None, overwrite_cache=False,curvature_range=0.0):
    """
    See documentation for compute_operators(). This essentailly just wraps a call to compute_operators, using a cache if possible.
    All arrays are always computed using double precision for stability, then truncated to single precision floats to store on disk, and finally returned as a tensor with dtype/device matching the `verts` input.
    """

    device = verts.device
    dtype = verts.dtype
    verts_np = toNP(verts)
    faces_np = toNP(faces)
    is_cloud = faces.numel() == 0

    if(np.isnan(verts_np).any()):
        raise RuntimeError("tried to construct operators from NaN verts")

    # Check the cache directory
    # Note 1: Collisions here are exceptionally unlikely, so we could probably just use the hash...
    #         but for good measure we check values nonetheless.
    # Note 2: There is a small possibility for race conditions to lead to bucket gaps or duplicate
    #         entries in this cache. The good news is that that is totally fine, and at most slightly
    #         slows performance with rare extra cache misses.
    found = False
    if op_cache_dir is not None:
        utils.ensure_dir_exists(op_cache_dir)
        hash_key_str = str(utils.hash_arrays((verts_np, faces_np)))
        # print("Building operators for input with hash: " + hash_key_str)

        # Search through buckets with matching hashes.  When the loop exits, this
        # is the bucket index of the file we should write to.
        i_cache_search = 0
        while True:

            # Form the name of the file to check
            search_path = os.path.join(
                op_cache_dir,
                hash_key_str + "_" + str(i_cache_search) + ".npz")
            
            try:
                # print('loading path: ' + str(search_path))
                npzfile = np.load(search_path, allow_pickle=True)
                cache_verts = npzfile["verts"]
                cache_faces = npzfile["faces"]
                cache_k_eig = npzfile["k_eig"].item()

                # If the cache doesn't match, keep looking
                if (not np.array_equal(verts, cache_verts)) or (not np.array_equal(faces, cache_faces)):
                    i_cache_search += 1
                    print("hash collision! searching next.")
                    continue

                # print("  cache hit!")

                # If we're overwriting, or there aren't enough eigenvalues, just delete it; we'll create a new
                # entry below more eigenvalues
                if overwrite_cache: 
                    print("  overwriting cache by request")
                    os.remove(search_path)
                    break
                
                if cache_k_eig < k_eig:
                    print("  overwriting cache --- not enough eigenvalues")
                    os.remove(search_path)
                    break
                
                if "L_data" not in npzfile:
                    print("  overwriting cache --- entries are absent")
                    os.remove(search_path)
                    break

                if (curvature_range > 0) and ("curvature_range" not in npzfile):
                    print("  overwriting cache --- curvature_range are absent")
                    os.remove(search_path)
                    break

                if npzfile["curvature_range"] != curvature_range:
                    print("  overwriting cache --- curvature_range mismatch")
                    os.remove(search_path)
                    break

                def read_sp_mat(prefix):
                    data = npzfile[prefix + "_data"]
                    indices = npzfile[prefix + "_indices"]
                    indptr = npzfile[prefix + "_indptr"]
                    shape = npzfile[prefix + "_shape"]
                    mat = scipy.sparse.csc_matrix((data, indices, indptr), shape=shape)
                    return mat

                # This entry matches! Return it.
                frames = npzfile["frames"]
                mass = npzfile["mass"]
                L = read_sp_mat("L")
                evals = npzfile["evals"][:k_eig]
                evecs = npzfile["evecs"][:,:k_eig]
                gradX = read_sp_mat("gradX")
                gradY = read_sp_mat("gradY")

                if curvature_range > 0:
                    curvature = npzfile["curvature"]
                    cMean = npzfile["cMean"]
                    cGauss = npzfile["cGauss"]
                    i1i2 = npzfile["i1i2"]

                    curvature = torch.from_numpy(curvature).to(device=device, dtype=dtype)
                    cMean = torch.from_numpy(cMean).to(device=device, dtype=dtype)
                    cGauss = torch.from_numpy(cGauss).to(device=device, dtype=dtype)
                    i1i2 = torch.from_numpy(i1i2).to(device=device, dtype=dtype)

                frames = torch.from_numpy(frames).to(device=device, dtype=dtype)
                mass = torch.from_numpy(mass).to(device=device, dtype=dtype)
                L = utils.sparse_np_to_torch(L).to(device=device, dtype=dtype)
                evals = torch.from_numpy(evals).to(device=device, dtype=dtype)
                evecs = torch.from_numpy(evecs).to(device=device, dtype=dtype)
                gradX = utils.sparse_np_to_torch(gradX).to(device=device, dtype=dtype)
                gradY = utils.sparse_np_to_torch(gradY).to(device=device, dtype=dtype)
                
                found = True
                
                break

            except FileNotFoundError:
                print("  cache miss -- constructing operators")
                break
            
            except Exception as E:
                print("unexpected error loading file: " + str(E))
                print("-- constructing operators")
                break

    if not found:

        # No matching entry found; recompute.
        if curvature_range > 0:
            frames, mass, L, evals, evecs, gradX, gradY, curvature, cMean, cGauss, i1i2, d1l, d2l = compute_operators(verts, faces, k_eig, normals=normals,curvature_range=curvature_range)
        else:
            frames, mass, L, evals, evecs, gradX, gradY = compute_operators(verts, faces, k_eig, normals=normals)

        dtype_np = np.float32

        # Store it in the cache
        if op_cache_dir is not None:

            L_np = utils.sparse_torch_to_np(L).astype(dtype_np)
            gradX_np = utils.sparse_torch_to_np(gradX).astype(dtype_np)
            gradY_np = utils.sparse_torch_to_np(gradY).astype(dtype_np)

            if curvature_range > 0:
                np.savez(search_path,
                        verts=verts_np.astype(dtype_np),
                        frames=toNP(frames).astype(dtype_np),
                        faces=faces_np,
                        k_eig=k_eig,
                        mass=toNP(mass).astype(dtype_np),
                        curvature_range=curvature_range,
                        curvature=toNP(curvature).astype(dtype_np),
                        cMean=toNP(cMean).astype(dtype_np),
                        cGauss=toNP(cGauss).astype(dtype_np),
                        i1i2=toNP(i1i2).astype(dtype_np),
                        L_data = L_np.data.astype(dtype_np),
                        L_indices = L_np.indices,
                        L_indptr = L_np.indptr,
                        L_shape = L_np.shape,
                        evals=toNP(evals).astype(dtype_np),
                        evecs=toNP(evecs).astype(dtype_np),
                        gradX_data = gradX_np.data.astype(dtype_np),
                        gradX_indices = gradX_np.indices,
                        gradX_indptr = gradX_np.indptr,
                        gradX_shape = gradX_np.shape,
                        gradY_data = gradY_np.data.astype(dtype_np),
                        gradY_indices = gradY_np.indices,
                        gradY_indptr = gradY_np.indptr,
                        gradY_shape = gradY_np.shape,
                        )
            else:
                np.savez(search_path,
                        verts=verts_np.astype(dtype_np),
                        frames=toNP(frames).astype(dtype_np),
                        faces=faces_np,
                        k_eig=k_eig,
                        mass=toNP(mass).astype(dtype_np),
                        L_data = L_np.data.astype(dtype_np),
                        L_indices = L_np.indices,
                        L_indptr = L_np.indptr,
                        L_shape = L_np.shape,
                        evals=toNP(evals).astype(dtype_np),
                        evecs=toNP(evecs).astype(dtype_np),
                        gradX_data = gradX_np.data.astype(dtype_np),
                        gradX_indices = gradX_np.indices,
                        gradX_indptr = gradX_np.indptr,
                        gradX_shape = gradX_np.shape,
                        gradY_data = gradY_np.data.astype(dtype_np),
                        gradY_indices = gradY_np.indices,
                        gradY_indptr = gradY_np.indptr,
                        gradY_shape = gradY_np.shape,
                        )

    if curvature_range > 0:
        return frames, mass, L, evals, evecs, gradX, gradY, curvature, cMean, cGauss, i1i2, d1l, d2l
    else:
        return frames, mass, L, evals, evecs, gradX, gradY

def to_basis(values, basis, massvec):
    """
    Transform data in to an orthonormal basis (where orthonormal is wrt to massvec)
    Inputs:
      - values: (B,V,D)
      - basis: (B,V,K)
      - massvec: (B,V)
    Outputs:
      - (B,K,D) transformed values
    """
    basisT = basis.transpose(-2, -1)
    return torch.matmul(basisT, values * massvec.unsqueeze(-1))


def from_basis(values, basis):
    """
    Transform data out of an orthonormal basis
    Inputs:
      - values: (K,D)
      - basis: (V,K)
    Outputs:
      - (V,D) reconstructed values
    """
    if values.is_complex() or basis.is_complex():
        return utils.cmatmul(utils.ensure_complex(basis), utils.ensure_complex(values))
    else:
        return torch.matmul(basis, values)

def compute_hks(evals, evecs, scales):
    """
    Inputs:
      - evals: (K) eigenvalues
      - evecs: (V,K) values
      - scales: (S) times
    Outputs:
      - (V,S) hks values
    """

    # expand batch
    if len(evals.shape) == 1:
        expand_batch = True
        evals = evals.unsqueeze(0)
        evecs = evecs.unsqueeze(0)
        scales = scales.unsqueeze(0)
    else:
        expand_batch = False

    # TODO could be a matmul
    power_coefs = torch.exp(-evals.unsqueeze(1) * scales.unsqueeze(-1)).unsqueeze(1) # (B,1,S,K)
    terms = power_coefs * (evecs * evecs).unsqueeze(2)  # (B,V,S,K)

    out = torch.sum(terms, dim=-1) # (B,V,S)

    if expand_batch:
        return out.squeeze(0)
    else:
        return out

def compute_hks_autoscale(evals, evecs, count):
    # these scales roughly approximate those suggested in the hks paper
    scales = torch.logspace(-2, 0., steps=count, device=evals.device, dtype=evals.dtype)
    return compute_hks(evals, evecs, scales)

def normalize_positions(pos, faces=None, method='mean', scale_method='max_rad'):
    # center and unit-scale positions

    if method == 'mean':
        # center using the average point position
        pos = (pos - torch.mean(pos, dim=-2, keepdim=True))
    elif method == 'bbox': 
        # center via the middle of the axis-aligned bounding box
        bbox_min = torch.min(pos, dim=-2).values
        bbox_max = torch.max(pos, dim=-2).values
        center = (bbox_max + bbox_min) / 2.
        pos -= center.unsqueeze(-2)
    else:
        raise ValueError("unrecognized method")

    if scale_method == 'max_rad':
        scale = torch.max(norm(pos), dim=-1, keepdim=True).values.unsqueeze(-1)
        pos = pos / scale
    elif scale_method == 'area': 
        if faces is None:
            raise ValueError("must pass faces for area normalization")
        coords = pos[faces]
        vec_A = coords[:, 1, :] - coords[:, 0, :]
        vec_B = coords[:, 2, :] - coords[:, 0, :]
        face_areas = torch.norm(torch.cross(vec_A, vec_B, dim=-1), dim=1) * 0.5
        total_area = torch.sum(face_areas)
        scale = (1. / torch.sqrt(total_area))
        pos = pos * scale
    else:
        raise ValueError("unrecognized scale method")
    return pos

# Finds the k nearest neighbors of source on target.
# Return is two tensors (distances, indices). Returned points will be sorted in increasing order of distance.
def find_knn(points_source, points_target, k, largest=False, omit_diagonal=False, method='brute'):

    if omit_diagonal and points_source.shape[0] != points_target.shape[0]:
        raise ValueError("omit_diagonal can only be used when source and target are same shape")

    if method != 'cpu_kd' and points_source.shape[0] * points_target.shape[0] > 1e8:
        method = 'cpu_kd'
        print("switching to cpu_kd knn")

    if method == 'brute':

        # Expand so both are NxMx3 tensor
        points_source_expand = points_source.unsqueeze(1)
        points_source_expand = points_source_expand.expand(-1, points_target.shape[0], -1)
        points_target_expand = points_target.unsqueeze(0)
        points_target_expand = points_target_expand.expand(points_source.shape[0], -1, -1)

        diff_mat = points_source_expand - points_target_expand
        dist_mat = norm(diff_mat)

        if omit_diagonal:
            torch.diagonal(dist_mat)[:] = float('inf')

        result = torch.topk(dist_mat, k=k, largest=largest, sorted=True)
        return result
    
    elif method == 'cpu_kd':

        if largest:
            raise ValueError("can't do largest with cpu_kd")

        points_source_np = toNP(points_source)
        points_target_np = toNP(points_target)

        # Build the tree
        kd_tree = sklearn.neighbors.KDTree(points_target_np)

        k_search = k+1 if omit_diagonal else k 
        _, neighbors = kd_tree.query(points_source_np, k=k_search)
        
        if omit_diagonal: 
            # Mask out self element
            mask = neighbors != np.arange(neighbors.shape[0])[:, np.newaxis]

            # make sure we mask out exactly one element in each row, in rare case of many duplicate points
            mask[np.sum(mask, axis=1) == mask.shape[1], -1] = False

            neighbors = neighbors[mask].reshape((neighbors.shape[0], neighbors.shape[1]-1))

        inds = torch.tensor(neighbors, device=points_source.device, dtype=torch.int64)
        dists = norm(points_source.unsqueeze(1).expand(-1, k, -1) - points_target[inds])

        return dists, inds
    
    else:
        raise ValueError("unrecognized method")


def farthest_point_sampling(points, n_sample):
    # Torch in, torch out. Returns a |V| mask with n_sample elements set to true.

    N = points.shape[0]
    if(n_sample > N): raise ValueError("not enough points to sample")

    chosen_mask = torch.zeros(N, dtype=torch.bool, device=points.device)
    min_dists = torch.ones(N, dtype=points.dtype, device=points.device) * float('inf')

    # pick the centermost first point
    points = normalize_positions(points)
    i = torch.min(norm2(points), dim=0).indices
    chosen_mask[i] = True

    for _ in range(n_sample-1):
        
        # update distance
        dists = norm2(points[i,:].unsqueeze(0) - points)
        min_dists = torch.minimum(dists, min_dists)

        # take the farthest
        i = torch.max(min_dists,dim=0).indices.item()
        chosen_mask[i] = True

    return chosen_mask


def geodesic_label_errors(target_verts, target_faces, pred_labels, gt_labels, normalization='diameter', geodesic_cache_dir=None):
    """
    Return a vector of distances between predicted and ground-truth lables (normalized by geodesic diameter or area)

    This method is SLOW when it needs to recompute geodesic distances.
    """

    # move all to numpy cpu
    target_verts = toNP(target_verts) 
    target_faces = toNP(target_faces) 

    pred_labels = toNP(pred_labels) 
    gt_labels = toNP(gt_labels) 

    dists = get_all_pairs_geodesic_distance(target_verts, target_faces, geodesic_cache_dir) 

    result_dists = dists[pred_labels, gt_labels]

    if normalization == 'diameter':
        geodesic_diameter = np.max(dists)
        normalized_result_dists = result_dists / geodesic_diameter
    elif normalization == 'area':
        total_area = torch.sum(face_area(torch.tensor(target_verts), torch.tensor(target_faces)))
        normalized_result_dists = result_dists / torch.sqrt(total_area)
    else:
        raise ValueError('unrecognized normalization')

    return normalized_result_dists

# This function and the helper class below are to support parallel computation of all-pairs geodesic distance
def all_pairs_geodesic_worker(verts, faces, i):
    import igl

    N = verts.shape[0]

    # TODO: this re-does a ton of work, since it is called independently each time. Some custom C++ code could surely make it faster.
    sources = np.array([i])[:,np.newaxis]
    targets = np.arange(N)[:,np.newaxis]
    dist_vec = igl.exact_geodesic(verts, faces, sources, targets)
    
    return dist_vec
        
class AllPairsGeodesicEngine(object):
    def __init__(self, verts, faces):
        self.verts = verts 
        self.faces = faces 
    def __call__(self, i):
        return all_pairs_geodesic_worker(self.verts, self.faces, i)


def get_all_pairs_geodesic_distance(verts_np, faces_np, geodesic_cache_dir=None):
    """
    Return a gigantic VxV dense matrix containing the all-pairs geodesic distance matrix. Internally caches, recomputing only if necessary.

    (numpy in, numpy out)
    """

    # need libigl for geodesic call
    try:
        import igl
    except ImportError as e:
        raise ImportError("Must have python libigl installed for all-pairs geodesics. `conda install -c conda-forge igl`")

    # Check the cache
    found = False 
    if geodesic_cache_dir is not None:
        utils.ensure_dir_exists(geodesic_cache_dir)
        hash_key_str = str(utils.hash_arrays((verts_np, faces_np)))
        # print("Building operators for input with hash: " + hash_key_str)

        # Search through buckets with matching hashes.  When the loop exits, this
        # is the bucket index of the file we should write to.
        i_cache_search = 0
        while True:

            # Form the name of the file to check
            search_path = os.path.join(
                geodesic_cache_dir,
                hash_key_str + "_" + str(i_cache_search) + ".npz")

            try:
                npzfile = np.load(search_path, allow_pickle=True)
                cache_verts = npzfile["verts"]
                cache_faces = npzfile["faces"]

                # If the cache doesn't match, keep looking
                if (not np.array_equal(verts_np, cache_verts)) or (not np.array_equal(faces_np, cache_faces)):
                    i_cache_search += 1
                    continue

                # This entry matches! Return it.
                found = True
                result_dists = npzfile["dist"]
                break

            except FileNotFoundError:
                break

    if not found:
                
        print("Computing all-pairs geodesic distance (warning: SLOW!)")

        # Not found, compute from scratch
        # warning: slowwwwwww

        N = verts_np.shape[0]

        try:
            pool = Pool(None) # on 8 processors
            engine = AllPairsGeodesicEngine(verts_np, faces_np)
            outputs = pool.map(engine, range(N))
        finally: # To make sure processes are closed in the end, even if errors happen
            pool.close()
            pool.join()

        result_dists = np.array(outputs)

        # replace any failed values with nan
        result_dists = np.nan_to_num(result_dists, nan=np.nan, posinf=np.nan, neginf=np.nan)

        # we expect that this should be a symmetric matrix, but it might not be. Take the min of the symmetric values to make it symmetric
        result_dists = np.fmin(result_dists, np.transpose(result_dists))

        # on rare occaisions MMP fails, yielding nan/inf; set it to the largest non-failed value if so
        max_dist = np.nanmax(result_dists)
        result_dists = np.nan_to_num(result_dists, nan=max_dist, posinf=max_dist, neginf=max_dist)

        print("...finished computing all-pairs geodesic distance")

        # put it in the cache if possible
        if geodesic_cache_dir is not None:

            print("saving geodesic distances to cache: " + str(geodesic_cache_dir))

            # TODO we're potentially saving a double precision but only using a single
            # precision here; could save storage by always saving as floats
            np.savez(search_path,
                     verts=verts_np,
                     faces=faces_np,
                     dist=result_dists
                     )

    return result_dists
