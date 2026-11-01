#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "macros.hpp"
#include "geometry_cuda.hcu"

#define CUDART_INF_F __int_as_float(0x7f800000)

// Existing helper kernels retained unchanged from the fork.
__global__ void kernel::vertices_mapping_lookup_cuda_kernel(
    const int num_vertices_new, const int num_vertices_old,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> vertices_new,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> vertices_old,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> mapping_new_to_old
) {
    int iV = blockDim.x * blockIdx.x + threadIdx.x;
    if (iV < num_vertices_new) {
        float v_x = vertices_new[iV][0];
        float v_y = vertices_new[iV][1];
        float v_z = vertices_new[iV][2];
        float s_dist = CUDART_INF_F;
        int s_id = -1;
        float d_x, d_y, d_z, s;
        for (int i = 0; i < num_vertices_old; i++) {
            d_x = vertices_old[i][0] - v_x;
            d_y = vertices_old[i][1] - v_y;
            d_z = vertices_old[i][2] - v_z;
            s = d_x * d_x + d_y * d_y + d_z * d_z;
            if (s < s_dist) {
                s_dist = s;
                s_id = i;
            }
        }
        mapping_new_to_old[iV] = s_id;
    }
}

__global__ void kernel::vertices_mapping_close_cuda_kernel(
    const int num_vertices_lookup, const int num_vertices_marker,
    const float max_distance_squared,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> vertices_lookup,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> vertices_marker,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> mapped_close_vertices
) {
    int iV = blockDim.x * blockIdx.x + threadIdx.x;
    if (iV < num_vertices_lookup) {
        float v_x = vertices_lookup[iV][0];
        float v_y = vertices_lookup[iV][1];
        float v_z = vertices_lookup[iV][2];
        float d_x, d_y, d_z, s;
        bool found = false;
        for (int i = 0; i < num_vertices_marker; i++) {
            d_x = vertices_marker[i][0] - v_x;
            d_y = vertices_marker[i][1] - v_y;
            d_z = vertices_marker[i][2] - v_z;
            s = d_x * d_x + d_y * d_y + d_z * d_z;
            if (s < max_distance_squared) {
                found = true;
                break;
            }
        }
        mapped_close_vertices[iV] = found ? 1 : 0;
    }
}

__global__ void kernel::get_minv_matrix_cuda_kernel(
    const int num_normals,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> normals,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> k,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> minv
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_normals) {
        float l_0 = k[i][1]*normals[i][2] - k[i][2]*normals[i][1];
        float l_1 = k[i][2]*normals[i][0] - k[i][0]*normals[i][2];
        float l_2 = k[i][0]*normals[i][1] - k[i][1]*normals[i][0];
        float len = sqrt(l_0*l_0 + l_1*l_1 + l_2*l_2);
        l_0 /= len; l_1 /= len; l_2 /= len;
        float kl_0 = l_1*normals[i][2] - l_2*normals[i][1];
        float kl_1 = l_2*normals[i][0] - l_0*normals[i][2];
        float kl_2 = l_0*normals[i][1] - l_1*normals[i][0];
        float kl_len = sqrt(kl_0*kl_0 + kl_1*kl_1 + kl_2*kl_2);
        kl_0 /= kl_len; kl_1 /= kl_len; kl_2 /= kl_len;
        minv[0][0][i] = normals[i][0];
        minv[1][0][i] = normals[i][1];
        minv[2][0][i] = normals[i][2];
        minv[0][1][i] = l_0;
        minv[1][1][i] = l_1;
        minv[2][1][i] = l_2;
        minv[0][2][i] = kl_0;
        minv[1][2][i] = kl_1;
        minv[2][2][i] = kl_2;
    }
}

__global__ void kernel::assign_vert_edge_outgoing_cuda_kernel(
    const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> edges,
    torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> vert_edge_outgoing,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> vert_edge_outgoing_count
) {
    const int eId = blockDim.x * blockIdx.x + threadIdx.x;
    if (eId >= edges.size(1)) return;

    const int tail_ind = edges[0][eId];
    const int tip_ind = edges[1][eId];
    if (tip_ind == tail_ind) return;

    const int location = atomicAdd(&vert_edge_outgoing_count[tail_ind], 1);
    // Keep writes in bounds, even when the caller supplied too little storage.
    // The host checks degree counts and raises before the gradient kernel runs.
    if (location < vert_edge_outgoing.size(1)) {
        vert_edge_outgoing[tail_ind][location] = eId;
    }
}

__global__ void kernel::build_grad_compressed_cuda_kernel(
    const int num_vertices, const int max_nhood,
    const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> edges,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> edge_tangent_vectors,
    const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> vert_edge_outgoing,
    const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> vert_edge_outgoing_count,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> row_inds,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> col_inds,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> data_vals_real,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> data_vals_imag,
    const float eps_reg, const float w_e
) {
    const int iV = blockDim.x * blockIdx.x + threadIdx.x;
    if (iV >= num_vertices) return;

    const int n = vert_edge_outgoing_count[iV];
    const int offset = iV * (max_nhood + 1);

    // Same local regularized least-squares formulation as NumPy CPU version:
    // (A^T A + eps I)^{-1} A^T B, with two nonzeros per row of B.
    // Double intermediates improve stability while preserving float output.
    double a00 = static_cast<double>(eps_reg);
    double a01 = 0.0;
    double a11 = static_cast<double>(eps_reg);
    for (int j = 0; j < n; ++j) {
        const int iE = vert_edge_outgoing[iV][j];
        const double x = static_cast<double>(w_e) * edge_tangent_vectors[iE][0];
        const double y = static_cast<double>(w_e) * edge_tangent_vectors[iE][1];
        a00 += x*x;
        a01 += x*y;
        a11 += y*y;
    }

    const double det = a00*a11 - a01*a01;
    const double inv00 =  a11 / det;
    const double inv01 = -a01 / det;
    const double inv11 =  a00 / det;
    double center_x = 0.0;
    double center_y = 0.0;

    for (int j = 0; j < n; ++j) {
        const int iE = vert_edge_outgoing[iV][j];
        const int jV = edges[1][iE];
        const double x = static_cast<double>(w_e) * edge_tangent_vectors[iE][0];
        const double y = static_cast<double>(w_e) * edge_tangent_vectors[iE][1];
        const double gx = static_cast<double>(w_e) * (inv00*x + inv01*y);
        const double gy = static_cast<double>(w_e) * (inv01*x + inv11*y);
        const int idx = offset + j + 1;
        row_inds[idx] = iV;
        col_inds[idx] = jV;
        data_vals_real[idx] = static_cast<float>(gx);
        data_vals_imag[idx] = static_cast<float>(gy);
        center_x -= gx;
        center_y -= gy;
    }

    row_inds[offset] = iV;
    col_inds[offset] = iV;
    data_vals_real[offset] = static_cast<float>(center_x);
    data_vals_imag[offset] = static_cast<float>(center_y);
}
