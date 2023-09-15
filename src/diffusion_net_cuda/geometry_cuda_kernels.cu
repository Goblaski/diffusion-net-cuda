#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "macros.hpp"
#include "geometry_cuda.hcu"

#define CUDART_INF_F __int_as_float(0x7f800000)

__global__ void kernel::vertices_mapping_lookup_cuda_kernel(
    const int num_vertices_new,
    const int num_vertices_old,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> vertices_new,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> vertices_old,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> mapping_new_to_old
) {
    int iV = blockDim.x * blockIdx.x + threadIdx.x;

    if (iV < num_vertices_new) { // Sanity check
        float v_x = vertices_new[iV][0];
        float v_y = vertices_new[iV][1];
        float v_z = vertices_new[iV][2];

        float s_dist = CUDART_INF_F;
        int s_id = -1;
        float d_x, d_y, d_z, s;

        for (int i=0; i<num_vertices_old;i++) {
            d_x = vertices_old[i][0]-v_x;
            d_y = vertices_old[i][1]-v_y;
            d_z = vertices_old[i][2]-v_z;

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
    const int num_vertices_lookup,
    const int num_vertices_marker,
    const float max_distance_squared,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> vertices_lookup,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> vertices_marker,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> mapped_close_vertices
) {
    int iV = blockDim.x * blockIdx.x + threadIdx.x;

    if (iV < num_vertices_lookup) { // Sanity check
        float v_x = vertices_lookup[iV][0];
        float v_y = vertices_lookup[iV][1];
        float v_z = vertices_lookup[iV][2];

        float d_x, d_y, d_z, s;
        bool found = false;

        for (int i=0; i<num_vertices_marker;i++) {
            d_x = vertices_marker[i][0]-v_x;
            d_y = vertices_marker[i][1]-v_y;
            d_z = vertices_marker[i][2]-v_z;

            s = d_x * d_x + d_y * d_y + d_z * d_z;

            if (s < max_distance_squared) {
                found = true;
                break;
            }
        }

        if (found == true) {
            mapped_close_vertices[iV] = 1;
        } else {
            mapped_close_vertices[iV] = 0;
        }
    }
}

__global__ void kernel::get_minv_matrix_cuda_kernel(
    const int num_normals,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> normals,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> k,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> minv
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num_normals) { // Sanity check
        float l_0 = k[i][1]*normals[i][2] - k[i][2]*normals[i][1];
        float l_1 = k[i][2]*normals[i][0] - k[i][0]*normals[i][2];
        float l_2 = k[i][0]*normals[i][1] - k[i][1]*normals[i][0];
        float len = sqrt(l_0*l_0 + l_1*l_1 + l_2*l_2);
        l_0 /= len;
        l_1 /= len;
        l_2 /= len;
        
        float kl_0 = l_1*normals[i][2] - l_2*normals[i][1];
        float kl_1 = l_2*normals[i][0] - l_0*normals[i][2];
        float kl_2 = l_0*normals[i][1] - l_1*normals[i][0];
        float kl_len = sqrt(kl_0*kl_0 + kl_1*kl_1 + kl_2*kl_2);
        kl_0 /= kl_len;
        kl_1 /= kl_len;
        kl_2 /= kl_len;

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
    // Get edge ID
    int eId = blockDim.x * blockIdx.x + threadIdx.x;

    if (eId < edges.size(1)) { // Sanity check
        int tail_ind = edges[0][eId];
        int tip_ind = edges[1][eId];

        if (tip_ind != tail_ind) {
            // Ensure single location access per kernel
            int location = atomicAdd(&vert_edge_outgoing_count[tail_ind],1);
            // Assign edge id to vertex
            vert_edge_outgoing[tail_ind][location] = eId;
        }
    }
}

__global__ void kernel::build_grad_compressed_cuda_kernel(
    const int num_vertices,
    const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> edges,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> edge_tangent_vectors,
    const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> vert_edge_outgoing,
    const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> vert_edge_outgoing_count,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> row_inds,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> col_inds,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> data_vals_real,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> data_vals_imag,
    const float eps_reg,
    const float w_e
) {
    // Get vertex ID
    int iV = blockDim.x * blockIdx.x + threadIdx.x;

    if (iV < num_vertices) { // Sanity check
        // Get current n_neigh
        const int n_neigh = vert_edge_outgoing_count[iV];

        // Determine offset in the global arrays
        int offset = iV*(MAX_NHOOD+1);

        // Allocate arrays for matrix computations
        float lhs_mat[MAX_NHOOD][2];
        float rhs_mat[MAX_NHOOD][MAX_NHOOD+1];
        float lhs_inv[2][MAX_NHOOD];
        float sol_mat[2][MAX_NHOOD+1];

        // Init first row_inds & col_inds
        row_inds[offset] = iV;
        col_inds[offset] = iV;

        sol_mat[0][0] = 0;
        sol_mat[1][0] = 0;

        // iterate over i_neigh
        for (int i_neigh = 0; i_neigh < n_neigh; i_neigh++) {
            // offset for the global arrays
            int cur_index = offset+i_neigh+1;

            // Get edge id for the vertex at i_neigh
            int iE = vert_edge_outgoing[iV][i_neigh];
            // Get corresponding tip edge
            int jV = edges[1][iE];

            // use the for loop for (pre)assigning
            row_inds[cur_index] = iV;
            col_inds[cur_index] = jV;
            lhs_inv[0][i_neigh] = 0;
            lhs_inv[1][i_neigh] = 0;
            sol_mat[0][i_neigh+1] = 0;
            sol_mat[1][i_neigh+1] = 0;

            // w_e multiplication to lhs_mat
            lhs_mat[i_neigh][0] = edge_tangent_vectors[iE][0] * w_e;
            lhs_mat[i_neigh][1] = edge_tangent_vectors[iE][1] * w_e;

            // rhs_mat population
            rhs_mat[i_neigh][0] = w_e * (-1);
            rhs_mat[i_neigh][i_neigh + 1] = w_e * 1;
        }

        // Lazy init
        float lhs_mul[2][2];
        lhs_mul[0][0] = 0;
        lhs_mul[0][1] = 0;
        lhs_mul[1][0] = 0;
        lhs_mul[1][1] = 0;

        // simple matrix multiplication 1 (lhs_mat.T @ lhs_mat)
        for (int i=0;i<2;i++) {
            for (int j=0;j<2;j++) {
                for (int k=0;k<n_neigh;k++) {
                    lhs_mul[i][j] += lhs_mat[k][i] * lhs_mat[k][j];
                }
            }
        }

        // add eps_reg
        lhs_mul[0][0] += eps_reg;
        lhs_mul[1][1] += eps_reg;

        // simple inversion
        float lhs_mul_inv[2][2];
        float det = 1.0/(lhs_mul[0][0]*lhs_mul[1][1]-lhs_mul[0][1]*lhs_mul[1][0]);

        lhs_mul_inv[0][0] = det * lhs_mul[1][1];
        lhs_mul_inv[0][1] = det * -lhs_mul[0][1];
        lhs_mul_inv[1][0] = det * -lhs_mul[1][0];
        lhs_mul_inv[1][1] = det * lhs_mul[1][1];

        // simple matrix multiplication 2 (lhs_mul_inv @ lhs_mat.T)
        for (int i=0;i<2;i++) {
            for (int j=0;j<n_neigh;j++) {
                for (int k=0;k<2;k++) {
                    //lhs_inv[i][j] += lhs_mul[k][i] * lhs_mat[j][k];
                    lhs_inv[i][j] += lhs_mul_inv[k][i] * lhs_mat[j][k];
                }
            }
        }

        // simple matrix multiplication 3 (sol_mat = lhs_inv @ rhs_mat)
        for (int i=0;i<(n_neigh);i++) {
            for (int j=0;j<(n_neigh+1);j++) {
                for (int k=0;k<2;k++) {
                    sol_mat[k][j] += lhs_inv[k][i] * rhs_mat[i][j];
                }
            }
        }

        // Assign sol_mat
        for (int i_neigh=0; i_neigh<(n_neigh+1); i_neigh++) {
            data_vals_real[offset+i_neigh] = sol_mat[0][i_neigh];
            data_vals_imag[offset+i_neigh] = sol_mat[1][i_neigh];
        }
    }
}