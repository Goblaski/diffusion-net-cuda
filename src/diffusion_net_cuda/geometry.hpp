#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <vector>

#include <torch/extension.h>

std::vector<torch::Tensor> build_grad(
    const torch::Tensor verts,
    const torch::Tensor edges,
    const torch::Tensor edge_tangent_vectors,
    const uint32_t n_neigh
);

void assign_vert_edge_outgoing_cuda(
    torch::Tensor edges,
    torch::Tensor vert_edge_outgoing,
    torch::Tensor vert_edge_outgoing_count
);

torch::Tensor vertices_mapping_lookup(
    const torch::Tensor vertices_new,
    const torch::Tensor vertices_old
);

torch::Tensor vertices_mapping_close(
    const torch::Tensor vertices_lookup,
    const torch::Tensor vertices_marker,
    const float max_distance_squared
);

void vertices_mapping_lookup_cuda(
    const int num_vertices_new,
    const int num_vertices_old,
    torch::Tensor vertices_new,
    torch::Tensor vertices_old,
    torch::Tensor mapping_new_to_old
);

void vertices_mapping_close_cuda(
    const int num_vertices_lookup,
    const int num_vertices_marker,
    const float max_distance_squared,
    torch::Tensor vertices_lookup,
    torch::Tensor vertices_marker,
    torch::Tensor mapped_close_vertices
);

torch::Tensor get_minv_matrix(
    const torch::Tensor normals
);

torch::Tensor get_minv_matrix(
    const torch::Tensor normals,
    const torch::Tensor k
);

void get_minv_matrix_cuda(
    const int num_normals,
    const torch::Tensor normals,
    const torch::Tensor k,
    torch::Tensor minv
);

void build_grad_compressed_cuda(
        const int num_vertices,
        const int max_nhood,
        torch::Tensor edges,
        torch::Tensor edge_tangent_vectors,
        torch::Tensor vert_edge_outgoing,
        torch::Tensor vert_edge_outgoing_count,
        torch::Tensor row_inds,
        torch::Tensor col_inds,
        torch::Tensor data_vals_real,
        torch::Tensor data_vals_imag,
        const float eps_reg,
        const float w_e
);

#endif
