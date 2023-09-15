#include <tuple>
#include <vector>

#include <torch/extension.h>

#include "macros.hpp"
#include "geometry.hpp"

//std::vector<torch::Tensor> build_grad(
std::vector<torch::Tensor> build_grad(
    const torch::Tensor verts,
    const torch::Tensor edges,
    const torch::Tensor edge_tangent_vectors,
    const uint32_t n_neigh
) {
    uint32_t neigh = 0;
    if (n_neigh <= 0) {
        neigh = MAX_NHOOD;
    } else {
        neigh = n_neigh;
    }
    // Sanity check(s)
    if (n_neigh > MAX_NHOOD) {
        std::cout << "ERROR: n_neigh > MAX_NHOOD, n_neigh cannot be higher than: " << MAX_NHOOD << std::endl;
        std::cout << "Use a lower n_neigh value." << std::endl;
        std::cout << "Alternatively, recompile with a higher MAX_NHOOD (slower and requires more vram)." << std::endl;
        throw std::invalid_argument("MAX_NHOOD smaller than n_neigh");
    }

    // Get the amount of vertices
    uint32_t n_vertices = (uint32_t)verts.size(0);

    // allocate & initate vert_edge_outgoing
    torch::Tensor vert_edge_outgoing = torch::zeros({n_vertices,neigh}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0));
    torch::Tensor vert_edge_outgoing_count = torch::zeros(n_vertices, torch::dtype(torch::kInt32).device(torch::kCUDA, 0));

    // Assign vert_edge_outgoing
    assign_vert_edge_outgoing_cuda(edges,vert_edge_outgoing,vert_edge_outgoing_count);

    // TODO: Error check

    // allocate & initate build_grad_compressed variables
    torch::Tensor row_ind = torch::full({n_vertices*(MAX_NHOOD+1)}, -1, torch::dtype(torch::kInt32).device(torch::kCUDA, 0));
    torch::Tensor cols_ind = torch::full({n_vertices*(MAX_NHOOD+1)}, -1, torch::dtype(torch::kInt32).device(torch::kCUDA, 0));
    torch::Tensor data_vals_real = torch::zeros({n_vertices*(MAX_NHOOD+1)}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
    torch::Tensor data_vals_imag = torch::zeros({n_vertices*(MAX_NHOOD+1)}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

    // Build compressed grad
    build_grad_compressed_cuda(n_vertices, MAX_NHOOD,
        edges,
        edge_tangent_vectors,
        vert_edge_outgoing,
        vert_edge_outgoing_count,
        row_ind,
        cols_ind,
        data_vals_real,
        data_vals_imag,
        1e-5,
        1.0
    );

    return {row_ind, cols_ind, data_vals_real, data_vals_imag, vert_edge_outgoing_count};

}

torch::Tensor vertices_mapping_lookup(
    const torch::Tensor vertices_new,
    const torch::Tensor vertices_old
) {
    // Get the amount of vertices
    uint32_t num_vertices_new = (uint32_t)vertices_new.size(0);
    uint32_t num_vertices_old = (uint32_t)vertices_old.size(0);

    // allocate & initate vert_edge_outgoing
    torch::Tensor mapping_new_to_old = torch::zeros(num_vertices_new, torch::dtype(torch::kInt32).device(torch::kCUDA, 0));
    

    // Assign vert_edge_outgoing
    vertices_mapping_lookup_cuda(num_vertices_new,num_vertices_old,vertices_new,vertices_old,mapping_new_to_old);


    return mapping_new_to_old;
}

torch::Tensor vertices_mapping_close(
    const torch::Tensor vertices_lookup,
    const torch::Tensor vertices_marker,
    const float max_distance_squared
) {
    // Get the amount of vertices
    uint32_t num_vertices_lookup = (uint32_t)vertices_lookup.size(0);
    uint32_t num_vertices_marker = (uint32_t)vertices_marker.size(0);

    // allocate & initate vert_edge_outgoing
    torch::Tensor mapped_close_vertices = torch::zeros(num_vertices_lookup, torch::dtype(torch::kInt32).device(torch::kCUDA, 0));
    

    // Assign vert_edge_outgoing
    vertices_mapping_close_cuda(num_vertices_lookup,num_vertices_marker,max_distance_squared,vertices_lookup,vertices_marker,mapped_close_vertices);


    return mapped_close_vertices;
}

torch::Tensor get_minv_matrix(
    const torch::Tensor normals
) {
    // Get the amount of vertices
    uint32_t num_normals = (uint32_t)normals.size(0);

    // allocate & initate vert_edge_outgoing
    torch::Tensor minv = torch::zeros({3,3,num_normals}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
    torch::Tensor k = torch::rand({num_normals,3},torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

    // Assign vert_edge_outgoing
    get_minv_matrix_cuda(num_normals,normals,k,minv);


    return minv;
}

torch::Tensor get_minv_matrix(
    const torch::Tensor normals,
    const torch::Tensor k
) {
    // Get the amount of vertices
    uint32_t num_normals = (uint32_t)normals.size(0);

    // allocate & initate vert_edge_outgoing
    torch::Tensor minv = torch::zeros({3,3,num_normals}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

    // Assign vert_edge_outgoing
    get_minv_matrix_cuda(num_normals,normals,k,minv);


    return minv;
}