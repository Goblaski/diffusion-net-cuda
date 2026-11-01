#include <tuple>
#include <vector>
#include <limits>

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include "macros.hpp"
#include "geometry.hpp"

std::vector<torch::Tensor> build_grad(
    const torch::Tensor verts,
    const torch::Tensor edges,
    const torch::Tensor edge_tangent_vectors,
    const uint32_t n_neigh
) {
    TORCH_CHECK(verts.is_cuda(), "verts must be on CUDA");
    TORCH_CHECK(edges.is_cuda(), "edges must be on CUDA");
    TORCH_CHECK(edge_tangent_vectors.is_cuda(), "edge_tangent_vectors must be on CUDA");
    TORCH_CHECK(verts.device() == edges.device() &&
                edges.device() == edge_tangent_vectors.device(),
                "All inputs must be on the same CUDA device");
    TORCH_CHECK(verts.dim() == 2 && verts.size(1) == 3,
                "verts must have shape (V,3)");
    TORCH_CHECK(edges.dim() == 2 && edges.size(0) == 2,
                "edges must have shape (2,E)");
    TORCH_CHECK(edge_tangent_vectors.dim() == 2 &&
                edge_tangent_vectors.size(1) == 2 &&
                edge_tangent_vectors.size(0) == edges.size(1),
                "edge_tangent_vectors must have shape (E,2)");
    TORCH_CHECK(edges.scalar_type() == torch::kInt32,
                "edges must be int32");
    TORCH_CHECK(edge_tangent_vectors.scalar_type() == torch::kFloat32,
                "edge_tangent_vectors must be float32");
    TORCH_CHECK(n_neigh > 0, "n_neigh must be positive");

    const int64_t n_vertices = verts.size(0);
    const int64_t stride = static_cast<int64_t>(n_neigh) + 1;
    const int64_t int_limit = std::numeric_limits<int>::max();
    TORCH_CHECK(n_vertices <= int_limit && n_neigh < int_limit &&
                n_vertices * stride <= int_limit,
                "Input exceeds 32-bit kernel indexing capacity");

    c10::cuda::CUDAGuard device_guard(verts.device());

    // Protect against invalid edge indices before accessing vertex adjacency.
    if (edges.numel() > 0) {
        TORCH_CHECK(n_vertices > 0, "non-empty edges for empty vertex set");
        TORCH_CHECK(edges.min().item<int>() >= 0 &&
                    edges.max().item<int>() < n_vertices,
                    "edges contain an out-of-range vertex index");
    }

    const auto int_opts = verts.options().dtype(torch::kInt32);
    const auto float_opts = verts.options().dtype(torch::kFloat32);

    auto vert_edge_outgoing = torch::zeros({n_vertices, static_cast<int64_t>(n_neigh)}, int_opts);
    auto vert_edge_outgoing_count = torch::zeros({n_vertices}, int_opts);

    assign_vert_edge_outgoing_cuda(edges, vert_edge_outgoing, vert_edge_outgoing_count);

    // The CUDA insertion kernel has a bounds guard, and these counts let us
    // fail cleanly before the gradient kernel could read truncated adjacency.
    const int max_degree = n_vertices > 0
        ? vert_edge_outgoing_count.max().item<int>() : 0;
    TORCH_CHECK(max_degree <= n_neigh,
                "Neighborhood capacity exceeded: degree=", max_degree,
                ", allocated=", n_neigh);

    const int64_t n_slots = n_vertices * stride;
    auto row_ind = torch::full({n_slots}, -1, int_opts);
    auto col_ind = torch::full({n_slots}, -1, int_opts);
    auto data_real = torch::zeros({n_slots}, float_opts);
    auto data_imag = torch::zeros({n_slots}, float_opts);

    build_grad_compressed_cuda(
        static_cast<int>(n_vertices), static_cast<int>(n_neigh),
        edges, edge_tangent_vectors, vert_edge_outgoing,
        vert_edge_outgoing_count, row_ind, col_ind,
        data_real, data_imag, 1e-5f, 1.0f
    );

    return {row_ind, col_ind, data_real, data_imag, vert_edge_outgoing_count};
}

torch::Tensor vertices_mapping_lookup(
    const torch::Tensor vertices_new,
    const torch::Tensor vertices_old
) {
    uint32_t num_vertices_new = (uint32_t)vertices_new.size(0);
    uint32_t num_vertices_old = (uint32_t)vertices_old.size(0);
    torch::Tensor mapping_new_to_old = torch::zeros(
        num_vertices_new, torch::dtype(torch::kInt32).device(torch::kCUDA, 0));
    vertices_mapping_lookup_cuda(num_vertices_new, num_vertices_old,
                                 vertices_new, vertices_old, mapping_new_to_old);
    return mapping_new_to_old;
}

torch::Tensor vertices_mapping_close(
    const torch::Tensor vertices_lookup,
    const torch::Tensor vertices_marker,
    const float max_distance_squared
) {
    uint32_t num_vertices_lookup = (uint32_t)vertices_lookup.size(0);
    uint32_t num_vertices_marker = (uint32_t)vertices_marker.size(0);
    torch::Tensor mapped_close_vertices = torch::zeros(
        num_vertices_lookup, torch::dtype(torch::kInt32).device(torch::kCUDA, 0));
    vertices_mapping_close_cuda(num_vertices_lookup, num_vertices_marker,
                                max_distance_squared, vertices_lookup,
                                vertices_marker, mapped_close_vertices);
    return mapped_close_vertices;
}

torch::Tensor get_minv_matrix(const torch::Tensor normals) {
    uint32_t num_normals = (uint32_t)normals.size(0);
    torch::Tensor minv = torch::zeros({3, 3, num_normals},
        torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
    torch::Tensor k = torch::rand({num_normals, 3},
        torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
    get_minv_matrix_cuda(num_normals, normals, k, minv);
    return minv;
}

torch::Tensor get_minv_matrix(const torch::Tensor normals,
                              const torch::Tensor k) {
    uint32_t num_normals = (uint32_t)normals.size(0);
    torch::Tensor minv = torch::zeros({3, 3, num_normals},
        torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
    get_minv_matrix_cuda(num_normals, normals, k, minv);
    return minv;
}
