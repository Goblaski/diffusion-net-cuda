#include <vector>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "macros.hpp"
#include "geometry.hpp"
#include "geometry_cuda.hcu"

cudaEvent_t startTimer() {
    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEventRecord(start);
    return start;
}

void stopTimer(cudaEvent_t start, std::string kernelName) {
    cudaEvent_t stop;
    cudaEventCreate(&stop);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << kernelName << ": " << milliseconds << "ms" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void checkError(std::string kernelName) {
    std::string error = cudaGetErrorString(cudaPeekAtLastError());
    std::cout << kernelName << ": " << error << std::endl;
    error = cudaGetErrorString(cudaDeviceSynchronize());
    std::cout << kernelName << ": " << error << std::endl;
}

void assign_vert_edge_outgoing_cuda(
    torch::Tensor edges, torch::Tensor vert_edge_outgoing,
    torch::Tensor vert_edge_outgoing_count
) {
    if (edges.size(1) == 0) return;
    const int blocks = (edges.size(1) + BLOCK_SIZE - 1) / BLOCK_SIZE;
#if DEBUG
    cudaEvent_t start = startTimer();
#endif
    kernel::assign_vert_edge_outgoing_cuda_kernel
        <<<blocks, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
        edges.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
        vert_edge_outgoing.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
        vert_edge_outgoing_count.packed_accessor32<int,1,torch::RestrictPtrTraits>()
    );
#if DEBUG
    stopTimer(start, "assign_vert_edge_outgoing_cuda");
    checkError("assign_vert_edge_outgoing_cuda");
#endif
}

void vertices_mapping_lookup_cuda(
    const int num_vertices_new, const int num_vertices_old,
    torch::Tensor vertices_new, torch::Tensor vertices_old,
    torch::Tensor mapping_new_to_old
) {
    if (num_vertices_new == 0) return;
    const int blocks = (num_vertices_new + BLOCK_SIZE - 1) / BLOCK_SIZE;
#if DEBUG
    cudaEvent_t start = startTimer();
#endif
    kernel::vertices_mapping_lookup_cuda_kernel<<<blocks, BLOCK_SIZE>>>(
        num_vertices_new, num_vertices_old,
        vertices_new.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        vertices_old.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        mapping_new_to_old.packed_accessor32<int,1,torch::RestrictPtrTraits>()
    );
#if DEBUG
    stopTimer(start, "vertices_mapping_lookup_cuda");
    checkError("vertices_mapping_lookup_cuda");
#endif
}

void vertices_mapping_close_cuda(
    const int num_vertices_lookup, const int num_vertices_marker,
    const float max_distance_squared,
    torch::Tensor vertices_lookup, torch::Tensor vertices_marker,
    torch::Tensor mapped_close_vertices
) {
    if (num_vertices_lookup == 0) return;
    const int blocks = (num_vertices_lookup + BLOCK_SIZE - 1) / BLOCK_SIZE;
#if DEBUG
    cudaEvent_t start = startTimer();
#endif
    kernel::vertices_mapping_close_cuda_kernel<<<blocks, BLOCK_SIZE>>>(
        num_vertices_lookup, num_vertices_marker, max_distance_squared,
        vertices_lookup.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        vertices_marker.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        mapped_close_vertices.packed_accessor32<int,1,torch::RestrictPtrTraits>()
    );
#if DEBUG
    stopTimer(start, "vertices_mapping_close_cuda");
    checkError("vertices_mapping_close_cuda");
#endif
}

void get_minv_matrix_cuda(
    const int num_normals, const torch::Tensor normals,
    const torch::Tensor k, torch::Tensor minv
) {
    if (num_normals == 0) return;
    const int blocks = (num_normals + BLOCK_SIZE - 1) / BLOCK_SIZE;
#if DEBUG
    cudaEvent_t start = startTimer();
#endif
    kernel::get_minv_matrix_cuda_kernel<<<blocks, BLOCK_SIZE>>>(
        num_normals,
        normals.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        k.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        minv.packed_accessor32<float,3,torch::RestrictPtrTraits>()
    );
#if DEBUG
    stopTimer(start, "get_minv_matrix_cuda");
    checkError("get_minv_matrix_cuda");
#endif
}

void build_grad_compressed_cuda(
    const int num_vertices, const int max_nhood,
    torch::Tensor edges, torch::Tensor edge_tangent_vectors,
    torch::Tensor vert_edge_outgoing,
    torch::Tensor vert_edge_outgoing_count,
    torch::Tensor row_inds, torch::Tensor col_inds,
    torch::Tensor data_vals_real, torch::Tensor data_vals_imag,
    const float eps_reg, const float w_e
) {
    if (num_vertices == 0) return;
    const int blocks = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
#if DEBUG
    cudaEvent_t start = startTimer();
#endif
    kernel::build_grad_compressed_cuda_kernel
        <<<blocks, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
        num_vertices, max_nhood,
        edges.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
        edge_tangent_vectors.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        vert_edge_outgoing.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
        vert_edge_outgoing_count.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
        row_inds.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
        col_inds.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
        data_vals_real.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
        data_vals_imag.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
        eps_reg, w_e
    );
#if DEBUG
    stopTimer(start, "build_grad_compressed_cuda");
    checkError("build_grad_compressed_cuda");
#endif
}
