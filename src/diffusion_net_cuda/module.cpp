#include <vector>

#include <pybind11/pybind11.h>

#include <torch/extension.h>

#include "geometry.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def(
		"build_grad",
		py::overload_cast<
		    const torch::Tensor,
		    const torch::Tensor,
            const torch::Tensor,
		    const uint32_t
		>(&build_grad),
		"CUDA implementation of build_grad.",
		py::arg("verts"),
		py::arg("edges"),
		py::arg("edge_tangent_vectors"),
		py::arg("n_neigh")
	);

	m.def(
		"vertices_mapping_lookup",
		py::overload_cast<
		    const torch::Tensor,
		    const torch::Tensor
		>(&vertices_mapping_lookup),
		"CUDA implementation of vertices_mapping_lookup.",
		py::arg("vertices_new"),
		py::arg("vertices_old")
	);

	m.def(
		"vertices_mapping_close",
		py::overload_cast<
		    const torch::Tensor,
		    const torch::Tensor,
			const float
		>(&vertices_mapping_close),
		"CUDA implementation of vertices_mapping_close.",
		py::arg("vertices_lookup"),
		py::arg("vertices_marker"),
		py::arg("max_distance_squared")
	);

	m.def(
		"get_minv_matrix",
		py::overload_cast<
		    const torch::Tensor
		>(&get_minv_matrix),
		"CUDA implementation of get_minv_matrix.",
		py::arg("normals")
	);

	m.def(
		"get_minv_matrix",
		py::overload_cast<
		    const torch::Tensor,
		    const torch::Tensor
		>(&get_minv_matrix),
		"CUDA implementation of get_minv_matrix.",
		py::arg("normals"),
		py::arg("k")
	);
// Overload disabled for now. Might be useful in full cuda workflow later
/*	m.def(
		"build_grad",
		py::overload_cast<
		    const std::vector<torch::Tensor>,
		    const std::vector<torch::Tensor>,
            const std::vector<torch::Tensor>,
		    const uint32_t
		>(&build_grad),
		"CUDA implementation of build_grad.",
		py::arg("verts"),
		py::arg("edges"),
		py::arg("edge_tangent_vectors"),
		py::arg("n_neigh")
	);*/
}