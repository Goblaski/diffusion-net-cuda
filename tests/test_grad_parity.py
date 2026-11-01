"""CUDA-vs-original-CPU gradient parity checks. Run after rebuilding the extension."""
import numpy as np
import torch

from diffusion_net.geometry import build_grad, build_grad_cuda


def check_case(name, xy, edge_pairs, atol=2e-4, rtol=2e-3):
    xy = torch.tensor(xy, dtype=torch.float32)
    vertices = torch.cat((xy, torch.zeros((len(xy), 1))), dim=1)
    edges = torch.tensor(edge_pairs, dtype=torch.int64).T.contiguous()
    assert edges.shape[0] == 2
    tangent = (vertices[edges[1]] - vertices[edges[0]])[:, :2].contiguous()

    reference = build_grad(vertices, edges, tangent).toarray()
    grad_x, grad_y = build_grad_cuda(
        vertices.cuda(), edges.int().cuda(), tangent.cuda(),
        n_neighbors_cloud=12, return_device='cpu',
    )
    actual = grad_x.to_dense().numpy() + 1j * grad_y.to_dense().numpy()
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol)
    np.testing.assert_allclose(actual @ np.ones(len(xy)),
                               np.zeros(len(xy)), atol=2e-5)
    maximum_error = float(np.max(np.abs(actual - reference)))
    print(f'PASS {name}  max error = {maximum_error:.3g}')


def main():
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is not available; run this test on a CUDA GPU')

    square = [[0., 0.], [1., 0.], [0., 1.], [1., 1.]]
    all_directed = [(i, j) for i in range(4) for j in range(4) if i != j]
    check_case('square', square, all_directed)

    # The original default capacity (12) is insufficient for the center.
    angles = np.linspace(0, 2*np.pi, 24, endpoint=False)
    star = np.vstack([[[0., 0.]], np.column_stack((np.cos(angles),
                                                   np.sin(angles)))])
    star_edges = [(0, j) for j in range(1, 25)] + [(j, 0) for j in range(1, 25)]
    check_case('degree-24 star', star, star_edges)

    rng = np.random.default_rng(31)
    xy = rng.normal(size=(30, 2))
    edges = [(i, j) for i in range(30) for j in range(30) if i != j]
    rng.shuffle(edges)  # Different atomic insertion order than the CPU traversal.
    check_case('high-valence random', xy, edges)

    # Duplicate neighbors should be summed correctly by COO coalescing.
    check_case('duplicate edges', square, all_directed + [(0, 1), (0, 1)])


if __name__ == '__main__':
    main()
