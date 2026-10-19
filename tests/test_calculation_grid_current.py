"""Boundary-current contract for finite physical calculation domains."""
from types import SimpleNamespace

import pytest
import torch

from nfgs_solver import NFGSSolverGPU


def test_linear_vector_field_has_correct_interior_curl_and_closed_edges():
    # A non-MUDPACK rectangular domain must still use its own physical edges.
    y, x = torch.meshgrid(torch.arange(8, dtype=torch.float64) * .7,
                          torch.arange(10, dtype=torch.float64) * .3, indexing='ij')
    grid = SimpleNamespace(Nx=10, Ny=8, dx=.3, dy=.7)
    jz = NFGSSolverGPU._derive_bottom_jz_fortran(grid, 2*y, 5*x, ijzbcor=1)
    torch.testing.assert_close(jz[1:-1, 1:-1], torch.full((6, 8), 3., dtype=torch.float64))
    assert torch.count_nonzero(jz[[0, -1], :]) == 0
    assert torch.count_nonzero(jz[:, [0, -1]]) == 0


@pytest.mark.parametrize('mode', [2, 3, 4])
def test_balancing_preserves_current_signs_and_equalizes_signed_integrals(mode):
    y, x = torch.meshgrid(torch.arange(8, dtype=torch.float64),
                          torch.arange(10, dtype=torch.float64), indexing='ij')
    grid = SimpleNamespace(Nx=10, Ny=8, dx=1., dy=1.)
    bx, by = .2 * y**2, torch.sin(x) * (1+y)
    raw = NFGSSolverGPU._derive_bottom_jz_fortran(grid, bx, by, ijzbcor=1)
    balanced = NFGSSolverGPU._derive_bottom_jz_fortran(grid, bx, by, ijzbcor=mode)
    assert torch.equal(torch.sign(raw), torch.sign(balanced))
    # Independent cell quadrature: each cell on the retained inner rectangle
    # contributes the average of its four corner values.
    cell_average = .25 * (balanced[1:-1, 1:-1] + balanced[2:, 1:-1]
                          + balanced[1:-1, 2:] + balanced[2:, 2:])
    assert abs(float(cell_average.sum())) < 1e-12


def test_unembedded_input_is_rejected():
    grid = SimpleNamespace(Nx=10, Ny=8, dx=1., dy=1.)
    with pytest.raises(ValueError, match='embedded'):
        NFGSSolverGPU._derive_bottom_jz_fortran(grid, torch.zeros(6, 8), torch.zeros(6, 8))
