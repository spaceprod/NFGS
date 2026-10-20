"""Boundary-current contract for finite physical calculation domains."""
from types import SimpleNamespace

import pytest
import torch

from nfgs_solver import NFGSSolverGPU
from nfgs_solver import StableNFPTSolverGPU


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


@pytest.mark.parametrize('mode', [1, 2, 3, 4])
@pytest.mark.parametrize('padding', ['symmetric', 'right_only', 'top_only', 'observed_embedding'])
def test_padded_domains_use_exact_legacy_current(mode, padding):
    # Exercise positive and negative currents, including nonzero edge values
    # that distinguish the legacy helper grid from the unpadded stencil.
    solver = NFGSSolverGPU.__new__(NFGSSolverGPU)
    solver.Nx, solver.Ny, solver.Nz = 10, 8, 7
    solver.dx, solver.dy = .3, .7
    solver._calculation_pad_xy = (0, 0)
    solver._input_shape_zyx = (7, 8, 10)
    solver._observed_shape_zyx = None
    if padding == 'symmetric':
        solver._input_shape_zyx = (7, 6, 8)
        solver._calculation_pad_xy = (1, 1)
    elif padding == 'right_only':
        solver._input_shape_zyx = (7, 8, 9)
    elif padding == 'top_only':
        solver._input_shape_zyx = (6, 8, 10)
    else:
        # Explicitly constructed larger domains acquire this metadata when
        # set_bottom_boundary embeds the smaller measured footprint.
        solver._observed_shape_zyx = (7, 6, 8)
    y, x = torch.meshgrid(torch.arange(8, dtype=torch.float64),
                          torch.arange(10, dtype=torch.float64), indexing='ij')
    bx, by = .2*y**2, torch.sin(x)*(1+y)
    expected = StableNFPTSolverGPU._derive_bottom_jz_fortran(solver, bx, by, ijzbcor=mode)
    actual = solver._derive_bottom_jz_fortran(bx, by, ijzbcor=mode)
    assert torch.equal(actual, expected)
    assert solver._bottom_current_grid_mode == 'legacy_helper'


def _configured(cls, **kwargs):
    solver = cls(11, 9, 7, 1., 1., 1., device='cpu', dtype=torch.float64,
                 buffer_thickness=0, source_zero_mode_policy='project',
                 d0_zero_mode_policy='project', **kwargs)
    y, x = torch.meshgrid(torch.arange(9, dtype=torch.float64),
                          torch.arange(11, dtype=torch.float64), indexing='ij')
    solver.set_bottom_vector_boundary(.2*y**2, torch.sin(x)*(1+y),
                                      2+torch.cos(x)*torch.cos(y), smooth_passes=0)
    return solver


@pytest.mark.parametrize('domain_mode', ['mudpack', 'mudpack_legacy', 'bottom_padding'])
def test_automatic_padding_matches_legacy_initialization_and_evolution(domain_mode):
    class LegacyNFGS(NFGSSolverGPU):
        _derive_bottom_jz_fortran = StableNFPTSolverGPU._derive_bottom_jz_fortran

    kwargs = dict(pad_calculation_domain=True, padded_buffer_zone=True,
                  calculation_domain_mode=domain_mode)
    actual, legacy = _configured(NFGSSolverGPU, **kwargs), _configured(LegacyNFGS, **kwargs)
    assert actual._bottom_current_grid_mode == 'legacy_helper'
    assert torch.equal(actual.Jz_obs, legacy.Jz_obs)
    state, expected = actual.compute_initial_conditions(), legacy.compute_initial_conditions()
    for _ in range(2):
        for a, b in zip(state, expected):
            assert torch.equal(a, b)
        state = actual.iteration_step(*state, mu=.3)
        expected = legacy.iteration_step(*expected, mu=.3)
    for a, b in zip(state, expected):
        assert torch.equal(a, b)


def test_buffer_permission_without_grid_extension_keeps_unpadded_policy():
    solver = _configured(NFGSSolverGPU, pad_calculation_domain=False, padded_buffer_zone=True)
    assert solver._bottom_current_grid_mode == 'calculation_grid'
    assert torch.count_nonzero(solver.Jz_obs[[0, -1], :]) == 0
    assert torch.count_nonzero(solver.Jz_obs[:, [0, -1]]) == 0


def test_explicit_embedding_selects_legacy_policy():
    solver = _configured(NFGSSolverGPU, pad_calculation_domain=False, padded_buffer_zone=True)
    by = torch.arange(7*9, dtype=torch.float64).reshape(7, 9)/50
    bx, bz = by.sin(), 2+by.cos()
    solver.set_bottom_vector_boundary(bx, by, bz, smooth_passes=0)
    expected = StableNFPTSolverGPU._derive_bottom_jz_fortran(solver, solver.Bx_obs, solver.By_obs)
    assert solver._bottom_current_grid_mode == 'legacy_helper'
    assert torch.equal(solver.Jz_obs, expected)


def test_explicit_current_is_preserved_and_not_rederived():
    solver = _configured(NFGSSolverGPU, pad_calculation_domain=False, padded_buffer_zone=True)
    jz = torch.ones_like(solver.Bz_obs)*.25
    solver.set_bottom_boundary(solver.Bz_obs, Jz_obs=jz, Bx_obs=solver.Bx_obs,
                               By_obs=solver.By_obs, zero_net_current=False, smooth_passes=0)
    assert solver._bottom_current_grid_mode == 'provided'
    assert torch.equal(solver.Jz_obs, jz)
