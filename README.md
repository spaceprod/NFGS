# NFGS

Gaussian-separated nonlinear force-free reconstruction with explicit control
of the photospheric coronal-current imprint (CCI).

Companion code for **Fast Reconstruction of Solar Coronal Nonlinear Force-Free
Fields using Gaussian Separation with Coronal-Current Imprint Control**.
The production settings reproduce the current manuscript's TD weight scan,
Low--Lou validation at rho = 0.5, common-current metrics, FastQSL connectivity
and twist, six-face energy and helicity, and diagnostic weight sensitivity.

## Layout

```text
nfgs_solver.py             Single-file solver; NumPy + PyTorch only
experiments/              One independently runnable script per experiment
configs/                  Experiment JSON configs, campaign configs, physical protocol
run_experiments.py        Integrated dependency-aware runner with logs and resume
notebooks/results_overview.ipynb  Overview of all result tables and figures
nfgs/                     Runner, diagnostics, portable numerical support modules
examples/                 Reconstruct a user-supplied vector boundary
reference_results/        Clearly labelled archived manuscript result tables
provenance/               Original source hashes and refactoring verification
third_party/FastQSL/      CUDA tracer with its separate license
```

Every public experiment script reads its corresponding JSON config. Numerical
support functions live under `nfgs/_backend`; they import the same single-file
solver as the public API. The original research checkout is not needed.

## Install

Python 3.10--3.12 is supported. Install a PyTorch build appropriate for your
CUDA driver, then install the editable project:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[notebook,topology,render]'
python -m ipykernel install --prefix .venv --name nfgs --display-name 'NFGS (.venv)'
```

NFGS itself runs on CPU or CUDA. Manuscript-sized reconstructions are designed
for CUDA. FastQSL requires an NVIDIA GPU and a CUDA toolkit with `nvcc` on PATH.
If CUDA was installed by the optional pip packages, add its `bin` directory to
PATH. `topology` uses CuPy for CUDA 12; use the matching CuPy distribution for
another toolkit version. CPU numerical checks need only `pip install -e .`.

## Single-file solver

Copy only `nfgs_solver.py` to use the solver independently of this package:

```python
from nfgs_solver import NFGSSolverGPU
solver = NFGSSolverGPU(Nx, Ny, Nz, dx, dy, dz,
    device='cuda', bottom_divergence_robin_weight=0.5)
solver.set_bottom_vector_boundary(Bx, By, Bz, smooth_passes=0)
history = solver.iterate(init_mode='gs_cfcf', max_iter=3000,
    tol=1e-5, patience=40, sigma_improvement_tol=1e-5, stop_mode='sigma_best')
fields = solver.fields_to_numpy(crop_to_observed_fov=True)
```

Boundary components have shape `(Ny, Nx)`; output fields have shape `(Nz, Ny,
Nx)`. The normalized current is `J = curl(B)`; convert to physical SI current
with the permeability factor when the field and length units are SI.
`examples/reconstruct_npz.py` supplies a complete command-line example.

## Reproduce the manuscript

Inspect the plan before running:

```bash
python run_experiments.py --config configs/campaign_paper.json --device cuda --dry-run
python run_experiments.py --config configs/campaign_paper.json --device cuda --resume
```

Archived NFPT comparison inputs are fetched from the versioned data release
and verified with `configs/archived_inputs_manifest.json`. They are data, not
NFGS output caches. A local archive can be used without network access:

```bash
python -m nfgs.inputs --manifest configs/archived_inputs_manifest.json \
  --destination data/nfpt --archive /path/to/nfpt_comparison_fields.tar
```

The default campaign reconstructs 33 TD model-weight pairs and the three
Low--Lou cases in both precisions, then derives the manuscript diagnostics.
`configs/paper_protocol.json` explicitly records source parameters, reference
grids, scoring domains, rho grid, zero-mode treatment, current operator and
stopping settings. TD case parameters are also applied directly from the TD
scan config. `configs/campaign_extended.json` adds manufactured solutions,
source attribution, a factorized formulation bridge and resolution scaling.

Run a single experiment, or selected experiments with their prerequisites:

```bash
python experiments/td_rho_scan.py --config configs/td_rho_scan.json --device cuda
python run_experiments.py --only td_rho_scan,low_lou_fp32 --device cuda --resume
```

Results go to `results/paper/` by default. Each experiment has its own output
folder and log. `campaign_state.json` records the exact configs, commands,
source fingerprint and output checksums. Resume requires all of them to agree.
After changing a config, use a new output root or explicitly use `--force`.
The final audit rejects incomplete TD/Low--Lou matrices and iteration-capped
production results. Selecting the minimum monitored sine is the manuscript's
heuristic state-selection rule; the audit does not establish fixed-point
convergence.

## Short CPU check

```bash
python run_experiments.py --profile quick --device cpu --output-root results/quick
```

This runs small manufactured problems, a reduced TD scan and a three-iteration
Low--Lou smoke. These outputs are labelled as checks, not manuscript evidence.

## Overview notebook

```bash
python -m jupyterlab notebooks/results_overview.ipynb
```

The notebook opens with the bundled `reference_results` tables so every figure
can be inspected immediately. Set `RESULT_ROOT` to `results/paper` to inspect a
fresh campaign. It shows CCI/force-freeness, all five field scores, runtime,
Low--Lou precision, height budgets, Q and endpoint metrics, twist-apex error,
energy/helicity and weight sensitivity together. Missing stages are shown
explicitly. Archived tables are never substituted for a newly executed run.

See `docs/reproduction.md` for the manuscript-to-experiment mapping and
`provenance/refactor_validation.json` for the completed verification.

## Attribution

Please cite the accompanying NFGS manuscript and Yi et al. (2022) for NFPT.
FastQSL retains its CC BY-NC-SA 4.0 license; see `THIRD_PARTY.md`.

`evaluate.py` provides common evaluation of supplied magnetic fields. The private NFPT Fortran solver and unauthorized external solver materials are not distributed.
