# FastQSL provenance

This is the minimal executable FastQSL source tree copied from
`archive/review_before_deletion/reference_code_noncore/third_party/FastQSL`
on 2026-07-19.  It intentionally excludes archived notebooks, generated
figures, test caches, and the nested Git metadata.

The analytic-validation entry point uses `fastqslpy.kernels` for GPU field-line
tracing and `fastqslpy.FastQSL.QCalcPlane` for the (y=0) TD diagnostic plane.
See `scripts/run_analytic_validation.py --td-topology-qsl --topology-only --help`.

Local patches retained here:

1. Import safety: `QCalcPlane` creates its default normal vector inside the
   function instead of at module import, so an import check can run before a
   CUDA context exists.
2. Device-heap safety (2026-07-21): `TraceAllBline` in `TraceBline*.cu` uses
   stack locals instead of device `new`/`delete`. Concurrent launches of a
   full-resolution `y=0` plane (~5e4 threads) used to exhaust the default CUDA
   malloc heap and raise `cudaErrorIllegalAddress` on NFGS fields.
