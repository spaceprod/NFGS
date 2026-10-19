# Manuscript reproduction map

| Manuscript evidence | Public experiment | Configuration | Main output |
| --- | --- | --- | --- |
| TD CCI scan and reconstruction metrics | `td_rho_scan.py` | `td_rho_scan.json` | `rho_full/metrics_all.csv` and saved fields |
| Primary scoring domains and common curl | `common_diagnostics.py` | `common_diagnostics.json` | `common_diagnostics/metrics_by_domain.csv` |
| Practical rho choice | `rho_figures.py` | `rho_figures.json` | `figures/` |
| Low--Lou LL1, LL2, LL3 at rho=0.5 | `low_lou_fp32.py`, `low_lou_fp64.py` | respective JSON files | `low_lou_family_rho0p5/` |
| Low--Lou rho=0 and 1 baselines | `low_lou_rho0.py`, `low_lou_rho1.py` | respective JSON files | `low_lou_baselines/` |
| Unified current evidence | `current_evidence.py` | `current_evidence.json` | `evidence_summary.json`, `low_lou_comparison.json` |
| Appendix A limiter/floor activation | `regularization.py` | `regularization.json` | `regularization/regularization_metrics.csv` |
| Q maps, footprint overlap and endpoints | `quantitative_topology.py` | `quantitative_topology.json` | `quantitative_topology/metrics.csv` and raw bundles |
| Twist-core apex and amplitude | `twist_apex.py` | `twist_apex.json` | `twist_apex/metrics.csv` |
| Six-face energy and helicity | `six_face_helicity.py` | `six_face_helicity.json` | `six_face_helicity/metrics.csv` |
| Height profiles, Q thresholds, diagnostic weights | `reader_diagnostics.py` | `reader_diagnostics.json` | `second_reader_revision/` |
| Qualitative total-field views | `fieldline_views.py` | `fieldline_views.json` | `figures/fieldline_comparison.png` |
| Completeness, state selection and finite metrics | `audit.py` | `audit.json` | `reproduction_audit.json` |
| Manufactured modal / complete PT verification | `modal_verification.py`, `extended_manufactured.py` | respective JSON files | `summary.json`, error tables |
| Additional source, formulation and resolution studies | `source_support.py`, `factorized_ablation.py`, `rho_gamma_scaling.py` | respective JSON files | experiment metrics and manifests |

All experiment scripts accept `--config`, `--output-root`, `--profile`,
`--device`, `--dry-run` and `--force`. Use the experiment JSON's `arguments`
object for numerical choices rather than editing Python source. The campaign
config chooses the experiment set, and dependencies are declared in each
experiment config. The integration runner executes those public scripts,
which in turn execute portable numerical routines through the JSON configs.

## Registered scientific protocol

- The primary TD scan includes 0, 0.1, ..., 1 and all three source configurations.
- Each TD case retains its own primary scoring mask. Plotting crops do not
  alter the scoring volume.
- The reported TD current is recomputed from the final B field using one
  common second-order curl, independently of the between-plane monitor.
- The Low--Lou curl uses dx=dy=2/79 and dz=2/71. The three cases use rho=0.5.
- Production selection is `sigma_best`, tolerance 1e-5, patience 40,
  improvement tolerance 1e-5, maximum 3000 iterations.
- CCI uses the re-separated observed/reconstructed photospheric fields.
  The denominator cutoff and the GS slope target remain distinct concepts.
- Q comparisons have zero smoothing and shared valid boundary termination
  conventions; high-Q threshold sensitivity uses the same frozen Q bundles.
- Published NFPT/VFVP/SSW metric rows are attributed historical values. Frozen
  NFPT volume fields are separate comparison inputs, verified before use.

## What the refactoring verified

The single-file export is bitwise equal to the current research solver for
three initialization modes and a nonlinear iteration on a small CPU grid.
The quick integrated campaign executes manufactured tests, a reduced TD scan
and a Low--Lou smoke. All config arguments are checked against the numerical
entry points. The overview notebook is executed with its archived dataset.
A full-size TD2 rho=0.5 reconstruction is also executed on CUDA.

The bundled archived tables preserve the paper's reported numerical values.
Fresh runs retain their own metrics, reference-array fingerprints, library
versions and selected iterations. CUDA/library versions and the heuristic
best-state criterion can change last digits and the selected iteration; a
fresh result is never replaced by an archived value to force agreement.
See `provenance/refactor_validation.json` for the measured checks and the
explicit scope of the reruns. The entire expensive 33-row/topology matrix is
not labelled as rerun unless it has actually been executed.

## Field rendering

The new renderer traces shared reference-derived seeds on the total field,
uses an orthographic camera, 3.8 Mm tubes and the presentation's specular
material settings. Its seed coordinates and display transforms are saved.
TD1's NFPT x reflection is a display transform; it does not alter scoring.
Archived manuscript figures retain the original frozen display traces. The
new renderer provides a view of fresh fields without claiming pixel-for-pixel
identity with the archived manuscript layout.

## Archived data

The versioned comparison-field archive is about 550 MB and is kept outside
Git history. `nfgs.inputs` verifies its archive checksum and each NPZ checksum;
unexpected archive members are rejected. The manifest is included in Git.
`reference_results/` contains small, explicitly archived tables and figures;
`results/` and downloaded fields are ignored by Git.

The current-evidence stage also traces the selected common weight if it is absent from the standard structural diagnostics. The overview notebook reads this choice from the exported summary and includes the Low--Lou CCI baselines.
