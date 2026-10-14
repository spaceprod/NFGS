"""Common quantitative topology metrics for Q and field-line endpoint maps."""

from __future__ import annotations

from typing import Any

import numpy as np


def q_map_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    high_logq: float,
) -> dict[str, float | int]:
    """Compare two Q maps using one finite-pixel and threshold contract."""
    ref_logq = np.log10(np.maximum(np.asarray(reference, dtype=float), 1.0))
    cand_logq = np.log10(np.maximum(np.asarray(candidate, dtype=float), 1.0))
    valid = np.isfinite(ref_logq) & np.isfinite(cand_logq)
    if not np.any(valid):
        return {
            "valid_pixels": 0,
            "logq_mae": float("nan"),
            "logq_rmse": float("nan"),
            "logq_pearson": float("nan"),
            "reference_high_q_pixels": 0,
            "candidate_high_q_pixels": 0,
            "high_q_intersection_pixels": 0,
            "high_q_union_pixels": 0,
            "high_q_dice": float("nan"),
            "high_q_jaccard": float("nan"),
        }
    ref = ref_logq[valid]
    cand = cand_logq[valid]
    residual = cand - ref
    ref_high = ref >= float(high_logq)
    cand_high = cand >= float(high_logq)
    intersection = int(np.count_nonzero(ref_high & cand_high))
    reference_high = int(np.count_nonzero(ref_high))
    candidate_high = int(np.count_nonzero(cand_high))
    dice_denominator = int(ref_high.sum() + cand_high.sum())
    union = int(np.count_nonzero(ref_high | cand_high))
    ref_centered = ref - float(np.mean(ref))
    cand_centered = cand - float(np.mean(cand))
    pearson_den = float(
        np.sqrt(np.sum(ref_centered**2) * np.sum(cand_centered**2))
    )
    pearson = (
        float(np.sum(ref_centered * cand_centered) / pearson_den)
        if pearson_den > 0.0
        else (1.0 if np.array_equal(ref, cand) else float("nan"))
    )
    return {
        "valid_pixels": int(valid.sum()),
        "logq_mae": float(np.mean(np.abs(residual))),
        "logq_rmse": float(np.sqrt(np.mean(residual**2))),
        "logq_pearson": pearson,
        "reference_high_q_pixels": reference_high,
        "candidate_high_q_pixels": candidate_high,
        "high_q_intersection_pixels": intersection,
        "high_q_union_pixels": union,
        "high_q_dice": (
            1.0 if dice_denominator == 0 else 2.0 * intersection / dice_denominator
        ),
        "high_q_jaccard": 1.0 if union == 0 else intersection / union,
    }


def endpoint_pair_metrics(
    reference_start: np.ndarray,
    reference_end: np.ndarray,
    candidate_start: np.ndarray,
    candidate_end: np.ndarray,
    *,
    reference_flag_start: np.ndarray | None = None,
    reference_flag_end: np.ndarray | None = None,
    candidate_flag_start: np.ndarray | None = None,
    candidate_flag_end: np.ndarray | None = None,
    spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, float | int]:
    """Measure unordered conjugate-endpoint displacement for common valid seeds.

    FastQSL boundary flags 1--6 denote termination on one of the six numerical
    domain faces.  Flag 0 (no exit) and flag 42 (uncategorized exit) do not
    define a physical conjugate boundary endpoint and are excluded from both
    displacement and boundary-pair statistics.
    """
    with np.errstate(invalid="ignore"):
        arrays = [
            np.asarray(value, dtype=float)
            for value in (
                reference_start,
                reference_end,
                candidate_start,
                candidate_end,
            )
        ]
    if (
        len({value.shape for value in arrays}) != 1
        or arrays[0].ndim < 1
        or arrays[0].shape[-1] != 3
    ):
        raise ValueError("endpoint arrays must have one identical (..., 3) shape")
    scale = np.asarray(spacing_xyz, dtype=float)
    if scale.shape != (3,) or np.any(scale <= 0.0):
        raise ValueError("spacing_xyz must contain three positive values")
    rs, re, cs, ce = [value * scale for value in arrays]
    valid = np.all(np.isfinite(rs), axis=-1) & np.all(np.isfinite(re), axis=-1)
    valid &= np.all(np.isfinite(cs), axis=-1) & np.all(np.isfinite(ce), axis=-1)

    boundary_fraction = float("nan")
    flags = (
        reference_flag_start,
        reference_flag_end,
        candidate_flag_start,
        candidate_flag_end,
    )
    flag_arrays: tuple[np.ndarray, ...] | None = None
    if all(value is not None for value in flags):
        flag_arrays = tuple(np.asarray(value) for value in flags)
        expected_shape = arrays[0].shape[:-1]
        if any(value.shape != expected_shape for value in flag_arrays):
            raise ValueError("endpoint flag arrays must match endpoint leading shape")
        valid &= np.logical_and.reduce(
            [(value >= 1) & (value <= 6) for value in flag_arrays]
        )

    total_seeds = int(valid.size)
    if not np.any(valid):
        return {
            "endpoint_valid_seeds": 0,
            "endpoint_valid_fraction": 0.0,
            "endpoint_pair_rmse": float("nan"),
            "endpoint_pair_mae": float("nan"),
            "endpoint_pair_p95": float("nan"),
            "endpoint_boundary_pair_match_fraction": float("nan"),
        }
    direct_a = np.linalg.norm(cs - rs, axis=-1)
    direct_b = np.linalg.norm(ce - re, axis=-1)
    swapped_a = np.linalg.norm(cs - re, axis=-1)
    swapped_b = np.linalg.norm(ce - rs, axis=-1)
    use_swap = swapped_a**2 + swapped_b**2 < direct_a**2 + direct_b**2
    pair_a = np.where(use_swap, swapped_a, direct_a)[valid]
    pair_b = np.where(use_swap, swapped_b, direct_b)[valid]
    displacement = np.concatenate((pair_a, pair_b))

    if flag_arrays is not None:
        rfs, rfe, cfs, cfe = flag_arrays
        direct_match = (cfs == rfs) & (cfe == rfe)
        swapped_match = (cfs == rfe) & (cfe == rfs)
        boundary_fraction = float(
            np.mean(np.where(use_swap, swapped_match, direct_match)[valid])
        )

    return {
        "endpoint_valid_seeds": int(np.count_nonzero(valid)),
        "endpoint_valid_fraction": float(np.count_nonzero(valid) / total_seeds),
        "endpoint_pair_rmse": float(np.sqrt(np.mean(displacement**2))),
        "endpoint_pair_mae": float(np.mean(displacement)),
        "endpoint_pair_p95": float(np.percentile(displacement, 95.0)),
        "endpoint_boundary_pair_match_fraction": boundary_fraction,
    }


def fastqsl_bundle_metrics(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    high_logq: float,
    spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, float | int]:
    """Combine Q-map and conjugate-endpoint comparisons when endpoints exist."""
    metrics = q_map_metrics(reference["q"], candidate["q"], high_logq=high_logq)
    endpoint_keys = {"start_xyz", "end_xyz", "flag_start", "flag_end"}
    if endpoint_keys.issubset(reference) and endpoint_keys.issubset(candidate):
        metrics.update(
            endpoint_pair_metrics(
                reference["start_xyz"],
                reference["end_xyz"],
                candidate["start_xyz"],
                candidate["end_xyz"],
                reference_flag_start=reference["flag_start"],
                reference_flag_end=reference["flag_end"],
                candidate_flag_start=candidate["flag_start"],
                candidate_flag_end=candidate["flag_end"],
                spacing_xyz=spacing_xyz,
            )
        )
    return metrics
