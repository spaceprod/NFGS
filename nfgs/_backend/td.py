from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class TDModelConfig:
    """Dimensionful Titov--Demoulin parameters in the paper convention.

    Lengths are in Mm, ``q`` is in ``10^12 Wb`` and ``I0`` is in
    ``10^12 A``.  In particular, ``I0`` keeps the sign printed in the paper;
    the archived analytic implementation uses the opposite internal current
    direction for the same Cartesian field orientation.
    """

    R: float
    a: float
    L: float
    d: float
    q: float
    I0: float
    Nt: float
    l_i: float = 0.5
    p_Bt_ratio: float = 1.0


@dataclass(frozen=True)
class TD2ReferenceGrid:
    """Actual lattice stored in the Yi TD2 ``bcor`` reference artifact.

    Yi et al. list 151 x 251 x 101 grid points.  The distributed ``bcor``
    artifact stores 150 x 250 x 100 field samples.  Its isotropic spacing is
    fixed by 150 samples spanning x=-180..180 Mm; y is centered on zero and z
    starts at the photosphere.
    """

    nx: int = 150
    ny: int = 250
    nz: int = 100
    x_min: float = -180.0
    x_max: float = 180.0

    @property
    def spacing(self) -> float:
        return (self.x_max - self.x_min) / float(self.nx - 1)

    def coordinates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = self.spacing
        x = np.linspace(self.x_min, self.x_max, self.nx, dtype=np.float64)
        y = (np.arange(self.ny, dtype=np.float64) - (self.ny - 1) / 2.0) * h
        z = np.arange(self.nz, dtype=np.float64) * h
        return x, y, z


YI_TD2_CONFIG = TDModelConfig(
    R=100.0,
    a=35.0,
    L=60.0,
    d=60.0,
    q=80.0,
    I0=-3.0,
    Nt=8.8,
)

YI_TD2_GRID = TD2ReferenceGrid()

# The factor 2500 is recovered exactly from the potential-field values on the
# outer faces.  The raw ``bcor`` artifact does not carry unit metadata, so keep
# this as a named legacy conversion rather than assigning an undocumented unit.
YI_TD2_FIELD_SCALE = 2500.0


# =============================================================================
# Carlson symmetric forms (ported from `mod_tdfluxrope.t`)
# =============================================================================
def _rf(x, y, z, *, errtol: float = 0.08):
    """
    Carlson symmetric form RF(x,y,z) (vectorized).

    This is a direct numerical-recipe style implementation matching the
    convergence/coefficients used in `mod_tdfluxrope.t`.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    # Constants from Fortran implementation
    third = 1.0 / 3.0
    c1 = 1.0 / 24.0
    c2 = 0.1
    c3 = 3.0 / 44.0
    c4 = 1.0 / 14.0
    tiny = 1.5e-38
    big = 3.0e37

    min_xyz = np.minimum(np.minimum(x, y), z)
    min_sum = np.minimum(np.minimum(x + y, x + z), y + z)
    max_xyz = np.maximum(np.maximum(x, y), z)
    invalid = (min_xyz < 0) | (min_sum < tiny) | (max_xyz > big)
    x = np.where(invalid, tiny, x)
    y = np.where(invalid, tiny, y)
    z = np.where(invalid, tiny, z)

    xt, yt, zt = x, y, z
    for _ in range(50):
        sqrtx = np.sqrt(xt)
        sqrty = np.sqrt(yt)
        sqrtz = np.sqrt(zt)
        alamb = sqrtx * (sqrty + sqrtz) + sqrty * sqrtz
        xt = 0.25 * (xt + alamb)
        yt = 0.25 * (yt + alamb)
        zt = 0.25 * (zt + alamb)

        ave = third * (xt + yt + zt)
        delx = (ave - xt) / ave
        dely = (ave - yt) / ave
        delz = (ave - zt) / ave

        if np.max(np.stack([np.abs(delx), np.abs(dely), np.abs(delz)], axis=0)) <= errtol:
            e2 = delx * dely - delz * delz
            e3 = delx * dely * delz
            out = (1.0 + (c1 * e2 - c2 - c3 * e3) * e2 + c4 * e3) / np.sqrt(ave)
            return np.where(invalid, np.nan, out)

    e2 = delx * dely - delz * delz
    e3 = delx * dely * delz
    out = (1.0 + (c1 * e2 - c2 - c3 * e3) * e2 + c4 * e3) / np.sqrt(ave)
    return np.where(invalid, np.nan, out)


def _rd(x, y, z, *, errtol: float = 0.05):
    """Carlson symmetric form RD(x,y,z) (vectorized), matching `mod_tdfluxrope.t`."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    c1 = 3.0 / 14.0
    c2 = 1.0 / 6.0
    c3 = 9.0 / 22.0
    c4 = 3.0 / 26.0
    c5 = 0.25 * c3
    c6 = 1.5 * c4
    tiny = 1.0e-25
    big = 4.5e21

    min_xy = np.minimum(x, y)
    min_xpyz = np.minimum(x + y, z)
    max_xyz = np.maximum(np.maximum(x, y), z)
    invalid = (min_xy < 0) | (min_xpyz < tiny) | (max_xyz > big)
    x = np.where(invalid, tiny, x)
    y = np.where(invalid, tiny, y)
    z = np.where(invalid, tiny, z)

    xt, yt, zt = x, y, z
    sum_ = 0.0
    fac = 1.0

    for _ in range(100):
        sqrtx = np.sqrt(xt)
        sqrty = np.sqrt(yt)
        sqrtz = np.sqrt(zt)
        alamb = sqrtx * (sqrty + sqrtz) + sqrty * sqrtz
        sum_ = sum_ + fac / (sqrtz * (zt + alamb))
        fac = 0.25 * fac
        xt = 0.25 * (xt + alamb)
        yt = 0.25 * (yt + alamb)
        zt = 0.25 * (zt + alamb)

        ave = 0.2 * (xt + yt + 3.0 * zt)
        delx = (ave - xt) / ave
        dely = (ave - yt) / ave
        delz = (ave - zt) / ave
        if np.max(np.stack([np.abs(delx), np.abs(dely), np.abs(delz)], axis=0)) <= errtol:
            ea = delx * dely
            eb = delz * delz
            ec = ea - eb
            ed = ea - 6.0 * eb
            ee = ed + ec + ec
            out = (
                3.0 * sum_
                + fac
                * (
                    1.0
                    + ed * (-c1 + c5 * ed - c6 * delz * ee)
                    + delz * (c2 * ee + delz * (-c3 * ec + delz * c4 * ea))
                )
                / (ave * np.sqrt(ave))
            )
            return np.where(invalid, np.nan, out)

    ea = delx * dely
    eb = delz * delz
    ec = ea - eb
    ed = ea - 6.0 * eb
    ee = ed + ec + ec
    out = (
        3.0 * sum_
        + fac
        * (
            1.0
            + ed * (-c1 + c5 * ed - c6 * delz * ee)
            + delz * (c2 * ee + delz * (-c3 * ec + delz * c4 * ea))
        )
        / (ave * np.sqrt(ave))
    )
    return np.where(invalid, np.nan, out)


def _validate_elliptic_inputs(invalid_mode: str, invalid_mask, *, func_name: str):
    invalid_mode = str(invalid_mode).lower()
    if invalid_mode not in {"nan", "raise"}:
        raise ValueError("invalid_mode must be 'nan' or 'raise'.")
    if invalid_mode == "raise" and np.any(invalid_mask):
        raise ValueError(f"{func_name}: invalid elliptic integral inputs encountered.")


def _ellf_complete(k, *, invalid_mode: str = "nan"):
    """Complete elliptic integral of the first kind: K(k) (ellf(pi/2,k))."""
    k = np.asarray(k, dtype=np.float64)
    m = k * k
    q = np.clip(1.0 - m, 0.0, None)
    invalid = q <= 0.0
    _validate_elliptic_inputs(invalid_mode, invalid, func_name="K(k)")
    return _rf(0.0, q, 1.0)


def _elle_complete(k, *, invalid_mode: str = "nan"):
    """Complete elliptic integral of the second kind: E(k) (elle(pi/2,k))."""
    k = np.asarray(k, dtype=np.float64)
    m = k * k
    q = np.clip(1.0 - m, 0.0, None)
    invalid = q <= 0.0
    _validate_elliptic_inputs(invalid_mode, invalid, func_name="E(k)")
    rf_val = _rf(0.0, q, 1.0)
    rd_val = _rd(0.0, q, 1.0)
    return rf_val - (m * rd_val) / 3.0


def _rc_scalar(x, y):
    """Carlson rc(x,y) (scalar; completeness only, not used in K/E here)."""
    third = 1.0 / 3.0
    errtol = 0.04
    tiny = 1.69e-38
    sqrtny = 1.3e-19
    big = 3.0e37
    tnby = tiny * big
    comp1 = 2.236 / sqrtny
    comp2 = tnby * tnby / 25.0
    c1 = 0.3
    c2 = 1.0 / 7.0
    c3 = 0.375
    c4 = 9.0 / 22.0

    if (
        x < 0.0
        or y == 0.0
        or (x + abs(y)) < tiny
        or (x + abs(y)) > big
        or (y < -comp1 and x > 0.0 and x < comp2)
    ):
        return np.nan

    if y > 0.0:
        xt = x
        yt = y
        w = 1.0
    else:
        xt = x - y
        yt = -y
        w = np.sqrt(x) / np.sqrt(xt)

    while True:
        alamb = 2.0 * np.sqrt(xt) * np.sqrt(yt) + yt
        xt = 0.25 * (xt + alamb)
        yt = 0.25 * (yt + alamb)
        ave = third * (xt + yt + yt)
        s = (yt - ave) / ave
        if abs(s) <= errtol:
            break

    return w * (1.0 + s * s * (c1 + s * (c2 + s * (c3 + s * c4)))) / np.sqrt(ave)


def _rj_scalar(x, y, z, p):
    """Carlson rj(x,y,z,p) (scalar; completeness only, not used in K/E here)."""
    errtol = 0.05
    tiny = 2.5e-13
    big = 9.0e11
    c1 = 3.0 / 14.0
    c2 = 1.0 / 3.0
    c3 = 3.0 / 22.0
    c4 = 3.0 / 26.0
    c5 = 0.75 * c3
    c6 = 1.5 * c4
    c7 = 0.5 * c2
    c8 = c3 + c3

    def rc(a, b):
        return _rc_scalar(a, b)

    if (
        min(x, y, z) < 0.0
        or min(x + y, x + z, y + z, abs(p)) < tiny
        or max(x, y, z, abs(p)) > big
    ):
        return np.nan

    sum_ = 0.0
    fac = 1.0
    if p > 0.0:
        xt, yt, zt, pt = x, y, z, p
    else:
        xt = min(x, y, z)
        zt = max(x, y, z)
        yt = x + y + z - xt - zt
        a = 1.0 / (yt - p)
        b = a * (zt - yt) * (yt - xt)
        pt = yt + b
        rho = xt * zt / yt
        tau = p * pt / yt
        rcx = rc(rho, tau)

    while True:
        sqrtx = np.sqrt(xt)
        sqrty = np.sqrt(yt)
        sqrtz = np.sqrt(zt)
        alamb = sqrtx * (sqrty + sqrtz) + sqrty * sqrtz
        alpha = (pt * (sqrtx + sqrty + sqrtz) + sqrtx * sqrty * sqrtz) ** 2
        beta = pt * (pt + alamb) ** 2
        sum_ = sum_ + fac * rc(alpha, beta)
        fac = 0.25 * fac
        xt = 0.25 * (xt + alamb)
        yt = 0.25 * (yt + alamb)
        zt = 0.25 * (zt + alamb)
        pt = 0.25 * (pt + alamb)
        ave = 0.2 * (xt + yt + zt + pt + pt)
        delx = (ave - xt) / ave
        dely = (ave - yt) / ave
        delz = (ave - zt) / ave
        delp = (ave - pt) / ave
        if max(abs(delx), abs(dely), abs(delz), abs(delp)) <= errtol:
            break

    ea = delx * (dely + delz) + dely * delz
    eb = delx * dely * delz
    ec = delp**2
    ed = ea - 3.0 * ec
    ee = eb + 2.0 * delp * (ea - ec)
    out = (
        3.0 * sum_
        + fac
        * (
            1.0
            + ed * (-c1 + c5 * ed - c6 * ee)
            + eb * (c7 + delp * (-c8 + delp * c4))
            + delp * ea * (c2 - delp * c3)
            - c2 * delp * ec
        )
        / (ave * np.sqrt(ave))
    )
    if p <= 0.0:
        rho = xt * zt / yt
        tau = p * pt / yt
        rcx = rc(rho, tau)
        out = a * (b * out + 3.0 * (rcx - _rf(xt, yt, zt)))
    return out

def generate_td_model(
    x_grid,
    y_grid,
    z_grid,
    *,
    d: float = 50.0,
    L: float = 50.0,
    q: float = 100.0,
    R: float = 85.0,
    a: float = 31.0,
    I_0: float = -7.0,
    mu_0: float = 4 * np.pi * 1e-1,
    l_i: float = 0.5,
    p_Bt_ratio: float = 1.0,
    elliptic_invalid: str = "nan",
    k_clip_eps: float = 1e-12,
    return_components: bool = False,
):
    """
    Titov–Démoulin (TDm) analytic magnetic field on a Cartesian grid.

    Notes:
    - Units must be consistent across x/y/z and (d, L, R, a).
    - `q` and `I_0` are in the same internal units as the original implementation.
      In this repository they are often interpreted as q in (1e12 Wb) and I_0 in (1e12 A),
      i.e., "TA" for I_0.
    - `elliptic_invalid="raise"` makes elliptic-integral singularities fail fast (closer to
      the Fortran behavior); the default `"nan"` preserves the previous Python behavior.
    """

    # Toroidal ring current I (legacy / paper-style with explicit mu_0 scaling)
    I = (8 * np.pi * q * L * R * (R**2 + L**2) ** (-1.5) / mu_0) / (
        np.log(8 * R / a) - 1.5 + l_i / 2
    )

    # Flatten 3D grids for vectorized operations
    x = x_grid.flatten()
    y = y_grid.flatten()
    z = z_grid.flatten()
    
    # --- common geometry / unit vectors ---
    r_perp = np.sqrt(y**2 + (z + d)**2)
    rho = np.sqrt(x**2 + (r_perp - R)**2)
    
    # theta_hat components (x, y, z)
    theta_hat_x = np.zeros_like(r_perp)
    theta_hat_y = -(z + d) / r_perp
    theta_hat_z = y / r_perp

    # Heaviside step (inside tube=1, outside=0)
    chi = np.where(a - rho > 0, 1.0, 0.0)

    # --- component 1: potential field from two point charges (B_q) ---
    r_plus_x, r_plus_y, r_plus_z = x - L, y, z + d
    r_minus_x, r_minus_y, r_minus_z = x + L, y, z + d
    
    dist_plus3 = (r_plus_x**2 + r_plus_y**2 + r_plus_z**2)**1.5
    dist_minus3 = (r_minus_x**2 + r_minus_y**2 + r_minus_z**2)**1.5

    B_q_x = q * (r_plus_x / dist_plus3 - r_minus_x / dist_minus3)
    B_q_y = q * (r_plus_y / dist_plus3 - r_minus_y / dist_minus3)
    B_q_z = q * (r_plus_z / dist_plus3 - r_minus_z / dist_minus3)

    # --- component 2: toroidal field from axial line current (B_theta) ---
    term3 = 1 / r_perp - 1 / R

    term1 = 1 / R**2
    tube_factor = np.maximum(0.0, 1.0 - rho**2 / a**2) * float(p_Bt_ratio)
    term2 = 2 * chi / a**2 * (I**2 / I_0**2) * tube_factor

    # Guard against negative values due to numerical issues
    inner_sqrt = np.maximum(term1 + term2, 0)
    B_theta_mag = (mu_0 * I_0 / (2 * np.pi)) * (np.sqrt(inner_sqrt) + term3)
    
    B_theta_x = B_theta_mag * theta_hat_x
    B_theta_y = B_theta_mag * theta_hat_y
    B_theta_z = B_theta_mag * theta_hat_z

    # --- component 3: poloidal field from ring current (B_I) ---
    # Port of `TD99_BI` from `mod_tdfluxrope.t`:
    # - exact (chain-rule) derivatives
    # - Carlson symmetric forms for complete elliptic integrals
    # - removes numerical central differences from the poloidal field path
    k_eps = float(k_clip_eps)
    if k_eps < 0.0:
        raise ValueError("k_clip_eps must be >= 0.")
    tiny = 1e-30

    r_safe = np.where(r_perp == 0.0, tiny, r_perp)

    # Eq. 27: k
    k = 2.0 * np.sqrt(r_perp * R) / np.sqrt((r_perp + R) ** 2 + x**2)
    if k_eps > 0.0:
        k = np.clip(k, 0.0, 1.0 - k_eps)

    # dk/dx (Fortran: dkdx)
    dkdx = -x * k / (((r_perp + R) ** 2) + x**2)

    # dk/dr_perp (Fortran: dkdrv)
    dkdrv = k * (R**2 + x**2 - r_perp**2) / (2.0 * r_safe * (((R + r_perp) ** 2) + x**2))

    # Eq. 30: ka (inside-wave matching)
    ka = 2.0 * np.sqrt(r_perp * R) / np.sqrt(4.0 * r_perp * R + a**2)
    if k_eps > 0.0:
        ka = np.clip(ka, 0.0, 1.0 - k_eps)

    # dk_adx = 0 (Fortran: dkadx)
    # dk_ad/dr_perp (Fortran: dkadrv)
    dkadrv = ka * (a**2) / (2.0 * r_safe * (a**2 + 4.0 * R * r_perp))

    # ---- A_k(k) and derivatives (Fortran: Ak, dAkdk) ----
    k2 = k * k
    one_minus_k2 = np.maximum(1.0 - k2, max(k_eps, 0.0))
    k_safe = np.where(k == 0.0, tiny, k)

    Kk = _ellf_complete(k, invalid_mode=elliptic_invalid)
    Ek = _elle_complete(k, invalid_mode=elliptic_invalid)

    Ak = ((2.0 - k2) * Kk - 2.0 * Ek) / k_safe

    dAkdk = ((2.0 - k2) * Ek - 2.0 * one_minus_k2 * Kk) / (
        np.maximum(k2, max(k_eps, 0.0)) * one_minus_k2
    )
    # For k -> 0 the numerator tends to 0 faster than the denominator; enforce 0 to avoid tiny/0 artifacts.
    dAkdk = np.where(k2 < max(k_eps, 1e-30), 0.0, dAkdk)

    # ---- A_k(ka) and derivatives (Fortran: Aka, dAkdkka, d2Akdk2ka) ----
    ka2 = ka * ka
    one_minus_ka2 = np.maximum(1.0 - ka2, max(k_eps, 0.0))
    ka_safe = np.where(ka == 0.0, tiny, ka)

    Kka = _ellf_complete(ka, invalid_mode=elliptic_invalid)
    Eka = _elle_complete(ka, invalid_mode=elliptic_invalid)

    Aka = ((2.0 - ka2) * Kka - 2.0 * Eka) / ka_safe

    dAkdkka = ((2.0 - ka2) * Eka - 2.0 * one_minus_ka2 * Kka) / (
        np.maximum(ka2, max(k_eps, 0.0)) * one_minus_ka2
    )
    # For ka -> 0 the derivative terms should vanish; enforce 0 to avoid numerical artifacts.
    dAkdkka = np.where(ka2 < max(k_eps, 1e-30), 0.0, dAkdkka)

    ka4 = ka2 * ka2
    d2Akdk2ka = (
        (
            (7.0 * ka2 - 4.0 - ka4) * (Eka / one_minus_ka2)
            + (4.0 - 5.0 * ka2) * Kka
        )
        / (ka_safe**3 * one_minus_ka2)
    )
    d2Akdk2ka = np.where(ka2 < max(k_eps, 1e-30), 0.0, d2Akdk2ka)

    # (mu0*I/(2*pi))*sqrt(R/r)
    prefactor = (mu_0 * I / (2.0 * np.pi)) * np.sqrt(R / r_safe)
    bi_fac = 1.0

    AIex = prefactor * Ak
    AIinwave = prefactor * (Aka + dAkdkka * (k - ka))

    inside_mask = chi  # 1.0 inside tube, 0.0 outside
    A_I = inside_mask * AIinwave + (1.0 - inside_mask) * AIex

    # Partial derivatives (Fortran: dAIdx / dAIdrv)
    dAIexdx = prefactor * (dAkdk * dkdx)
    dAIexdrv = prefactor * (dAkdk * dkdrv) - AIex * 0.5 / r_safe

    dAIinwavedx = prefactor * (dAkdkka * dkdx)  # dkadx = 0
    dAIinwavedrv = (
        prefactor * (dAkdkka * dkdrv + d2Akdk2ka * dkadrv * (k - ka))
        - AIinwave * 0.5 / r_safe
    )

    dAIdx = inside_mask * dAIinwavedx + (1.0 - inside_mask) * dAIexdx
    dAIdrv = inside_mask * dAIinwavedrv + (1.0 - inside_mask) * dAIexdrv

    # Poloidal field projection (matching Fortran's unit-vector projection)
    B_I_x = bi_fac * (dAIdrv + A_I / r_safe)
    B_I_perp = bi_fac * (-dAIdx)

    # Convert B_I to Cartesian components
    B_I_y = B_I_perp * (y / r_safe)
    B_I_z = B_I_perp * ((z + d) / r_safe)

    # --- final superposition ---
    B_total_x = B_q_x + B_theta_x + B_I_x
    B_total_y = B_q_y + B_theta_y + B_I_y
    B_total_z = B_q_z + B_theta_z + B_I_z

    # Restore original grid shape
    Bx = B_total_x.reshape(x_grid.shape)
    By = B_total_y.reshape(y_grid.shape)
    Bz = B_total_z.reshape(z_grid.shape)

    if not return_components:
        return Bx, By, Bz

    comp = {
        "Bx": Bx,
        "By": By,
        "Bz": Bz,
        "Bx_q": B_q_x.reshape(x_grid.shape),
        "By_q": B_q_y.reshape(y_grid.shape),
        "Bz_q": B_q_z.reshape(z_grid.shape),
        "Bx_theta": B_theta_x.reshape(x_grid.shape),
        "By_theta": B_theta_y.reshape(y_grid.shape),
        "Bz_theta": B_theta_z.reshape(z_grid.shape),
        "Bx_I": B_I_x.reshape(x_grid.shape),
        "By_I": B_I_y.reshape(y_grid.shape),
        "Bz_I": B_I_z.reshape(z_grid.shape),
        "I_ring": float(I),
    }
    return comp


def yi_td2_reference_path() -> Path:
    """Return the repository's authoritative Yi TD2 reference artifact."""

    return Path(__file__).resolve().parent / "data" / "analytic" / "td2" / "bcor"


def _validate_yi_td2_config(config: TDModelConfig, *, atol: float = 1e-12) -> None:
    """Require the exact paper configuration used by the frozen TD2 field."""

    mismatches = []
    for item in fields(TDModelConfig):
        name = item.name
        actual = float(getattr(config, name))
        expected = float(getattr(YI_TD2_CONFIG, name))
        if not np.isclose(actual, expected, rtol=0.0, atol=atol):
            mismatches.append(f"{name}={actual:g} (expected {expected:g})")
    if mismatches:
        detail = ", ".join(mismatches)
        raise ValueError(
            "The frozen Yi TD2 reference is defined only for the published "
            f"TD2 configuration; mismatched values: {detail}. "
            "Use backend='analytic' for a different TD configuration."
        )


def _load_td2_bcor_views(
    path: Path,
    grid: TD2ReferenceGrid,
    *,
    mmap_mode: Literal["r", "r+"] | None = "r",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load component views from the Fortran-order TD2 binary artifact."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Yi TD2 reference field not found: {path}")

    npt = grid.nx * grid.ny * grid.nz
    expected_bytes = 3 * npt * np.dtype(np.float64).itemsize
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Yi TD2 reference size mismatch: got {actual_bytes} bytes, "
            f"expected {expected_bytes}."
        )

    if mmap_mode is None:
        raw = np.fromfile(path, dtype=np.float64)
    else:
        raw = np.memmap(path, dtype=np.float64, mode=mmap_mode, shape=(3 * npt,))

    shape = (grid.nx, grid.ny, grid.nz)
    bx = raw[:npt].reshape(shape, order="F")
    by = raw[npt : 2 * npt].reshape(shape, order="F")
    bz = raw[2 * npt :].reshape(shape, order="F")
    return bx, by, bz


def _generate_yi_td2_analytic_compat(
    config: TDModelConfig,
    grid: TD2ReferenceGrid,
    *,
    chunk_depth: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the archived TD99 formula using the Yi TD2 conventions.

    This path is useful for formula studies.  It reproduces the paper current
    sign, the actual isotropic lattice, the legacy field unit and the five
    non-photospheric potential-field faces found in ``bcor``.  It is not the
    frozen reference itself: the distributed artifact's interior torus-current
    values are not reproduced by the archived continuous-formula port.
    """

    chunk_depth = int(chunk_depth)
    if chunk_depth < 1:
        raise ValueError("chunk_depth must be >= 1.")

    x, y, z = grid.coordinates()
    bx = np.empty((grid.nx, grid.ny, grid.nz), dtype=np.float64)
    by = np.empty_like(bx)
    bz = np.empty_like(bx)

    # The archived implementation's positive internal I_0 produces the field
    # orientation printed as negative I0 in Yi et al. Table 1.
    internal_i0 = -float(config.I0)
    scale = float(YI_TD2_FIELD_SCALE)

    for k0 in range(0, grid.nz, chunk_depth):
        k1 = min(k0 + chunk_depth, grid.nz)
        xx, yy, zz = np.meshgrid(x, y, z[k0:k1], indexing="ij")
        comp = generate_td_model(
            xx,
            yy,
            zz,
            d=config.d,
            L=config.L,
            q=config.q,
            R=config.R,
            a=config.a,
            I_0=internal_i0,
            l_i=config.l_i,
            p_Bt_ratio=config.p_Bt_ratio,
            return_components=True,
        )

        total = [comp["Bx"], comp["By"], comp["Bz"]]
        potential = [
            comp["Bx_q"] + comp["Bx_theta"],
            comp["By_q"] + comp["By_theta"],
            comp["Bz_q"] + comp["Bz_theta"],
        ]

        # The frozen artifact contains no torus-current contribution on x/y
        # side faces or on the top face.  The photospheric face remains the
        # full TD field.
        for values, outer in zip(total, potential):
            values[[0, -1], :, :] = outer[[0, -1], :, :]
            values[:, [0, -1], :] = outer[:, [0, -1], :]
            if k1 == grid.nz:
                values[:, :, -1] = outer[:, :, -1]

        bx[:, :, k0:k1] = scale * total[0]
        by[:, :, k0:k1] = scale * total[1]
        bz[:, :, k0:k1] = scale * total[2]

    return bx, by, bz


def generate_yi_td2_reference(
    config: TDModelConfig = YI_TD2_CONFIG,
    *,
    backend: Literal["frozen", "analytic"] = "frozen",
    reference_path: str | Path | None = None,
    mmap_mode: Literal["r", "r+"] | None = "r",
    chunk_depth: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a Yi TD2 field in the current reference layout and units.

    ``backend='frozen'`` is the reproducibility path and returns zero-copy
    views of the authoritative ``bcor`` artifact.  It accepts only the exact
    TD2 configuration from Yi et al.  ``backend='analytic'`` evaluates the
    archived TD99 formula with all recovered TD2 conventions, and may be used
    with other parameter values, but it must not be presented as byte-identical
    to the frozen reference.
    """

    backend = str(backend).lower()
    if backend == "frozen":
        _validate_yi_td2_config(config)
        path = yi_td2_reference_path() if reference_path is None else Path(reference_path)
        return _load_td2_bcor_views(path, YI_TD2_GRID, mmap_mode=mmap_mode)
    if backend == "analytic":
        if reference_path is not None:
            raise ValueError("reference_path is only valid with backend='frozen'.")
        return _generate_yi_td2_analytic_compat(
            config,
            YI_TD2_GRID,
            chunk_depth=chunk_depth,
        )
    raise ValueError("backend must be 'frozen' or 'analytic'.")


def _vti_dataarray_ascii(name, data_flat, n_components=1, dtype="Float32"):
    if n_components == 1:
        header = f'<DataArray type="{dtype}" Name="{name}" format="ascii">'
        body = " ".join(f"{v:.8e}" for v in data_flat)
        return f"{header}\n{body}\n</DataArray>"
    header = f'<DataArray type="{dtype}" Name="{name}" NumberOfComponents="{n_components}" format="ascii">'
    body = " ".join(f"{v:.8e}" for v in data_flat)
    return f"{header}\n{body}\n</DataArray>"


def save_vti_image_data(path, Bx, By, Bz, x_range, y_range, z_range, vector_name="B"):
    """
    Save a 3D vector field as VTK XML ImageData (.vti), in ASCII format.

    VTK expects x-index (i) to vary fastest in the flattened buffer. For arrays shaped
    (nx, ny, nz) indexed as [i, j, k], use Fortran-order flattening (order='F').
    """
    Bx = np.asarray(Bx)
    By = np.asarray(By)
    Bz = np.asarray(Bz)
    if Bx.shape != By.shape or Bx.shape != Bz.shape or Bx.ndim != 3:
        raise ValueError("Bx/By/Bz must be 3D arrays with identical shapes (nx, ny, nz).")

    nx, ny, nz = Bx.shape
    if len(x_range) != nx or len(y_range) != ny or len(z_range) != nz:
        raise ValueError("x_range/y_range/z_range lengths must match B field shape (nx, ny, nz).")

    # Origin/spacing from coordinate vectors (assume uniform spacing)
    origin = (float(x_range[0]), float(y_range[0]), float(z_range[0]))
    dx = float(x_range[1] - x_range[0]) if nx > 1 else 1.0
    dy = float(y_range[1] - y_range[0]) if ny > 1 else 1.0
    dz = float(z_range[1] - z_range[0]) if nz > 1 else 1.0
    spacing = (dx, dy, dz)

    # Flatten with VTK-compatible ordering
    bx_f = Bx.astype(np.float32, copy=False).ravel(order="F")
    by_f = By.astype(np.float32, copy=False).ravel(order="F")
    bz_f = Bz.astype(np.float32, copy=False).ravel(order="F")

    # Interleave components per point: (Bx0, By0, Bz0, Bx1, By1, Bz1, ...)
    b_interleaved = np.empty((bx_f.size * 3,), dtype=np.float32)
    b_interleaved[0::3] = bx_f
    b_interleaved[1::3] = by_f
    b_interleaved[2::3] = bz_f

    whole_extent = f"0 {nx-1} 0 {ny-1} 0 {nz-1}"

    xml = []
    xml.append('<?xml version="1.0"?>')
    xml.append('<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">')
    xml.append(
        f'  <ImageData WholeExtent="{whole_extent}" Origin="{origin[0]} {origin[1]} {origin[2]}" '
        f'Spacing="{spacing[0]} {spacing[1]} {spacing[2]}">'
    )
    xml.append(f'    <Piece Extent="{whole_extent}">')
    xml.append("      <PointData>")
    # Save both the 3-component vector and its individual components
    xml.append("        " + _vti_dataarray_ascii(vector_name, b_interleaved, n_components=3, dtype="Float32").replace("\n", "\n        "))
    xml.append("        " + _vti_dataarray_ascii("Bx", bx_f, n_components=1, dtype="Float32").replace("\n", "\n        "))
    xml.append("        " + _vti_dataarray_ascii("By", by_f, n_components=1, dtype="Float32").replace("\n", "\n        "))
    xml.append("        " + _vti_dataarray_ascii("Bz", bz_f, n_components=1, dtype="Float32").replace("\n", "\n        "))
    xml.append("      </PointData>")
    xml.append("      <CellData/>")
    xml.append("    </Piece>")
    xml.append("  </ImageData>")
    xml.append("</VTKFile>")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(xml))


def generate_td_model_vti(
    out_path,
    x_range,
    y_range,
    z_range,
    vector_name="B",
):
    X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing="ij")
    Bx, By, Bz = generate_td_model(X, Y, Z)
    save_vti_image_data(out_path, Bx, By, Bz, x_range, y_range, z_range, vector_name=vector_name)
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Titov-Demoulin analytic NLFFF field and save as .vti.")
    parser.add_argument("--out", type=str, required=True, help="Output .vti path")
    parser.add_argument("--nx", type=int, default=64, help="Number of grid points in x")
    parser.add_argument("--ny", type=int, default=64, help="Number of grid points in y")
    parser.add_argument("--nz", type=int, default=64, help="Number of grid points in z (z>=0)")
    parser.add_argument("--x0", type=float, default=-100.0, help="x min")
    parser.add_argument("--x1", type=float, default=100.0, help="x max")
    parser.add_argument("--y0", type=float, default=-100.0, help="y min")
    parser.add_argument("--y1", type=float, default=100.0, help="y max")
    parser.add_argument("--z0", type=float, default=0.0, help="z min (should be >=0)")
    parser.add_argument("--z1", type=float, default=100.0, help="z max")
    parser.add_argument("--name", type=str, default="B", help="Vector field name in VTK")
    args = parser.parse_args()

    x_range = np.linspace(args.x0, args.x1, args.nx, dtype=np.float64)
    y_range = np.linspace(args.y0, args.y1, args.ny, dtype=np.float64)
    z_range = np.linspace(args.z0, args.z1, args.nz, dtype=np.float64)

    out_path = generate_td_model_vti(args.out, x_range, y_range, z_range, vector_name=args.name)
    print(f"Saved VTI: {out_path}")
