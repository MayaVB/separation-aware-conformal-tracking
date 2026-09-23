"""
CP-aware fusion weight for CPWeightedFusionTracker.

Philosophy
----------
Restores the fusion principle from the earlier 1-D "Conformal
Prediction-Aware DOA Tracking" paper: a scalar weight w_t,k, derived purely
from scalar descriptors of speaker k's CP/LCP region at frame t, controls
how much the tracker trusts the current measurement vs. the propagated
motion prediction. w does NOT touch the motion transition itself (see
cp_weighted_tracker.py) -- only the belief interpolation.

    w_t,k = exp[-(lambda_var * var_norm + lambda_size * size_norm + lambda_span * span_norm)]

    compact / confident region -> descriptors near 0 -> w near 1 -> trust measurement
    large / irregular region   -> descriptors near 1 -> w near 0 -> trust motion prediction

This function only consumes the scalar dict returned by
cp_features.extract_cp_features(cp_region=...) -- it has no notion of
whether that region came from global CP or LCP, so it is generic across
both by construction.

lambda_var/lambda_size/lambda_span are NEW, independently tunable
parameters for this 2-D tracker. They are initialized at the same starting
values as the 1-D paper's lambda_sp/lambda_sz/lambda_span (1.0, 2.0, 1.0)
but do not need to numerically match it -- the descriptors themselves are
different (new 2-D definitions, normalized to this grid). Not tuned here;
inspect on calibration data before relying on them.
"""

import numpy as np


def compute_cp_weight(cp_features, grid_shape,
                       lambda_var=1.0, lambda_size=2.0, lambda_span=1.0):
    """Compute the scalar CP-aware fusion weight for one speaker's region.

    Parameters
    ----------
    cp_features : dict
        Output of cp_features.extract_cp_features for this speaker's CP
        region at this frame. Must contain "cp_area_norm", "cp_width_total",
        "cp_var".
    grid_shape : tuple of int
        (nele, nazi) -- shape of the likelihood / CP grid.
    lambda_var, lambda_size, lambda_span : float
        Weight-formula coefficients. Defaults (1.0, 2.0, 1.0) are starting
        values only -- see module docstring.

    Returns
    -------
    w : float, in (0, 1]
        Fusion weight: w=1 -> fully trust the measurement, w=0 -> fully
        trust the motion prediction.
    descriptors : dict
        {"size_norm": float, "span_norm": float, "var_norm": float}, each
        in [0, 1] after the nonfinite/clip handling below -- returned for
        inspection/debugging (see cp_weighted_tracker.py's `debug` output).
    """
    nele, nazi = int(grid_shape[0]), int(grid_shape[1])
    diag2 = float((nele - 1) ** 2 + (nazi - 1) ** 2)
    diag = float(np.sqrt(diag2))

    size_raw = cp_features.get("cp_area_norm", np.nan)          # already in [0, 1]
    span_raw = cp_features.get("cp_width_total", np.nan) / diag  # rescaled
    var_raw = cp_features.get("cp_var", np.nan) / diag2          # rescaled

    # Nonfinite (e.g. empty CP region) -> maximal uncertainty, not a crash.
    # Otherwise clip to [0, 1] -- span_norm in particular can slightly exceed
    # 1.0 at the extreme because cp_width_total's "+1 inclusive cell count"
    # convention maxes out a bit above `diag`'s index-range convention; the
    # clip absorbs that rather than redefining either existing feature.
    size_norm, span_norm, var_norm = [
        1.0 if not np.isfinite(x) else float(np.clip(x, 0.0, 1.0))
        for x in (size_raw, span_raw, var_raw)
    ]

    w = float(np.exp(-(
        lambda_var * var_norm + lambda_size * size_norm + lambda_span * span_norm
    )))

    return w, {"size_norm": size_norm, "span_norm": span_norm, "var_norm": var_norm}
