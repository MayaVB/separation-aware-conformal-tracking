"""
Mondrian-hatD separation-aware conformal prediction (the CP half of the main method).

Deployable: every step uses only the two IDL *estimated* DOAs, never ground truth.

    1. hatD  = great-circle angle between the two estimated DOAs of a frame
    2. M = 5 approximately equal-frequency groups of the CALIBRATION hatD values
       (tie-safe: every cut lies strictly between two distinct observed values)
    3. lambda[m, k] per group m and IDL slot k, with the same finite-sample conformal
       rule as Global CP (lcp.calibrate_global_lambda_from_arrays on the group's frames)
    4. at inference: hatD -> group m -> each slot's region grown with lambda[m, k]

The resulting regions are what CPWeightedFusionTracker.for_mondrian() consumes:

    calib = calibrate_mondrian(lm_c, est_c, true_c, room, lambda_list, alpha)
    regions, groups = build_mondrian_regions(lm_t, est_t, calib, nele, nazi)
    result = CPWeightedFusionTracker.for_mondrian().run(lm_t, regions, est_grid_t)

Calibration and region growing use the external CP framework (Code.crc_ssl via lcp.py /
npz_adapter.py), so this module is not imported by the package __init__.
Functions 1-2 are exact copies of the versions used for the paper results
(analysis/mondrian_separation: eval_deployable_mondrian_separation.delta_hat_deg_batch,
eval_mondrian_hatD_equal_freq.compute_equal_freq_boundaries / assign_bins).
"""

import numpy as np

from Code.two_speaker_tracking.lcp import calibrate_global_lambda_from_arrays
from Code.two_speaker_tracking.npz_adapter import build_cp_regions_for_frames

M_DEFAULT = 5


def estimated_separation_deg(est_rad):
    """hatD in degrees. est_rad: (n, 2, 2) estimated [ele (polar), azi] radians of the
    two IDL slots. Uses no ground truth."""
    th0, az0 = est_rad[:, 0, 0], est_rad[:, 0, 1]
    th1, az1 = est_rad[:, 1, 0], est_rad[:, 1, 1]
    cosD = np.cos(th0) * np.cos(th1) + np.sin(th0) * np.sin(th1) * np.cos(az0 - az1)
    return np.degrees(np.arccos(np.clip(cosD, -1.0, 1.0)))


def equal_freq_boundaries(hatD_calib, M=M_DEFAULT):
    """Equal-frequency group boundaries from CALIBRATION hatD only. Raw cuts are
    np.quantile(hatD, i/M, method="linear"), each snapped to the midpoint of the gap
    between the nearest distinct observed values, so ties are never split. Returns a
    sorted array of length <= M-1 (shorter only if groups collapse)."""
    hatD_calib = np.asarray(hatD_calib, dtype=float)
    unique_vals = np.unique(hatD_calib)
    if len(unique_vals) < 2:
        return np.array([])
    boundaries = []
    for i in range(1, M):
        raw_q = np.quantile(hatD_calib, i / M, method="linear")
        j = int(np.clip(np.searchsorted(unique_vals, raw_q, side="left"), 1, len(unique_vals) - 1))
        boundaries.append((unique_vals[j - 1] + unique_vals[j]) / 2.0)
    return np.unique(np.array(boundaries, dtype=float))


def assign_groups(hatD, boundaries):
    """Group index 0..len(boundaries) for each hatD value (same frozen boundaries for
    calibration and test frames)."""
    if len(boundaries) == 0:
        return np.zeros(len(hatD), dtype=int)
    return np.searchsorted(boundaries, hatD, side="right")


def calibrate_mondrian(likelihood_maps, est_rad, true_rad, room, lambda_list, alpha, M=M_DEFAULT):
    """Calibrate Mondrian-hatD CP on calibration frames.

    Parameters
    ----------
    likelihood_maps : (n, 2, nele, nazi) raw maps; est_rad, true_rad : (n, 2, 2) radians
    room : grid object (npz_adapter._build_room); lambda_list : threshold grid; alpha : miscoverage

    Returns
    -------
    dict(boundaries (M-1,), lambdas (M, 2) = lambda[m, k], n_calib (M,))

    Raises ValueError if groups collapse or a group cannot be calibrated -- a tracker
    needs a region for every frame, so there is no silent fallback to Global CP.
    """
    hatD = estimated_separation_deg(np.asarray(est_rad, dtype=float))
    boundaries = equal_freq_boundaries(hatD, M)
    if len(boundaries) != M - 1:
        raise ValueError(f"Mondrian groups collapsed: {len(boundaries) + 1} of {M} realized")
    groups = assign_groups(hatD, boundaries)
    lambdas = np.zeros((M, 2))
    for m in range(M):
        mask = groups == m
        if not mask.any():
            raise ValueError(f"no calibration frames in hatD group {m}")
        lambdas[m] = calibrate_global_lambda_from_arrays(
            likelihood_maps[mask], est_rad[mask], true_rad[mask], room, lambda_list, alpha)
    return dict(boundaries=boundaries, lambdas=lambdas, n_calib=np.bincount(groups, minlength=M))


def build_mondrian_regions(likelihood_maps, est_rad, calib, nele, nazi):
    """Mondrian CP regions for evaluation frames (IDL-slot order).

    Returns (regions (T, 2, nele, nazi) bool, groups (T,) int)."""
    est_rad = np.asarray(est_rad, dtype=float)
    groups = assign_groups(estimated_separation_deg(est_rad), calib["boundaries"])
    regions = np.zeros((len(est_rad), 2, nele, nazi), dtype=bool)
    for t in range(len(est_rad)):
        regions[t] = build_cp_regions_for_frames(likelihood_maps[t:t + 1], est_rad[t:t + 1],
                                                 calib["lambdas"][groups[t]], nele, nazi)[0]
    return regions, groups
