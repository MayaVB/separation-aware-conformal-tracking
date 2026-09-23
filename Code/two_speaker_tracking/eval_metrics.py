"""
Shared per-frame localization-error metrics.

Extracted from eval_heatmap_noburst.py so eval_tracker_conditions.py (and any
other script) can reuse the same Hungarian-matched azimuth/elevation error
logic instead of duplicating it. No behavior change from the original
module-level functions — only the implicit dependency on module-global `K`
was made an explicit parameter (derived from the input arrays' own length).
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def wrap_azi_err_deg(a_rad, b_rad):
    """Wrapped azimuth error between two values in radians -> degrees."""
    diff = np.abs(((a_rad - b_rad + np.pi) % (2 * np.pi)) - np.pi)
    return np.degrees(diff)


def argmax_to_angles(lm_frame, ele_grid, azi_grid):
    """
    lm_frame : (K, nele, nazi)
    ele_grid : (nele,) radians
    azi_grid : (nazi,) radians
    Returns est_ele, est_azi each shape (K,) in radians.
    """
    K = lm_frame.shape[0]
    nele, nazi = len(ele_grid), len(azi_grid)
    est_ele = np.zeros(K)
    est_azi = np.zeros(K)
    for k in range(K):
        flat_idx = np.argmax(lm_frame[k])
        ei, ai = np.unravel_index(flat_idx, (nele, nazi))
        est_ele[k] = ele_grid[ei]
        est_azi[k] = azi_grid[ai]
    return est_ele, est_azi


def hungarian_errors(est_ele, est_azi, gt_ele, gt_azi, threshold_deg=30.0):
    """
    Hungarian-matched errors for one frame.

    Returns lists (length K) of:
        azi_errs, ele_errs, matched_flags
    and per-source MDR / FAR flags:
        gt_unmatched (K,) bool, est_unmatched (K,) bool
    """
    K = len(est_azi)
    cost = np.array([[wrap_azi_err_deg(est_azi[e], gt_azi[g])
                      for g in range(K)] for e in range(K)])
    est_idx, gt_idx = linear_sum_assignment(cost)

    azi_errs, ele_errs, matched_flags = [], [], []
    gt_matched  = np.zeros(K, dtype=bool)
    est_matched = np.zeros(K, dtype=bool)

    for e, g in zip(est_idx, gt_idx):
        ae = wrap_azi_err_deg(est_azi[e], gt_azi[g])
        ee = np.degrees(np.abs(est_ele[e] - gt_ele[g]))
        m  = ae < threshold_deg
        azi_errs.append(ae)
        ele_errs.append(ee)
        matched_flags.append(m)
        if m:
            gt_matched[g]  = True
            est_matched[e] = True

    gt_unmatched  = ~gt_matched
    est_unmatched = ~est_matched
    return azi_errs, ele_errs, matched_flags, gt_unmatched, est_unmatched


def compute_metrics(azi_list, ele_list, matched_list, gt_unmatch_list, est_unmatch_list):
    azi   = np.array(azi_list)
    ele   = np.array(ele_list)
    mat   = np.array(matched_list, dtype=float)
    gt_u  = np.array(gt_unmatch_list, dtype=float)
    est_u = np.array(est_unmatch_list, dtype=float)

    return dict(
        MAE_azi  = np.mean(azi),
        MAE_ele  = np.mean(ele),
        RMSE_azi = np.sqrt(np.mean(azi ** 2)),
        RMSE_ele = np.sqrt(np.mean(ele ** 2)),
        ACC30    = np.mean(mat),
        MDR      = np.mean(gt_u),
        FAR      = np.mean(est_u),
        N_pairs  = len(azi),
    )
