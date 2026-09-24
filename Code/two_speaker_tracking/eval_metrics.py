"""
Shared per-frame localization-error metrics.

Extracted from eval_heatmap_noburst.py so eval_tracker_conditions.py (and any
other script) can reuse the same Hungarian-matched azimuth/elevation error
logic instead of duplicating it. No behavior change from the original
module-level functions — only the implicit dependency on module-global `K`
was made an explicit parameter (derived from the input arrays' own length).

Two kinds of metric live here -- do not confuse them:

- LOCALIZATION-ONLY (hungarian_errors, and the "hungarian" outputs of
  persistent_identity_eval): estimates are re-matched to GT independently at
  EVERY frame. This measures how close the set of estimates is to the set
  of sources, and is BLIND to identity swaps (a tracker that swaps its two
  tracks every frame scores the same as a persistent one).
- PERSISTENT-IDENTITY (persistent_identity_eval): one track->GT mapping is
  fixed at the start of the trajectory and kept for the whole trajectory;
  also counts identity switches.

GT is used here for EVALUATION ONLY. Nothing in this module is (or may be)
called by the tracker at inference time.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from Code.two_speaker_tracking.utils import great_circle_deg


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
    Hungarian-matched errors for one frame. LOCALIZATION-ONLY metric: the
    matching is recomputed at every frame (azimuth-only cost), so it cannot
    see identity swaps -- see module docstring.

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


# ---------------------------------------------------------------------------
# Persistent-identity evaluation (K = 2)
# ---------------------------------------------------------------------------

_PERMS_2 = ((0, 1), (1, 0))  # perm[k] = GT source assigned to track k


def persistent_identity_eval(track_rad, gt_rad, valid=None, min_margin_deg=0.0,
                             init_margin_deg=0.0):
    """Persistent-identity evaluation of one two-speaker trajectory.

    EVALUATION ONLY -- uses GT; never call from inference-time code.

    Parameters
    ----------
    track_rad : (T, 2, 2) radians [ele (polar), azi], TRACK order (track k at [:, k])
    gt_rad    : (T, 2, 2) radians, GT source order (persistent simulator index)
    valid     : (T,) bool or None -- frames with real GT (npz gt_valid_per_frame)
    min_margin_deg : float
        A frame's locally-best matching only counts toward identity switches /
        mismatches if it beats the other permutation by at least this many
        degrees (total great-circle cost). 0.0 = pure definition. >0 makes
        the switch count robust to flicker when the two sources coincide.
    init_margin_deg : float
        Initialisation rule: the fixed mapping is taken at the FIRST valid
        frame t_init whose margin |cost(perm A) - cost(perm B)| >= this value
        (0.0 = first valid frame). Frames before t_init are excluded from the
        persistent metrics (they would need later GT to be scored); the
        Hungarian (localization-only) error is still reported for them. If no
        frame qualifies, persistent metrics are NaN for this trajectory.

    Definitions (all angular errors are great-circle degrees, per track):
    - sigma_t : locally-best permutation at frame t (min total error over the
                2 permutations) -- this IS the per-frame Hungarian matching.
    - initial mapping pi = sigma_t_init at the first valid frame whose margin
      >= init_margin_deg (only that frame's GT), then held FIXED.
    - persistent error e_t,k = GC(track_k(t), gt_pi(k)(t)).
    - hungarian error  h_t,k = GC(track_k(t), gt_sigma_t(k)(t))  (localization-only).
    - identity switch: walking forward from t0 with a "current" mapping
      initialised to pi, a switch is counted at frame t when sigma_t differs
      from the current mapping (and its margin >= min_margin_deg); the
      current mapping is then updated to sigma_t. This is the CLEAR-MOT IDSW
      notion specialised to 2 always-present sources/tracks, counted once per
      swap event (a swap moves both tracks, it is not counted twice).
    - mismatch fraction: fraction of valid frames (margin >= min_margin_deg)
      where sigma_t != pi, i.e. the fixed mapping is currently wrong.

    Returns
    -------
    dict with
      "initial_mapping"  : tuple, pi (pi[k] = GT source of track k)
      "persistent_err"   : (T, 2) float, NaN on invalid frames
      "hungarian_err"    : (T, 2) float, NaN on invalid frames
      "sigma"            : (T,) int, index into ((0,1),(1,0)); -1 on invalid frames
      "margin_deg"       : (T,) float, |cost(perm A) - cost(perm B)|
      "id_switches"      : int
      "mismatch_frac"    : float
      "t_init"           : int or None
      "n_pre_init"       : int, valid frames excluded before t_init
      "switch_frames"    : list of int, frames at which an identity switch was counted
    """
    track_rad = np.asarray(track_rad, dtype=float)
    gt_rad = np.asarray(gt_rad, dtype=float)
    T = track_rad.shape[0]
    assert track_rad.shape == (T, 2, 2) and gt_rad.shape == (T, 2, 2), (
        f"expected (T,2,2) inputs, got {track_rad.shape}, {gt_rad.shape}")
    valid = np.ones(T, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)

    # err[t, k, g] = GC(track k, GT g)
    err = great_circle_deg(track_rad[:, :, None, 0], track_rad[:, :, None, 1],
                           gt_rad[:, None, :, 0], gt_rad[:, None, :, 1])
    perm_cost = np.stack([err[:, 0, p[0]] + err[:, 1, p[1]] for p in _PERMS_2], axis=1)  # (T, 2)
    sigma = np.argmin(perm_cost, axis=1)  # ties -> perm 0
    margin = np.abs(perm_cost[:, 0] - perm_cost[:, 1])

    persistent = np.full((T, 2), np.nan)
    hungarian = np.full((T, 2), np.nan)
    sigma_out = np.full(T, -1, dtype=int)
    valid_idx = np.flatnonzero(valid)
    for t in valid_idx:
        s = _PERMS_2[sigma[t]]
        hungarian[t] = [err[t, k, s[k]] for k in range(2)]
        sigma_out[t] = sigma[t]

    init_candidates = valid_idx[margin[valid_idx] >= init_margin_deg]
    if len(init_candidates) == 0:
        return dict(initial_mapping=None, persistent_err=persistent, hungarian_err=hungarian,
                    sigma=sigma_out, margin_deg=margin, id_switches=np.nan, mismatch_frac=np.nan,
                    t_init=None, n_pre_init=int(len(valid_idx)), switch_frames=[])
    t_init = int(init_candidates[0])

    pi_idx = int(sigma[t_init])
    pi = _PERMS_2[pi_idx]
    current = pi_idx
    id_switches = 0
    n_counted = 0
    n_mismatch = 0
    switch_frames = []
    for t in valid_idx[valid_idx >= t_init]:
        persistent[t] = [err[t, k, pi[k]] for k in range(2)]
        if margin[t] >= min_margin_deg:
            n_counted += 1
            n_mismatch += int(sigma[t] != pi_idx)
            if sigma[t] != current:
                id_switches += 1
                switch_frames.append(int(t))
                current = int(sigma[t])

    return dict(initial_mapping=pi, persistent_err=persistent, hungarian_err=hungarian,
                sigma=sigma_out, margin_deg=margin, id_switches=id_switches,
                mismatch_frac=(n_mismatch / n_counted) if n_counted else np.nan,
                t_init=t_init, n_pre_init=int(np.sum(valid_idx < t_init)),
                switch_frames=switch_frames)
