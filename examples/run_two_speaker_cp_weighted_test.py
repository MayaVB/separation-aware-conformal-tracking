"""
CPWeightedFusionTracker verification against real SRP-DNN npz exports.

Sanity-checks the new CP-weighted full-belief fusion tracker (see
Code/two_speaker_tracking/cp_weighted_tracker.py) on real data:
  - output shapes are correct
  - every per-frame belief map sums to 1 (within tolerance)
  - no NaN/Inf anywhere in beliefs, positions, or w
  - prints summary distributions of w, size_norm, span_norm, var_norm
  - prints a handful of raw per-frame values for spot inspection
  - raw-vs-tracked DOA error as a sanity check (reuses eval_metrics.py,
    unmodified), also compared against the existing baseline
    TwoSpeakerTracker for context

Does NOT tune lambda_var/lambda_size/lambda_span, and does NOT touch
eval_tracker_conditions.py -- this is a standalone diagnostic only.

Run from repo root:
    python examples/run_two_speaker_cp_weighted_test.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from Code.two_speaker_tracking.tracker import TwoSpeakerTracker
from Code.two_speaker_tracking.cp_weighted_tracker import CPWeightedFusionTracker
from Code.two_speaker_tracking.npz_adapter import (
    calibrate_lambda_thresholds, build_cp_regions_for_frames,
    positions_to_grid_indices, grid_index_to_radians,
)
from Code.two_speaker_tracking.eval_metrics import wrap_azi_err_deg, hungarian_errors, compute_metrics

DATA_ROOT = "/src/data"
CALIB_PATH = f"{DATA_ROOT}/npz_output_tracking_no_burst_calib/Reverb_400_ms_SNR_15_dB/speakers_2_flat.npz"
FRAMES_PER_TRAJECTORY = 104
N_TRAJECTORIES = 5      # small slice, this is a smoke test not a full eval
ALPHA = 0.1
LAMBDA_STEPS = 200      # reduced from eval_tracker_conditions.py's default 500 -- dev-speed tradeoff
N_CALIB_FRAMES = 3000   # subsample calibration frames for speed (see calibrate_lambda_thresholds docstring)
SEED = 1234567890


def main():
    print(f"Loading {CALIB_PATH}")
    d = np.load(CALIB_PATH, allow_pickle=True)
    nele, nazi = int(d["nele"]), int(d["nazi"])
    lm_flat = d["all_likelihood_maps"]
    est_flat = d["all_estimated_positions"]
    gt_flat = d["speaker_pos"]
    gt_valid_flat = d["gt_valid_per_frame"]
    K = lm_flat.shape[1]

    print(f"[1/3] Calibrating lambda thresholds (alpha={ALPHA}, "
          f"lambda_steps={LAMBDA_STEPS}, n_calib_frames={N_CALIB_FRAMES})")
    lambdas_by_alpha = calibrate_lambda_thresholds(
        CALIB_PATH, [ALPHA], lambda_steps=LAMBDA_STEPS,
        n_calib_frames=N_CALIB_FRAMES, seed=SEED)
    lambdas = lambdas_by_alpha[ALPHA]
    print(f"    lambdas={lambdas}")

    # ------------------------------------------------------------------
    # Accumulators
    # ------------------------------------------------------------------
    w_all, size_norm_all, span_norm_all, var_norm_all = [], [], [], []
    n_nan_inf = 0
    n_belief_checked = 0
    sum_dev_max = 0.0

    raw_acc = {"azi": [], "ele": [], "mat": [], "gtu": [], "estu": []}
    baseline_acc = {"azi": [], "ele": [], "mat": [], "gtu": [], "estu": []}
    cpw_acc = {"azi": [], "ele": [], "mat": [], "gtu": [], "estu": []}

    print(f"[2/3] Running trackers over {N_TRAJECTORIES} trajectories")
    for i in range(N_TRAJECTORIES):
        s, e = i * FRAMES_PER_TRAJECTORY, (i + 1) * FRAMES_PER_TRAJECTORY
        lm_traj = lm_flat[s:e]              # (T,K,nele,nazi) raw
        est_rad_traj = est_flat[s:e]        # (T,K,2) radians
        gt_rad_traj = gt_flat[s:e]          # (T,K,2) radians
        valid_traj = gt_valid_flat[s:e]     # (T,) bool

        cp_regions = build_cp_regions_for_frames(lm_traj, est_rad_traj, lambdas, nele, nazi)
        est_grid_traj = positions_to_grid_indices(est_rad_traj, nele, nazi)

        baseline_tracker = TwoSpeakerTracker(n_speakers=K)
        baseline_result = baseline_tracker.run(lm_traj, cp_regions.astype(float), est_grid_traj)

        cpw_tracker = CPWeightedFusionTracker(n_speakers=K)
        cpw_result = cpw_tracker.run(lm_traj, cp_regions.astype(float), est_grid_traj)

        # ---- shape checks -------------------------------------------------
        assert cpw_result["posterior_maps"].shape == (FRAMES_PER_TRAJECTORY, K, nele, nazi), \
            f"posterior_maps shape mismatch: {cpw_result['posterior_maps'].shape}"
        assert cpw_result["assignments"].shape == (FRAMES_PER_TRAJECTORY, K), \
            f"assignments shape mismatch: {cpw_result['assignments'].shape}"
        for k in range(K):
            assert cpw_result["tracks"][k].shape == (FRAMES_PER_TRAJECTORY, 2), \
                f"tracks[{k}] shape mismatch: {cpw_result['tracks'][k].shape}"

        # ---- belief-sums-to-1 + no-NaN/Inf checks --------------------------
        beliefs = cpw_result["posterior_maps"]
        n_belief_checked += beliefs.shape[0] * beliefs.shape[1]
        sums = beliefs.sum(axis=(2, 3))
        sum_dev_max = max(sum_dev_max, float(np.max(np.abs(sums - 1.0))))
        if not np.all(np.isfinite(beliefs)):
            n_nan_inf += int(np.sum(~np.isfinite(beliefs)))
        for k in range(K):
            pos = cpw_result["tracks"][k]
            if not np.all(np.isfinite(pos)):
                n_nan_inf += int(np.sum(~np.isfinite(pos)))

        # ---- collect debug descriptors (skip frame 0, which is NaN by design) ----
        dbg = cpw_result["debug"]
        w_all.append(dbg["w"][1:].ravel())
        size_norm_all.append(dbg["size_norm"][1:].ravel())
        span_norm_all.append(dbg["span_norm"][1:].ravel())
        var_norm_all.append(dbg["var_norm"][1:].ravel())
        if not np.all(np.isfinite(dbg["w"][1:])):
            n_nan_inf += int(np.sum(~np.isfinite(dbg["w"][1:])))

        # ---- DOA-error sanity check (raw vs baseline vs cp-weighted) ------
        tracked_rad_baseline = np.stack(
            [np.array([grid_index_to_radians(baseline_result["tracks"][k][t], nele, nazi)
                       for t in range(FRAMES_PER_TRAJECTORY)]) for k in range(K)],
            axis=1)
        tracked_rad_cpw = np.stack(
            [np.array([grid_index_to_radians(cpw_result["tracks"][k][t], nele, nazi)
                       for t in range(FRAMES_PER_TRAJECTORY)]) for k in range(K)],
            axis=1)

        for t in range(FRAMES_PER_TRAJECTORY):
            if not valid_traj[t]:
                continue
            gt_ele, gt_azi = gt_rad_traj[t, :, 0], gt_rad_traj[t, :, 1]
            for acc, pos_traj in ((raw_acc, est_rad_traj),
                                   (baseline_acc, tracked_rad_baseline),
                                   (cpw_acc, tracked_rad_cpw)):
                est_ele, est_azi = pos_traj[t, :, 0], pos_traj[t, :, 1]
                ae, ee, mat, gtu, estu = hungarian_errors(est_ele, est_azi, gt_ele, gt_azi)
                acc["azi"].extend(ae)
                acc["ele"].extend(ee)
                acc["mat"].extend(mat)
                acc["gtu"].extend(gtu.tolist())
                acc["estu"].extend(estu.tolist())

        print(f"    trajectory {i}: OK (max |belief_sum - 1| = {sum_dev_max:.2e})")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("\n[3/3] Results")
    print("=" * 70)
    print(f"Beliefs checked        : {n_belief_checked}")
    print(f"Max |belief_sum - 1|   : {sum_dev_max:.2e}")
    print(f"NaN/Inf count (beliefs+positions+w): {n_nan_inf}")

    w_all = np.concatenate(w_all)
    size_norm_all = np.concatenate(size_norm_all)
    span_norm_all = np.concatenate(span_norm_all)
    var_norm_all = np.concatenate(var_norm_all)

    def _stats(name, arr):
        print(f"  {name:<10s}  min={np.min(arr):.4f}  median={np.median(arr):.4f}  "
              f"mean={np.mean(arr):.4f}  max={np.max(arr):.4f}")

    print("\nDescriptor / weight distributions (frames 1..T-1, both speakers, "
          f"{N_TRAJECTORIES} trajectories, n={w_all.size}):")
    _stats("w", w_all)
    _stats("size_norm", size_norm_all)
    _stats("span_norm", span_norm_all)
    _stats("var_norm", var_norm_all)

    print("\nSpot check -- first 10 frames, trajectory 0, speaker 0:")
    dbg0 = None
    # recompute for trajectory 0 for the printout (cheap, already ran above but not retained)
    lm_traj = lm_flat[0:FRAMES_PER_TRAJECTORY]
    est_rad_traj = est_flat[0:FRAMES_PER_TRAJECTORY]
    cp_regions = build_cp_regions_for_frames(lm_traj, est_rad_traj, lambdas, nele, nazi)
    est_grid_traj = positions_to_grid_indices(est_rad_traj, nele, nazi)
    cpw_tracker = CPWeightedFusionTracker(n_speakers=K)
    cpw_result = cpw_tracker.run(lm_traj, cp_regions.astype(float), est_grid_traj)
    dbg0 = cpw_result["debug"]
    print(f"{'t':>2}  {'w':>7}  {'size_norm':>9}  {'span_norm':>9}  {'var_norm':>9}")
    for t in range(10):
        print(f"{t:>2}  {dbg0['w'][t, 0]:>7.4f}  {dbg0['size_norm'][t, 0]:>9.4f}  "
              f"{dbg0['span_norm'][t, 0]:>9.4f}  {dbg0['var_norm'][t, 0]:>9.4f}")

    print("\nDOA error sanity check (Hungarian-matched, degrees):")
    for label, acc in (("raw (untracked)", raw_acc),
                        ("baseline TwoSpeakerTracker", baseline_acc),
                        ("CPWeightedFusionTracker", cpw_acc)):
        m = compute_metrics(acc["azi"], acc["ele"], acc["mat"], acc["gtu"], acc["estu"])
        print(f"  {label:<28s}  MAE_azi={m['MAE_azi']:.2f}  MAE_ele={m['MAE_ele']:.2f}  "
              f"RMSE_azi={m['RMSE_azi']:.2f}  ACC30={m['ACC30']:.3f}  N={m['N_pairs']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
