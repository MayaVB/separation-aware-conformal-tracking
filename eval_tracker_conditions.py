"""
Evaluate TwoSpeakerTracker (CP-region-modulated Bayesian tracker) against
real SRP-DNN heatmap exports, across noise conditions, to answer: in which
noise scenario does the tracker actually help over the raw (untracked)
SRP-DNN estimate?

Two hypotheses under test:
  (a) noise that expands the CP region around the true speaker (candidates:
      spatial-noise-cloud, explicit-diffuse) -> does a bigger region make the
      tracker lean on its motion prior more, and does that help?
  (b) burst scenarios where the heatmap peak gets "stolen" -> does the
      tracker's smoothing reduce the resulting single-frame trajectory jump?

See /home/mayavb/.claude/plans/ok-we-need-to-lazy-pumpkin.md for the full
design. Known methodological caveat (deferred, not fixed here): calibration
always uses the clean no_burst_calib set while testing on burst/cloud/diffuse
-- calib/test distributions don't match, which can affect conformal coverage
independent of the perturbation itself.
"""

import argparse
import os

import numpy as np
from scipy.optimize import linear_sum_assignment

from Code.two_speaker_tracking.npz_adapter import (
    calibrate_lambda_thresholds, build_cp_regions_for_frames,
    positions_to_grid_indices, grid_index_to_radians,
)
from Code.two_speaker_tracking.eval_metrics import (
    wrap_azi_err_deg, hungarian_errors, compute_metrics,
)
from Code.two_speaker_tracking.tracker import TwoSpeakerTracker

DATA_ROOT = "/src/data"
DEFAULT_CALIB = f"{DATA_ROOT}/npz_output_tracking_no_burst_calib/Reverb_400_ms_SNR_15_dB/speakers_2_flat.npz"
DEFAULT_TEST_PATHS = [
    ("baseline", DEFAULT_CALIB),
    ("burst",    f"{DATA_ROOT}/npz_output_tracking_burst_test/Reverb_400_ms_SNR_15_dB/speakers_2_flat.npz"),
    ("cloud",    f"{DATA_ROOT}/npz_output_tracking_cloud_test_norm/Reverb_400_ms_SNR_15_dB/speakers_2_flat.npz"),
    ("diffuse",  f"{DATA_ROOT}/npz_output_tracking_diffuse_test_norm/Reverb_400_ms_SNR_15_dB/speakers_2_flat.npz"),
]
ACTIVE_FLAG_KEY = {"burst": "burst_active_per_frame", "cloud": "cloud_active_per_frame"}
FRAMES_PER_TRAJECTORY = 104  # verified constant across all 4 real exports


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calib_path", default=DEFAULT_CALIB)
    p.add_argument("--n_trajectories", type=int, default=None, help="None = all 100")
    p.add_argument("--lambda_steps", type=int, default=500)
    p.add_argument("--significance_levels", type=float, nargs="+", default=[0.1, 0.05])
    p.add_argument("--alpha", type=float, default=0.1, help="which significance level drives cp_regions/tracker")
    p.add_argument("--n_calib_frames", type=int, default=None, help="None = all calib frames")
    p.add_argument("--out_dir", default="Results")
    p.add_argument("--seed", type=int, default=1234567890)
    return p.parse_args()


def reindex_by_gt_match(est_ele, est_azi, gt_ele, gt_azi):
    """Hungarian-match K estimates to K GT sources; return est reindexed so
    slot g holds the estimate matched to GT speaker g (fair identity for
    frame-to-frame jump comparisons -- see module docstring)."""
    K = len(est_azi)
    cost = np.array([[wrap_azi_err_deg(est_azi[e], gt_azi[g]) for g in range(K)] for e in range(K)])
    est_idx, gt_idx = linear_sum_assignment(cost)
    out_ele = np.full(K, np.nan)
    out_azi = np.full(K, np.nan)
    for e, g in zip(est_idx, gt_idx):
        out_ele[g] = est_ele[e]
        out_azi[g] = est_azi[e]
    return out_ele, out_azi


def eval_condition(name, npz_path, lambdas, alpha, n_trajectories, seed):
    d = np.load(npz_path, allow_pickle=True)
    nele, nazi = int(d["nele"]), int(d["nazi"])
    lm_flat = d["all_likelihood_maps"]
    est_flat = d["all_estimated_positions"]
    gt_flat = d["speaker_pos"]
    gt_valid_flat = d["gt_valid_per_frame"]
    N_flat = lm_flat.shape[0]
    K = lm_flat.shape[1]
    n_traj = N_flat // FRAMES_PER_TRAJECTORY
    assert n_traj * FRAMES_PER_TRAJECTORY == N_flat, (
        f"{name}: N_flat={N_flat} not a multiple of {FRAMES_PER_TRAJECTORY}")

    active_flat = d[ACTIVE_FLAG_KEY[name]].astype(bool) if name in ACTIVE_FLAG_KEY else None

    use_traj = n_traj if n_trajectories is None else min(n_trajectories, n_traj)

    # raw/tracked azimuth error + jump accumulators, split by active/inactive/overall
    buckets = ["overall"] + (["active", "inactive"] if active_flat is not None else [])
    acc = {b: {"raw": {"azi": [], "ele": [], "mat": [], "gtu": [], "estu": []},
               "tracked": {"azi": [], "ele": [], "mat": [], "gtu": [], "estu": []}}
           for b in buckets}
    jump_acc = {b: {"raw": [], "tracked": []} for b in buckets}

    for i in range(use_traj):
        s, e = i * FRAMES_PER_TRAJECTORY, (i + 1) * FRAMES_PER_TRAJECTORY
        lm_traj = lm_flat[s:e]              # (T,K,nele,nazi) raw
        est_rad_traj = est_flat[s:e]        # (T,K,2) radians
        gt_rad_traj = gt_flat[s:e]          # (T,K,2) radians
        valid_traj = gt_valid_flat[s:e]     # (T,) bool
        active_traj = active_flat[s:e] if active_flat is not None else None

        cp_regions = build_cp_regions_for_frames(lm_traj, est_rad_traj, lambdas, nele, nazi)
        est_grid_traj = positions_to_grid_indices(est_rad_traj, nele, nazi)

        tracker = TwoSpeakerTracker(n_speakers=K)
        result = tracker.run(lm_traj, cp_regions.astype(float), est_grid_traj)
        tracked_rad_traj = np.stack(
            [np.array([grid_index_to_radians(result["tracks"][k][t], nele, nazi)
                       for t in range(FRAMES_PER_TRAJECTORY)]) for k in range(K)],
            axis=1)  # (T,K,2) radians

        # per-frame GT-aligned identity + metrics + jumps
        prev_raw_by_gt = None
        prev_tracked_by_gt = None
        for t in range(FRAMES_PER_TRAJECTORY):
            if not valid_traj[t]:
                prev_raw_by_gt = None
                prev_tracked_by_gt = None
                continue
            gt_ele, gt_azi = gt_rad_traj[t, :, 0], gt_rad_traj[t, :, 1]

            for label, pos_traj in (("raw", est_rad_traj), ("tracked", tracked_rad_traj)):
                est_ele, est_azi = pos_traj[t, :, 0], pos_traj[t, :, 1]
                ae, ee, mat, gtu, estu = hungarian_errors(est_ele, est_azi, gt_ele, gt_azi)
                which_buckets = ["overall"]
                if active_traj is not None:
                    which_buckets.append("active" if active_traj[t] else "inactive")
                for b in which_buckets:
                    acc[b][label]["azi"].extend(ae)
                    acc[b][label]["ele"].extend(ee)
                    acc[b][label]["mat"].extend(mat)
                    acc[b][label]["gtu"].extend(gtu.tolist())
                    acc[b][label]["estu"].extend(estu.tolist())

            raw_by_gt_ele, raw_by_gt_azi = reindex_by_gt_match(
                est_rad_traj[t, :, 0], est_rad_traj[t, :, 1], gt_ele, gt_azi)
            tracked_by_gt_ele, tracked_by_gt_azi = reindex_by_gt_match(
                tracked_rad_traj[t, :, 0], tracked_rad_traj[t, :, 1], gt_ele, gt_azi)

            if prev_raw_by_gt is not None:
                which_buckets = ["overall"]
                if active_traj is not None:
                    which_buckets.append("active" if active_traj[t] else "inactive")
                for g in range(K):
                    j_raw = wrap_azi_err_deg(raw_by_gt_azi[g], prev_raw_by_gt[g])
                    j_trk = wrap_azi_err_deg(tracked_by_gt_azi[g], prev_tracked_by_gt[g])
                    for b in which_buckets:
                        jump_acc[b]["raw"].append(j_raw)
                        jump_acc[b]["tracked"].append(j_trk)

            prev_raw_by_gt = raw_by_gt_azi
            prev_tracked_by_gt = tracked_by_gt_azi

    metrics = {}
    for b in buckets:
        metrics[b] = {}
        for label in ("raw", "tracked"):
            a = acc[b][label]
            if len(a["azi"]) == 0:
                metrics[b][label] = None
                continue
            m = compute_metrics(a["azi"], a["ele"], a["mat"], a["gtu"], a["estu"])
            j = jump_acc[b][label]
            m["mean_jump_deg"] = float(np.mean(j)) if j else float("nan")
            m["median_jump_deg"] = float(np.median(j)) if j else float("nan")
            m["max_jump_deg"] = float(np.max(j)) if j else float("nan")
            metrics[b][label] = m

    return metrics, use_traj, N_flat if n_trajectories is None else use_traj * FRAMES_PER_TRAJECTORY


def fmt_row(label, m):
    if m is None:
        return f"{label:<22s}  (no frames in this bucket)"
    return (f"{label:<22s}  {m['MAE_azi']:>7.2f}  {m['MAE_ele']:>7.2f}  "
            f"{m['RMSE_azi']:>8.2f}  {m['RMSE_ele']:>8.2f}  {m['ACC30']:>7.3f}  "
            f"{m['mean_jump_deg']:>9.2f}  {m['median_jump_deg']:>9.2f}  {m['max_jump_deg']:>8.2f}")


def main():
    args = parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[1/2] Calibrating lambda thresholds from {args.calib_path} "
          f"(n_calib_frames={args.n_calib_frames}, lambda_steps={args.lambda_steps})")
    lambdas_by_alpha = calibrate_lambda_thresholds(
        args.calib_path, args.significance_levels, lambda_steps=args.lambda_steps,
        n_calib_frames=args.n_calib_frames, seed=args.seed)
    for a, lam in lambdas_by_alpha.items():
        print(f"    alpha={a}: lambda={lam}")
    lambdas = lambdas_by_alpha[args.alpha]
    print(f"[1/2] Using alpha={args.alpha} -> lambdas={lambdas} for cp_regions/tracker")

    csv_rows = []
    header = ("label", "MAE_azi", "MAE_ele", "RMSE_azi", "RMSE_ele", "ACC30",
              "mean_jump_deg", "median_jump_deg", "max_jump_deg")

    for name, path in DEFAULT_TEST_PATHS:
        print(f"\n[2/2] Condition: {name}  ({path})")
        metrics, use_traj, n_flat_used = eval_condition(
            name, path, lambdas, args.alpha, args.n_trajectories, args.seed)

        col_header = (f"{'':22s}  {'MAE_azi':>7s}  {'MAE_ele':>7s}  {'RMSE_azi':>8s}  "
                      f"{'RMSE_ele':>8s}  {'ACC@30%':>7s}  {'MeanJump':>9s}  "
                      f"{'MedJump':>9s}  {'MaxJump':>8s}")
        sep = "-" * len(col_header)
        lines = [f"Condition={name}  trajectories={use_traj}  frames={n_flat_used}", "", col_header, sep]
        for bucket in metrics:
            for label in ("raw", "tracked"):
                m = metrics[bucket][label]
                lines.append(fmt_row(f"{bucket}/{label}:", m))
                if m is not None:
                    csv_rows.append({"condition": name, "bucket": bucket, "label": label, **{
                        k: m[k] for k in ("MAE_azi", "MAE_ele", "RMSE_azi", "RMSE_ele", "ACC30",
                                          "mean_jump_deg", "median_jump_deg", "max_jump_deg")}})
        text = "\n".join(lines) + "\n"
        print(text)
        out_txt = os.path.join(
            args.out_dir, f"results_tracker_eval_{name}_Reverb_400_ms_SNR_15_dB_speakers2.txt")
        with open(out_txt, "w") as fh:
            fh.write(text)
        print(f"    -> {out_txt}")

    csv_path = os.path.join(args.out_dir, "tracker_eval_summary.csv")
    with open(csv_path, "w") as fh:
        cols = ["condition", "bucket", "label"] + list(header[1:])
        fh.write(",".join(cols) + "\n")
        for row in csv_rows:
            fh.write(",".join(str(row[c]) for c in cols) + "\n")
    print(f"\nSummary CSV -> {csv_path}")


if __name__ == "__main__":
    main()
