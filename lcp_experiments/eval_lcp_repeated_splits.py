"""
Repeated-trajectory-split LCP evaluation.

Generalizes eval_lcp_mixed_calib_experiment.py's single 50/50 calibration/test
trajectory split into MANY independent random 50/50 splits of the same
100-trajectory burst dataset, so we can report mean +/- variability of
active-frame coverage and region area for Global CP vs LCP, instead of a
single point estimate from one arbitrary split.

Each split:
  - shuffles the 100 trajectories with its own RNG seed, takes the first
    calib_traj_frac as the calibration pool, the rest as the test pool
    (trajectory-level, so no frame ever appears on both sides);
  - calibrates global CP (Code.crc_ssl.CoverageSet, read-only reuse) and LCP
    (Code/two_speaker_tracking/lcp.py, Algorithm 7.8) independently on that
    split's calibration pool;
  - evaluates both on that split's test pool, bucketed by burst_active_per_frame
    (the label is used ONLY for bucketing/reporting here, never given to LCP).

Does not touch the tracker, Code/crc_ssl.py, or the npz export pipeline.
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))

import argparse
import os

import numpy as np

from Code.crc_ssl import CoverageSet
from Code.utilities import normalize
from Code.two_speaker_tracking.npz_adapter import radians_to_grid_index, _build_room
from Code.two_speaker_tracking.lcp import (
    compute_raw_context_features, fit_standardizer, standardize_features,
    compute_calibration_scores, widest_path_score_map, localized_cp_decision,
    calibrate_global_lambda_from_arrays, run_unit_checks,
)

DATA_ROOT = "/src/data"
DEFAULT_PATH = f"{DATA_ROOT}/npz_output_tracking_burst_test/Reverb_400_ms_SNR_15_dB/speakers_2_flat.npz"
FRAMES_PER_TRAJECTORY = 104


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_path", default=DEFAULT_PATH,
                    help="single burst dataset used for every split (e.g. the new 5-burst dataset)")
    p.add_argument("--n_splits", type=int, default=20)
    p.add_argument("--calib_traj_frac", type=float, default=0.5)
    p.add_argument("--n_calib_frames", type=int, default=500)
    p.add_argument("--n_test_frames", type=int, default=400)
    p.add_argument("--lambda_steps", type=int, default=500)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--lcp_bandwidth", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0, help="base seed; split i uses seed + i")
    p.add_argument("--out_dir", default="Results")
    return p.parse_args()


def run_one_split(d, room, nele, nazi, split_seed, args):
    """Runs one random 50/50 trajectory split end-to-end. Returns a dict of
    per-split summary stats (mean over that split's records, one number per
    bucket/method/metric) -- NOT raw per-record data, since we aggregate
    across splits at the mean level."""
    rng = np.random.default_rng(split_seed)

    lm_all = d["all_likelihood_maps"]
    est_all = d["all_estimated_positions"]
    true_all = d["speaker_pos"]
    burst_active_all = d["burst_active_per_frame"].astype(bool)
    gt_valid_all = d["gt_valid_per_frame"].astype(bool)
    N_flat, K = lm_all.shape[:2]
    n_traj = N_flat // FRAMES_PER_TRAJECTORY

    traj_order = rng.permutation(n_traj)
    n_calib_traj = int(round(n_traj * args.calib_traj_frac))
    calib_traj = traj_order[:n_calib_traj]
    test_traj = traj_order[n_calib_traj:]

    def traj_to_frames(traj_ids):
        return np.concatenate([np.arange(tr * FRAMES_PER_TRAJECTORY, (tr + 1) * FRAMES_PER_TRAJECTORY)
                                for tr in traj_ids])

    calib_frame_pool = traj_to_frames(calib_traj)
    test_frame_pool = traj_to_frames(test_traj)

    calib_idx = np.sort(rng.choice(calib_frame_pool, size=min(args.n_calib_frames, len(calib_frame_pool)),
                                    replace=False))
    lm_calib = lm_all[calib_idx]
    est_calib = est_all[calib_idx]
    true_calib = true_all[calib_idx]
    calib_burst_active = burst_active_all[calib_idx]

    lambda_list = np.linspace(0.0, 1.0, args.lambda_steps)
    lambdas_global = calibrate_global_lambda_from_arrays(lm_calib, est_calib, true_calib, room, lambda_list, args.alpha)

    X_calib_raw = compute_raw_context_features(lm_calib)
    S_calib = compute_calibration_scores(lm_calib, est_calib, true_calib, lambda_list, nele, nazi)
    standardizers = [fit_standardizer(X_calib_raw[:, k, :]) for k in range(K)]

    valid_test_idx = test_frame_pool[gt_valid_all[test_frame_pool]]
    test_idx = np.sort(rng.choice(valid_test_idx, size=min(args.n_test_frames, len(valid_test_idx)), replace=False))

    records = []
    for t in test_idx:
        true_order, est_order = CoverageSet._match_estimated_to_source(true_all[t], est_all[t])
        for true_s, est_s in zip(true_order, est_order):
            k = int(est_s)
            raw_map = lm_all[t, k]
            norm_map = normalize(raw_map)
            seed = tuple(radians_to_grid_index(est_all[t, k], nele, nazi).astype(int))
            true_idx_grid = tuple(radians_to_grid_index(true_all[t, true_s], nele, nazi).astype(int))

            global_region = CoverageSet.neighbours_coverage_set(norm_map, float(lambdas_global[k]),
                                                                 estimated_position=seed)
            global_covered = bool(global_region[true_idx_grid])
            global_area = int(global_region.sum())

            X_test_raw = compute_raw_context_features(raw_map[None, ...])[0]
            mean_k, std_k = standardizers[k]
            X_test_std = standardize_features(X_test_raw, mean_k, std_k)
            X_calib_std_k = standardize_features(X_calib_raw[:, k, :], mean_k, std_k)

            Lambda_map = widest_path_score_map(norm_map, seed)
            S_test_map = -Lambda_map

            valid_calib_mask = np.isfinite(S_calib[:, k])
            region_lcp, tildeS_test_map, q_map, w_row = localized_cp_decision(
                S_calib[valid_calib_mask, k], X_calib_std_k[valid_calib_mask],
                S_test_map, X_test_std, args.lcp_bandwidth, args.alpha)

            lcp_covered = bool(region_lcp[true_idx_grid])
            lcp_area = int(region_lcp.sum())

            records.append(dict(burst_active=bool(burst_active_all[t]),
                                 global_covered=global_covered, global_area=global_area,
                                 lcp_covered=lcp_covered, lcp_area=lcp_area))

    active = [r for r in records if r["burst_active"]]
    inactive = [r for r in records if not r["burst_active"]]
    overall = records

    def summarize(recs):
        if not recs:
            return None
        return dict(
            n=len(recs),
            global_cov=float(np.mean([r["global_covered"] for r in recs])),
            lcp_cov=float(np.mean([r["lcp_covered"] for r in recs])),
            global_area=float(np.mean([r["global_area"] for r in recs])),
            lcp_area=float(np.mean([r["lcp_area"] for r in recs])),
        )

    return dict(
        n_calib_traj=len(calib_traj), n_test_traj=len(test_traj),
        calib_burst_frac=float(calib_burst_active.mean()),
        overall=summarize(overall), active=summarize(active), inactive=summarize(inactive),
    )


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print("Step 0: unit checks")
    print("=" * 70)
    run_unit_checks(verbose=True)

    print()
    print("=" * 70)
    print(f"Step 1: load {args.data_path} once, reused across all {args.n_splits} splits")
    print("=" * 70)
    d = np.load(args.data_path, allow_pickle=True)
    nele, nazi = int(d["nele"]), int(d["nazi"])
    room = _build_room(d)
    N_flat = d["all_likelihood_maps"].shape[0]
    n_traj = N_flat // FRAMES_PER_TRAJECTORY
    overall_burst_frac = d["burst_active_per_frame"].astype(bool).mean()
    print(f"  {n_traj} trajectories, {N_flat} frames, overall burst-active fraction = {overall_burst_frac:.4f}")

    print()
    print("=" * 70)
    print(f"Step 2: {args.n_splits} independent random 50/50 trajectory splits")
    print("=" * 70)
    split_results = []
    for i in range(args.n_splits):
        res = run_one_split(d, room, nele, nazi, split_seed=args.seed + i, args=args)
        split_results.append(res)
        act = res["active"]
        act_str = (f"n={act['n']:4d} global_cov={act['global_cov']:.3f} lcp_cov={act['lcp_cov']:.3f} "
                   f"global_area={act['global_area']:6.1f} lcp_area={act['lcp_area']:6.1f}") if act else "n=0"
        print(f"  split {i:3d} (calib_burst_frac={res['calib_burst_frac']:.4f})  active: {act_str}")

    print()
    print("=" * 70)
    print("Step 3: aggregate across splits -- mean +/- std")
    print("=" * 70)

    def agg(bucket_name, metric):
        vals = [r[bucket_name][metric] for r in split_results if r[bucket_name] is not None]
        if not vals:
            return None
        return float(np.mean(vals)), float(np.std(vals)), len(vals)

    header = f"{'bucket':<10s} {'metric':<12s} {'mean':>8s} {'std':>8s} {'n_splits':>9s}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    out_lines = [header, sep]
    for bucket in ["overall", "active", "inactive"]:
        for metric in ["global_cov", "lcp_cov", "global_area", "lcp_area"]:
            r = agg(bucket, metric)
            if r is None:
                line = f"{bucket:<10s} {metric:<12s} {'(no data)':>8s}"
            else:
                mean, std, n = r
                line = f"{bucket:<10s} {metric:<12s} {mean:8.3f} {std:8.3f} {n:9d}"
            print(line)
            out_lines.append(line)

    # Also report the per-split difference (LCP - global) for active coverage/area,
    # since that's the quantity that actually answers "does LCP help vs global".
    print()
    print("Per-split (LCP - global), active bucket:")
    diffs_cov = [r["active"]["lcp_cov"] - r["active"]["global_cov"]
                 for r in split_results if r["active"] is not None]
    diffs_area = [r["active"]["lcp_area"] - r["active"]["global_area"]
                  for r in split_results if r["active"] is not None]
    if diffs_cov:
        line1 = f"  coverage diff: mean={np.mean(diffs_cov):+.4f} std={np.std(diffs_cov):.4f} (n={len(diffs_cov)})"
        line2 = f"  area diff:     mean={np.mean(diffs_area):+.2f} std={np.std(diffs_area):.2f} (n={len(diffs_area)})"
        print(line1)
        print(line2)
        out_lines += ["", "Per-split (LCP - global), active bucket:", line1, line2]

    out_txt = os.path.join(args.out_dir, "results_lcp_repeated_splits.txt")
    with open(out_txt, "w") as fh:
        fh.write("\n".join(out_lines) + "\n")
    print(f"\nSaved -> {out_txt}")


if __name__ == "__main__":
    main()
