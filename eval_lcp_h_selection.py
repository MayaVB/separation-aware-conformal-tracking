"""
Step 1 of the "improve LCP gradually" request: select the RBF bandwidth h
using calibration data only, per outer 50/50 trajectory split.

For each of the same 20 random 50/50 trajectory splits used by
eval_lcp_repeated_splits.py:
  - build the calibration pool exactly as before (n_calib_frames sampled
    from the calibration-half trajectories);
  - select ONE bandwidth h per speaker via leave-one-out over the ENTIRE
    calibration pool (Code/two_speaker_tracking/lcp.py:select_bandwidth_via_loo)
    -- h_grid = [0.25, 0.5, 1.0, 2.0, 4.0]. No test data and no burst_active
    label are used for this selection (select_bandwidth_via_loo never reads
    burst_active_per_frame). No inner split -- the entire calibration half is
    used both to fit the LCP quantile machinery AND to select h, which is a
    known reuse-of-calibration-data limitation the user explicitly accepted
    for this exploratory pass;
  - freeze the selected h (per speaker), evaluate ONCE on that split's test
    half, bucketed by burst_active_per_frame (label used only for reporting);
  - ALSO evaluate the fixed h=1.0 baseline on the exact same test half, for
    a direct per-split comparison.

Does not touch the tracker, Code/crc_ssl.py, the npz export pipeline, or the
LCP context features (still the 2-D raw_mean/raw_peak_to_background of the
original implementation).
"""

import argparse
import json
import os

import numpy as np

from Code.crc_ssl import CoverageSet
from Code.utilities import normalize
from Code.two_speaker_tracking.npz_adapter import radians_to_grid_index, _build_room
from Code.two_speaker_tracking.lcp import (
    compute_raw_context_features, fit_standardizer, standardize_features,
    compute_calibration_scores, widest_path_score_map, localized_cp_decision,
    calibrate_global_lambda_from_arrays, select_bandwidth_via_loo, run_unit_checks,
)

DATA_ROOT = "/src/data"
DEFAULT_PATH = f"{DATA_ROOT}/npz_output_tracking_burst5_test/Reverb_400_ms_SNR_15_dB/speakers_2_flat.npz"
FRAMES_PER_TRAJECTORY = 104
H_GRID = [0.25, 0.5, 1.0, 2.0, 4.0]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_path", default=DEFAULT_PATH)
    p.add_argument("--n_splits", type=int, default=20)
    p.add_argument("--calib_traj_frac", type=float, default=0.5)
    p.add_argument("--n_calib_frames", type=int, default=500)
    p.add_argument("--n_test_frames", type=int, default=400)
    p.add_argument("--lambda_steps", type=int, default=500)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--fixed_h", type=float, default=1.0, help="baseline h for direct comparison")
    p.add_argument("--seed", type=int, default=0, help="base seed; split i uses seed + i")
    p.add_argument("--out_dir", default="Results")
    return p.parse_args()


def evaluate_test_half(lm_all, est_all, true_all, burst_active_all, gt_valid_all, test_idx,
                        lambdas_global, X_calib_raw, S_calib, standardizers, h_per_speaker, alpha):
    """Runs global CP + LCP-at-h_per_speaker[k] over test_idx. Returns records list."""
    records = []
    for t in test_idx:
        true_order, est_order = CoverageSet._match_estimated_to_source(true_all[t], est_all[t])
        for true_s, est_s in zip(true_order, est_order):
            k = int(est_s)
            raw_map = lm_all[t, k]
            norm_map = normalize(raw_map)
            seed = tuple(radians_to_grid_index(est_all[t, k], *norm_map.shape).astype(int))
            true_idx_grid = tuple(radians_to_grid_index(true_all[t, true_s], *norm_map.shape).astype(int))

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
            region_lcp, _, _, _ = localized_cp_decision(
                S_calib[valid_calib_mask, k], X_calib_std_k[valid_calib_mask],
                S_test_map, X_test_std, h_per_speaker[k], alpha)

            lcp_covered = bool(region_lcp[true_idx_grid])
            lcp_area = int(region_lcp.sum())

            records.append(dict(burst_active=bool(burst_active_all[t]),
                                 global_covered=global_covered, global_area=global_area,
                                 lcp_covered=lcp_covered, lcp_area=lcp_area))
    return records


def summarize(recs, cov_key, area_key):
    if not recs:
        return None
    return dict(n=len(recs),
                cov=float(np.mean([r[cov_key] for r in recs])),
                area=float(np.mean([r[area_key] for r in recs])))


def run_one_split(d, room, nele, nazi, split_seed, args):
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

    lambda_list = np.linspace(0.0, 1.0, args.lambda_steps)
    lambdas_global = calibrate_global_lambda_from_arrays(lm_calib, est_calib, true_calib, room, lambda_list, args.alpha)

    X_calib_raw = compute_raw_context_features(lm_calib)
    S_calib = compute_calibration_scores(lm_calib, est_calib, true_calib, lambda_list, nele, nazi)
    standardizers = [fit_standardizer(X_calib_raw[:, k, :]) for k in range(K)]
    X_calib_std_full = np.stack(
        [standardize_features(X_calib_raw[:, k, :], *standardizers[k]) for k in range(K)], axis=1)  # (n_calib,K,2)

    # --- Step 1: select h per speaker via LOO on the calibration pool only ---
    selected_h = np.zeros(K, dtype=float)
    per_h_stats_by_speaker = []
    for k in range(K):
        h_k, stats_k = select_bandwidth_via_loo(
            lm_calib, est_calib, true_calib, S_calib, X_calib_std_full[:, k, :],
            speaker_k=k, h_grid=H_GRID, alpha=args.alpha, nele=nele, nazi=nazi)
        selected_h[k] = h_k
        per_h_stats_by_speaker.append(stats_k)

    valid_test_idx = test_frame_pool[gt_valid_all[test_frame_pool]]
    test_idx = np.sort(rng.choice(valid_test_idx, size=min(args.n_test_frames, len(valid_test_idx)), replace=False))

    records_selected = evaluate_test_half(lm_all, est_all, true_all, burst_active_all, gt_valid_all, test_idx,
                                           lambdas_global, X_calib_raw, S_calib, standardizers,
                                           h_per_speaker=selected_h, alpha=args.alpha)
    fixed_h = np.full(K, args.fixed_h, dtype=float)
    records_fixed = evaluate_test_half(lm_all, est_all, true_all, burst_active_all, gt_valid_all, test_idx,
                                        lambdas_global, X_calib_raw, S_calib, standardizers,
                                        h_per_speaker=fixed_h, alpha=args.alpha)

    def bucket(recs, active):
        return [r for r in recs if r["burst_active"] == active]

    result = dict(
        selected_h=selected_h.tolist(),
        per_h_stats_by_speaker=per_h_stats_by_speaker,
        calib_coverage_at_selected_h=[per_h_stats_by_speaker[k][selected_h[k]]["coverage"] for k in range(K)],
        calib_mean_area_at_selected_h=[per_h_stats_by_speaker[k][selected_h[k]]["mean_area"] for k in range(K)],
        test_selected=dict(
            overall=summarize(records_selected, "lcp_covered", "lcp_area"),
            active=summarize(bucket(records_selected, True), "lcp_covered", "lcp_area"),
            inactive=summarize(bucket(records_selected, False), "lcp_covered", "lcp_area"),
            global_overall=summarize(records_selected, "global_covered", "global_area"),
            global_active=summarize(bucket(records_selected, True), "global_covered", "global_area"),
            global_inactive=summarize(bucket(records_selected, False), "global_covered", "global_area"),
        ),
        test_fixed_h=dict(
            overall=summarize(records_fixed, "lcp_covered", "lcp_area"),
            active=summarize(bucket(records_fixed, True), "lcp_covered", "lcp_area"),
            inactive=summarize(bucket(records_fixed, False), "lcp_covered", "lcp_area"),
        ),
    )
    return result


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print("Step 0: unit checks")
    print("=" * 70)
    run_unit_checks(verbose=True)

    print()
    print(f"Loading {args.data_path}")
    d = np.load(args.data_path, allow_pickle=True)
    nele, nazi = int(d["nele"]), int(d["nazi"])
    room = _build_room(d)
    N_flat = d["all_likelihood_maps"].shape[0]
    n_traj = N_flat // FRAMES_PER_TRAJECTORY
    print(f"  {n_traj} trajectories, {N_flat} frames, h_grid={H_GRID}, fixed_h={args.fixed_h}")

    print()
    print(f"Running {args.n_splits} splits (LOO h-selection on full calib pool each split -- slow)")
    split_results = []
    for i in range(args.n_splits):
        res = run_one_split(d, room, nele, nazi, split_seed=args.seed + i, args=args)
        split_results.append(res)
        act_sel = res["test_selected"]["active"]
        act_fix = res["test_fixed_h"]["active"]
        print(f"  split {i:3d}  selected_h={res['selected_h']}  "
              f"test_active_cov(selected)={act_sel['cov']:.3f} area={act_sel['area']:.1f}  "
              f"test_active_cov(h=1.0)={act_fix['cov']:.3f} area={act_fix['area']:.1f}"
              if act_sel and act_fix else f"  split {i:3d}  selected_h={res['selected_h']}  (no active frames)")

    # persist raw per-split results as JSON for later re-analysis
    raw_path = os.path.join(args.out_dir, "lcp_h_selection_raw.json")
    with open(raw_path, "w") as fh:
        json.dump(split_results, fh, indent=2)
    print(f"\nSaved raw per-split results -> {raw_path}")

    # --- aggregate ---
    print()
    print("=" * 70)
    print("Aggregate across splits")
    print("=" * 70)

    all_h = np.array(split_results[0]["selected_h"]).size
    K = all_h
    hist = {}
    for res in split_results:
        for h in res["selected_h"]:
            hist[h] = hist.get(h, 0) + 1
    print(f"Histogram of selected h (across {args.n_splits} splits x {K} speakers = {args.n_splits*K} selections):")
    for h in sorted(hist):
        print(f"  h={h:<5} count={hist[h]}")

    def agg_metric(which, bucket, metric):
        vals = [r[which][bucket][metric] for r in split_results if r[which][bucket] is not None]
        if not vals:
            return None
        return float(np.mean(vals)), float(np.std(vals)), len(vals)

    out_lines = [f"Histogram of selected h: {hist}", ""]
    header = f"{'source':<16s} {'bucket':<10s} {'metric':<8s} {'mean':>8s} {'std':>8s} {'n_splits':>9s}"
    sep = "-" * len(header)
    print(header); print(sep)
    out_lines += [header, sep]
    for source, which in [("selected_h", "test_selected"), (f"fixed_h={args.fixed_h}", "test_fixed_h")]:
        for bucket in ["overall", "active", "inactive"]:
            for metric in ["cov", "area"]:
                r = agg_metric(which, bucket, metric)
                if r is None:
                    line = f"{source:<16s} {bucket:<10s} {metric:<8s} {'(no data)':>8s}"
                else:
                    mean, std, n = r
                    line = f"{source:<16s} {bucket:<10s} {metric:<8s} {mean:8.3f} {std:8.3f} {n:9d}"
                print(line)
                out_lines.append(line)

    # global CP reference (from the selected-h run's records; identical baseline either way)
    print()
    print("Global CP reference (same test halves, for context):")
    out_lines += ["", "Global CP reference (same test halves, for context):"]
    for bucket in ["global_overall", "global_active", "global_inactive"]:
        for metric in ["cov", "area"]:
            r = agg_metric("test_selected", bucket, metric)
            if r is None:
                line = f"{'global_cp':<16s} {bucket:<16s} {metric:<8s} {'(no data)':>8s}"
            else:
                mean, std, n = r
                line = f"{'global_cp':<16s} {bucket:<16s} {metric:<8s} {mean:8.3f} {std:8.3f} {n:9d}"
            print(line)
            out_lines.append(line)

    out_txt = os.path.join(args.out_dir, "results_lcp_h_selection.txt")
    with open(out_txt, "w") as fh:
        fh.write("\n".join(out_lines) + "\n")
    print(f"\nSaved -> {out_txt}")


if __name__ == "__main__":
    main()
