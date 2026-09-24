"""
Static-context-feature ablation for LCP (Algorithm 7.8), speaker-wise.

Runs the exact same 20 random 50/50 trajectory splits as
eval_lcp_repeated_splits.py (same 100-trajectory, 5-burst dataset
npz_output_tracking_burst5_test, same base seed=0 -> split i uses seed+i,
same calib_traj_frac/n_calib_frames/n_test_frames/lambda_steps defaults),
with h and alpha FIXED -- the earlier LOO h-selection experiment
(eval_lcp_h_selection.py, see Results/results_lcp_h_selection.txt) did not
improve results, so this script does NOT optimize h:
    h = 1.0
    alpha = 0.1

For each split, evaluates Global CP once (feature-independent) and LCP under
4 static context-feature configurations
(Code.two_speaker_tracking.lcp.FEATURE_SETS):
    original : [raw_mean, raw_peak_to_background]        (existing LCP baseline, eps=1e-8)
    F2       : [raw_median, raw_std]                      (eps=0.02)
    F3       : [raw_median, raw_std, median_over_mean]     (eps=0.02)
    F4       : [raw_median, raw_std, median_over_mean, log_peak_over_median]  (eps=0.02)
Numerical definitions and eps=0.02 match analyze_lcp_feature_diagnostic_v2.py
exactly. No temporal features, effective-neighborhood control, adaptive
bandwidth, or new kernels are added; water-filling/global CP/the tracker are
untouched.

burst_active_per_frame is used ONLY to bucket results into active/inactive
for reporting -- never in fitting, standardization, calibration, weighting,
or feature construction. Feature standardization (fit_standardizer /
standardize_features) is calibration-set-only, fit separately per speaker,
exactly as in the existing LCP implementation -- just re-fit per feature
config instead of once, since each config has its own feature space.

Per-split expensive steps that do NOT depend on the feature config -- global
CP calibration (CoverageSet.calibrate()), the LCP calibration scores S_i
(compute_calibration_scores, purely a function of likelihood maps + the
lambda grid, not of X), and the test-time widest-path score maps S_{n+1}(y)
(widest_path_score_map, purely a function of the raw map + estimated seed) --
are each computed ONCE per split and reused across all 4 feature configs,
instead of redone per config. Only feature computation + standardization +
the (cheap, vectorized) localized_cp_decision call varies per config.

Does not touch the tracker, Code/crc_ssl.py, the npz export pipeline,
compute_raw_context_features, or eval_lcp_repeated_splits.py.
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..")))

import argparse
import json
import os

import numpy as np

from Code.crc_ssl import CoverageSet
from Code.utilities import normalize
from Code.two_speaker_tracking.npz_adapter import radians_to_grid_index, _build_room
from Code.two_speaker_tracking.lcp import (
    FEATURE_SETS, compute_context_features, fit_standardizer, standardize_features,
    compute_calibration_scores, widest_path_score_map, localized_cp_decision,
    calibrate_global_lambda_from_arrays, run_unit_checks,
)

DATA_ROOT = "/src/data"
DEFAULT_PATH = f"{DATA_ROOT}/npz_output_tracking_burst5_test/Reverb_400_ms_SNR_15_dB/speakers_2_flat.npz"
FRAMES_PER_TRAJECTORY = 104
ALL_CONFIGS = ["original", "F2", "F3", "F4"]
BUCKETS = ["overall", "active", "inactive"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_path", default=DEFAULT_PATH)
    p.add_argument("--n_splits", type=int, default=20)
    p.add_argument("--calib_traj_frac", type=float, default=0.5)
    p.add_argument("--n_calib_frames", type=int, default=500)
    p.add_argument("--n_test_frames", type=int, default=400)
    p.add_argument("--lambda_steps", type=int, default=500)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--lcp_bandwidth", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0, help="base seed; split i uses seed + i")
    p.add_argument("--out_dir", default="Results")
    p.add_argument("--configs", nargs="+", default=ALL_CONFIGS, choices=list(FEATURE_SETS.keys()),
                    help="subset of FEATURE_SETS to evaluate against global CP + each other "
                         "(default: all of them; e.g. --configs original F3 for a cheaper 2-config run)")
    return p.parse_args()


def summarize(recs, cov_key, area_key):
    if not recs:
        return None
    return dict(n=len(recs),
                cov=float(np.mean([r[cov_key] for r in recs])),
                area=float(np.mean([r[area_key] for r in recs])))


def run_one_split(d, room, nele, nazi, split_seed, args, configs):
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

    # ---- feature-independent, computed ONCE per split, shared by all configs ----
    lambdas_global = calibrate_global_lambda_from_arrays(
        lm_calib, est_calib, true_calib, room, lambda_list, args.alpha)
    S_calib = compute_calibration_scores(lm_calib, est_calib, true_calib, lambda_list, nele, nazi)

    valid_test_idx = test_frame_pool[gt_valid_all[test_frame_pool]]
    test_idx = np.sort(rng.choice(valid_test_idx, size=min(args.n_test_frames, len(valid_test_idx)), replace=False))

    shared_records = []
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

            Lambda_map = widest_path_score_map(norm_map, seed)
            S_test_map = -Lambda_map

            shared_records.append(dict(
                k=k, raw_map=raw_map, true_idx_grid=true_idx_grid, S_test_map=S_test_map,
                burst_active=bool(burst_active_all[t]),
                global_covered=global_covered, global_area=global_area,
            ))

    # ---- per feature-config: standardize (calib-only, per speaker) + LCP decision ----
    per_config_records = {cfg: [] for cfg in configs}
    for cfg in configs:
        feature_names, eps = FEATURE_SETS[cfg]
        X_calib_raw = compute_context_features(lm_calib, feature_names, eps)  # (n_calib, K, d)
        standardizers = [fit_standardizer(X_calib_raw[:, kk, :]) for kk in range(K)]

        for rec in shared_records:
            k = rec["k"]
            mean_k, std_k = standardizers[k]
            X_test_raw = compute_context_features(rec["raw_map"][None, ...], feature_names, eps)[0]
            X_test_std = standardize_features(X_test_raw, mean_k, std_k)
            X_calib_std_k = standardize_features(X_calib_raw[:, k, :], mean_k, std_k)

            valid_calib_mask = np.isfinite(S_calib[:, k])
            region_lcp, _, _, _ = localized_cp_decision(
                S_calib[valid_calib_mask, k], X_calib_std_k[valid_calib_mask],
                rec["S_test_map"], X_test_std, args.lcp_bandwidth, args.alpha)

            per_config_records[cfg].append(dict(
                burst_active=rec["burst_active"],
                lcp_covered=bool(region_lcp[rec["true_idx_grid"]]),
                lcp_area=int(region_lcp.sum()),
            ))

    global_summary = dict(
        overall=summarize(shared_records, "global_covered", "global_area"),
        active=summarize([r for r in shared_records if r["burst_active"]], "global_covered", "global_area"),
        inactive=summarize([r for r in shared_records if not r["burst_active"]], "global_covered", "global_area"),
    )
    config_summaries = {}
    for cfg in configs:
        recs = per_config_records[cfg]
        config_summaries[cfg] = dict(
            overall=summarize(recs, "lcp_covered", "lcp_area"),
            active=summarize([r for r in recs if r["burst_active"]], "lcp_covered", "lcp_area"),
            inactive=summarize([r for r in recs if not r["burst_active"]], "lcp_covered", "lcp_area"),
        )

    return dict(n_calib_traj=len(calib_traj), n_test_traj=len(test_traj),
                calib_burst_frac=float(calib_burst_active.mean()),
                global_cp=global_summary, **config_summaries)


def get_bucket(split_result, method, bucket):
    return split_result[method][bucket]


def agg(split_results, method, bucket, metric):
    vals = [get_bucket(r, method, bucket)[metric] for r in split_results
            if get_bucket(r, method, bucket) is not None]
    if not vals:
        return None
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    configs = args.configs
    methods = ["global_cp"] + configs

    print("=" * 70)
    print("Step 0: unit checks")
    print("=" * 70)
    run_unit_checks(verbose=True)

    print()
    print("=" * 70)
    print(f"Step 1: load {args.data_path} once, reused across all {args.n_splits} splits x {len(configs)} configs")
    print("=" * 70)
    d = np.load(args.data_path, allow_pickle=True)
    nele, nazi = int(d["nele"]), int(d["nazi"])
    room = _build_room(d)
    N_flat = d["all_likelihood_maps"].shape[0]
    n_traj = N_flat // FRAMES_PER_TRAJECTORY
    overall_burst_frac = d["burst_active_per_frame"].astype(bool).mean()
    print(f"  {n_traj} trajectories, {N_flat} frames, overall burst-active fraction = {overall_burst_frac:.4f}")
    print(f"  h={args.lcp_bandwidth} (fixed), alpha={args.alpha} (fixed)")
    for cfg in configs:
        names, eps = FEATURE_SETS[cfg]
        print(f"  {cfg:<10s} features={names} eps={eps}")

    print()
    print("=" * 70)
    print(f"Step 2: {args.n_splits} independent random 50/50 trajectory splits")
    print("=" * 70)
    split_results = []
    for i in range(args.n_splits):
        res = run_one_split(d, room, nele, nazi, split_seed=args.seed + i, args=args, configs=configs)
        split_results.append(res)
        act_strs = []
        for method in methods:
            b = res[method]["active"]
            act_strs.append(f"{method}:cov={b['cov']:.3f},area={b['area']:6.1f}" if b else f"{method}:n=0")
        print(f"  split {i:3d} (calib_burst_frac={res['calib_burst_frac']:.4f})  active: " + "  ".join(act_strs))

    print()
    print("=" * 70)
    print("Step 3: final compact table -- mean +/- std across splits")
    print("=" * 70)
    header = (f"{'method':<20s} {'overall_cov':>13s} {'overall_area':>14s} "
              f"{'active_cov':>12s} {'active_area':>13s} {'inactive_cov':>13s} {'inactive_area':>14s}")
    sep = "-" * len(header)
    print(header)
    print(sep)
    out_lines = [header, sep]
    csv_lines = ["method,bucket,coverage_mean,coverage_std,area_mean,area_std,n_splits"]
    for method in methods:
        cells = []
        for bucket in BUCKETS:
            cov = agg(split_results, method, bucket, "cov")
            area = agg(split_results, method, bucket, "area")
            if cov is None:
                cells.append("(no data)")
                cells.append("")
                csv_lines.append(f"{method},{bucket},,,,,0")
            else:
                cells.append(f"{cov[0]:.3f}+/-{cov[1]:.3f}")
                cells.append(f"{area[0]:.1f}+/-{area[1]:.1f}")
                csv_lines.append(f"{method},{bucket},{cov[0]:.4f},{cov[1]:.4f},{area[0]:.2f},{area[1]:.2f},{cov[2]}")
        display_name = "Global CP" if method == "global_cp" else (
            "Original LCP [mean,ptb]" if method == "original" else method)
        line = (f"{display_name:<20s} {cells[0]:>13s} {cells[1]:>14s} "
                f"{cells[2]:>12s} {cells[3]:>13s} {cells[4]:>13s} {cells[5]:>14s}")
        print(line)
        out_lines.append(line)

    print()
    print("=" * 70)
    print("Step 4: per-split differences, active bucket")
    print("=" * 70)
    out_lines += ["", "Per-split differences, active bucket:"]

    def diff_report(method_a, method_b, label):
        """method_a - method_b, active coverage and area, per split."""
        cov_a = [get_bucket(r, method_a, "active")["cov"] for r in split_results
                 if get_bucket(r, method_a, "active") and get_bucket(r, method_b, "active")]
        cov_b = [get_bucket(r, method_b, "active")["cov"] for r in split_results
                 if get_bucket(r, method_a, "active") and get_bucket(r, method_b, "active")]
        area_a = [get_bucket(r, method_a, "active")["area"] for r in split_results
                  if get_bucket(r, method_a, "active") and get_bucket(r, method_b, "active")]
        area_b = [get_bucket(r, method_b, "active")["area"] for r in split_results
                  if get_bucket(r, method_a, "active") and get_bucket(r, method_b, "active")]
        if not cov_a:
            return
        diffs_cov = np.array(cov_a) - np.array(cov_b)
        diffs_area = np.array(area_a) - np.array(area_b)
        n_win = int(np.sum(diffs_cov > 0))
        n = len(diffs_cov)
        win_area_diff = diffs_area[diffs_cov > 0]
        line1 = f"  [{label}] coverage diff: mean={diffs_cov.mean():+.4f} std={diffs_cov.std():.4f} (n={n})"
        line2 = f"  [{label}] area diff:     mean={diffs_area.mean():+.2f} std={diffs_area.std():.2f} (n={n})"
        line3 = (f"  [{label}] beats {method_b} on active coverage in {n_win}/{n} splits"
                  + (f"; on those wins, mean area diff = {win_area_diff.mean():+.2f}" if n_win else ""))
        print(line1)
        print(line2)
        print(line3)
        out_lines.extend([line1, line2, line3])

    print("-- vs Global CP --")
    out_lines.append("-- vs Global CP --")
    for cfg in configs:
        diff_report(cfg, "global_cp", f"{cfg} - global_cp")

    if "original" in configs:
        print("-- vs Original LCP --")
        out_lines.append("-- vs Original LCP --")
        for cfg in configs:
            if cfg != "original":
                diff_report(cfg, "original", f"{cfg} - original")

    out_txt = os.path.join(args.out_dir, "results_lcp_feature_ablation.txt")
    with open(out_txt, "w") as fh:
        fh.write("\n".join(out_lines) + "\n")
    out_csv = os.path.join(args.out_dir, "lcp_feature_ablation_summary.csv")
    with open(out_csv, "w") as fh:
        fh.write("\n".join(csv_lines) + "\n")
    out_json = os.path.join(args.out_dir, "lcp_feature_ablation_raw.json")
    with open(out_json, "w") as fh:
        json.dump(dict(args=vars(args), split_results=split_results), fh, indent=2)
    print(f"\nSaved -> {out_txt}")
    print(f"Saved -> {out_csv}")
    print(f"Saved -> {out_json}")


if __name__ == "__main__":
    main()
