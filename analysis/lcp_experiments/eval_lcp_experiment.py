"""
Minimal first Localized Conformal Prediction (LCP) experiment: speaker-wise LCP
(Algorithm 7.8, exact localized weighting + recalibration) vs the existing
global CP, on the same calibration source and test condition. Does NOT touch
the tracker, the global CP implementation (Code/crc_ssl.py), or the npz export
pipeline -- see Code/two_speaker_tracking/lcp.py's module docstring for the
full design (context features, score construction, binary_fill_holes caveat).
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..")))

import argparse
import os

import numpy as np

from Code.crc_ssl import CoverageSet
from Code.utilities import normalize
from Code.two_speaker_tracking.npz_adapter import calibrate_lambda_thresholds, radians_to_grid_index
from Code.two_speaker_tracking.lcp import (
    compute_raw_context_features, fit_standardizer, standardize_features,
    compute_calibration_scores, widest_path_score_map, localized_cp_decision,
    run_unit_checks,
)

DATA_ROOT = "/src/data"
DEFAULT_CALIB = f"{DATA_ROOT}/npz_output_tracking_no_burst_calib/Reverb_400_ms_SNR_15_dB/speakers_2_flat.npz"
DEFAULT_TEST = f"{DATA_ROOT}/npz_output_tracking_burst_test/Reverb_400_ms_SNR_15_dB/speakers_2_flat.npz"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calib_path", default=DEFAULT_CALIB)
    p.add_argument("--test_path", default=DEFAULT_TEST)
    p.add_argument("--n_calib_frames", type=int, default=500)
    p.add_argument("--n_test_frames", type=int, default=200,
                    help="random subsample of valid test frames (test file is always a "
                         "different file from calib, so this is trajectory-safe by construction)")
    p.add_argument("--lambda_steps", type=int, default=500)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--lcp_bandwidth", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_examples_to_print", type=int, default=8)
    p.add_argument("--out_dir", default="Results")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print("Step 0: unit checks")
    print("=" * 70)
    run_unit_checks(verbose=True)

    print()
    print("=" * 70)
    print(f"Step 1: load + subsample calibration ({args.calib_path})")
    print("=" * 70)
    dc = np.load(args.calib_path, allow_pickle=True)
    nele, nazi = int(dc["nele"]), int(dc["nazi"])
    lm_calib_all = dc["all_likelihood_maps"]
    est_calib_all = dc["all_estimated_positions"]
    true_calib_all = dc["speaker_pos"]
    n_calib_total, K = lm_calib_all.shape[:2]

    calib_idx = np.sort(rng.choice(n_calib_total, size=min(args.n_calib_frames, n_calib_total), replace=False))
    lm_calib = lm_calib_all[calib_idx]
    est_calib = est_calib_all[calib_idx]
    true_calib = true_calib_all[calib_idx]
    print(f"  n_calib used: {len(calib_idx)} / {n_calib_total} frames, K={K} speakers")

    print()
    print("Step 2: global lambda (existing, unchanged calibrate_lambda_thresholds)")
    lambdas_by_alpha = calibrate_lambda_thresholds(
        args.calib_path, [args.alpha], lambda_steps=args.lambda_steps,
        n_calib_frames=args.n_calib_frames, seed=args.seed)
    lambdas_global = lambdas_by_alpha[args.alpha]
    print(f"  global lambdas (per speaker): {lambdas_global}")

    print()
    print("Step 3: LCP calibration -- raw context features + per-example scores S_i")
    X_calib_raw = compute_raw_context_features(lm_calib)  # (n_calib, K, 2)
    lambda_list = np.linspace(0.0, 1.0, args.lambda_steps)
    S_calib = compute_calibration_scores(lm_calib, est_calib, true_calib, lambda_list, nele, nazi)  # (n_calib, K)
    standardizers = [fit_standardizer(X_calib_raw[:, k, :]) for k in range(K)]
    for k in range(K):
        mean_k, std_k = standardizers[k]
        print(f"  speaker {k}: X mean={mean_k}, std={std_k}, "
              f"S_i range=[{np.nanmin(S_calib[:, k]):.4f}, {np.nanmax(S_calib[:, k]):.4f}]")

    print()
    print("=" * 70)
    print(f"Step 4: load test condition ({args.test_path})")
    print("=" * 70)
    dt = np.load(args.test_path, allow_pickle=True)
    lm_test_all = dt["all_likelihood_maps"]
    est_test_all = dt["all_estimated_positions"]
    true_test_all = dt["speaker_pos"]
    burst_active_all = dt["burst_active_per_frame"].astype(bool)
    gt_valid_all = dt["gt_valid_per_frame"].astype(bool)

    valid_idx = np.where(gt_valid_all)[0]
    test_idx = np.sort(rng.choice(valid_idx, size=min(args.n_test_frames, len(valid_idx)), replace=False))
    print(f"  n_test frames used: {len(test_idx)} / {len(valid_idx)} valid frames")

    print()
    print("=" * 70)
    print("Step 5: per-frame/speaker global CP vs LCP")
    print("=" * 70)

    records = []
    for t in test_idx:
        true_order, est_order = CoverageSet._match_estimated_to_source(true_test_all[t], est_test_all[t])
        for true_s, est_s in zip(true_order, est_order):
            k = int(est_s)
            raw_map = lm_test_all[t, k]
            norm_map = normalize(raw_map)
            seed = tuple(radians_to_grid_index(est_test_all[t, k], nele, nazi).astype(int))
            true_idx_grid = tuple(radians_to_grid_index(true_test_all[t, true_s], nele, nazi).astype(int))

            # --- global CP (existing mechanism, read-only reuse) ---
            global_region = CoverageSet.neighbours_coverage_set(norm_map, float(lambdas_global[k]),
                                                                 estimated_position=seed)
            global_covered = bool(global_region[true_idx_grid])
            global_area = int(global_region.sum())

            # --- LCP ---
            X_test_raw = compute_raw_context_features(raw_map[None, ...])[0]  # (2,)
            mean_k, std_k = standardizers[k]
            X_test_std = standardize_features(X_test_raw, mean_k, std_k)
            X_calib_std_k = standardize_features(X_calib_raw[:, k, :], mean_k, std_k)

            Lambda_map = widest_path_score_map(norm_map, seed)
            S_test_map = -Lambda_map

            valid_calib_mask = np.isfinite(S_calib[:, k])
            region_lcp, tildeS_test_map, q_map, _w = localized_cp_decision(
                S_calib[valid_calib_mask, k], X_calib_std_k[valid_calib_mask],
                S_test_map, X_test_std, args.lcp_bandwidth, args.alpha)

            lcp_covered = bool(region_lcp[true_idx_grid])
            lcp_area = int(region_lcp.sum())

            records.append(dict(
                frame=int(t), speaker=k, burst_active=bool(burst_active_all[t]),
                raw_mean=float(X_test_raw[0]), raw_ptb=float(X_test_raw[1]),
                global_lambda=float(lambdas_global[k]), global_covered=global_covered, global_area=global_area,
                lcp_decision_stat_at_true=float(tildeS_test_map[true_idx_grid]),
                lcp_threshold_at_true=float(q_map[true_idx_grid]),
                lcp_covered=lcp_covered, lcp_area=lcp_area,
            ))

    print(f"  processed {len(records)} (frame, speaker) records")

    print()
    print("=" * 70)
    print("Step 6: coverage / area report")
    print("=" * 70)

    def bucket_stats(recs, key_area, key_cov):
        if not recs:
            return None
        cov = np.mean([r[key_cov] for r in recs])
        area = np.mean([r[key_area] for r in recs])
        return cov, area, len(recs)

    overall = records
    active = [r for r in records if r["burst_active"]]
    inactive = [r for r in records if not r["burst_active"]]

    header = f"{'bucket':<10s} {'n':>5s} {'global_cov':>11s} {'lcp_cov':>9s} {'global_area':>12s} {'lcp_area':>9s}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    rows_txt = [header, sep]
    for name, recs in [("overall", overall), ("active", active), ("inactive", inactive)]:
        stats = bucket_stats(recs, "global_area", "global_covered")
        stats_lcp = bucket_stats(recs, "lcp_area", "lcp_covered")
        if stats is None:
            row = f"{name:<10s} {'(no frames in this bucket)':>50s}"
        else:
            gcov, garea, n = stats
            _, larea, _ = stats_lcp
            lcov = np.mean([r["lcp_covered"] for r in recs])
            row = f"{name:<10s} {n:5d} {gcov:11.3f} {lcov:9.3f} {garea:12.1f} {larea:9.1f}"
        print(row)
        rows_txt.append(row)

    print()
    print("=" * 70)
    print(f"Step 7: {args.n_examples_to_print} example rows")
    print("=" * 70)
    example_records = records[:args.n_examples_to_print]
    ex_header = (f"{'frame':>6s} {'spk':>3s} {'active':>6s} {'raw_mean':>9s} {'raw_ptb':>8s} "
                 f"{'g_lambda':>9s} {'lcp_stat':>9s} {'lcp_thresh':>10s} {'g_area':>7s} {'lcp_area':>8s} "
                 f"{'g_cov':>6s} {'lcp_cov':>8s}")
    print(ex_header)
    print("-" * len(ex_header))
    ex_rows_txt = [ex_header, "-" * len(ex_header)]
    for r in example_records:
        row = (f"{r['frame']:6d} {r['speaker']:3d} {str(r['burst_active']):>6s} "
               f"{r['raw_mean']:9.4f} {r['raw_ptb']:8.2f} {r['global_lambda']:9.4f} "
               f"{r['lcp_decision_stat_at_true']:9.4f} {r['lcp_threshold_at_true']:10.4f} "
               f"{r['global_area']:7d} {r['lcp_area']:8d} "
               f"{str(r['global_covered']):>6s} {str(r['lcp_covered']):>8s}")
        print(row)
        ex_rows_txt.append(row)

    out_txt = os.path.join(args.out_dir, "results_lcp_experiment_burst_test.txt")
    with open(out_txt, "w") as fh:
        fh.write("\n".join(rows_txt) + "\n\n" + "\n".join(ex_rows_txt) + "\n")
    print(f"\nSaved -> {out_txt}")


if __name__ == "__main__":
    main()
