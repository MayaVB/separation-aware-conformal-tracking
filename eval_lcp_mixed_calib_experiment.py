"""
Mixed-calibration LCP experiment.

Calibrates BOTH the existing global CP and LCP (Algorithm 7.8) on a
calibration pool drawn from npz_output_tracking_burst_test itself (not the
clean-only no_burst_calib file), split TRAJECTORY-level from the test pool
within that same file. The burst/non-burst label is never given to either
algorithm -- only the raw context features [raw_mean, raw_peak_to_background]
(Code/two_speaker_tracking/lcp.py) are used for localization. This is the
direct comparison to eval_lcp_experiment.py's clean-calibration result (kept
unchanged, on purpose, as the "cannot rescue an unseen regime" baseline).

Tests:
  - does LCP's burst-active coverage improve now that its calibration pool
    actually contains burst examples?
  - does global CP calibrated on the same mixed pool stay roughly unchanged
    (bursts are ~5% of frames, washed out in one global threshold)?
  - do burst-active test frames actually receive disproportionate kernel
    weight from burst-active calibration neighbors? (post-hoc diagnostic --
    the burst label is used ONLY here, for analysis, never fed to the
    algorithm itself)

Does not touch the tracker, Code/crc_ssl.py, or the npz export pipeline.
"""

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
DEFAULT_MIXED_PATH = f"{DATA_ROOT}/npz_output_tracking_burst_test/Reverb_400_ms_SNR_15_dB/speakers_2_flat.npz"
FRAMES_PER_TRAJECTORY = 104


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mixed_path", default=DEFAULT_MIXED_PATH)
    p.add_argument("--calib_traj_end", type=int, default=50,
                    help="trajectories [0, calib_traj_end) -> calibration pool; "
                         "[calib_traj_end, n_traj) -> test pool")
    p.add_argument("--n_calib_frames", type=int, default=500)
    p.add_argument("--n_test_frames", type=int, default=200)
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
    print(f"Step 1: load {args.mixed_path}, split trajectory-level")
    print("=" * 70)
    d = np.load(args.mixed_path, allow_pickle=True)
    nele, nazi = int(d["nele"]), int(d["nazi"])
    lm_all = d["all_likelihood_maps"]
    est_all = d["all_estimated_positions"]
    true_all = d["speaker_pos"]
    burst_active_all = d["burst_active_per_frame"].astype(bool)
    gt_valid_all = d["gt_valid_per_frame"].astype(bool)
    N_flat, K = lm_all.shape[:2]
    n_traj = N_flat // FRAMES_PER_TRAJECTORY
    assert n_traj * FRAMES_PER_TRAJECTORY == N_flat

    calib_frame_pool = np.arange(0, args.calib_traj_end * FRAMES_PER_TRAJECTORY)
    test_frame_pool = np.arange(args.calib_traj_end * FRAMES_PER_TRAJECTORY, N_flat)
    print(f"  calib pool: trajectories [0, {args.calib_traj_end}) = {len(calib_frame_pool)} frames, "
          f"burst-active fraction = {burst_active_all[calib_frame_pool].mean():.4f}")
    print(f"  test  pool: trajectories [{args.calib_traj_end}, {n_traj}) = {len(test_frame_pool)} frames, "
          f"burst-active fraction = {burst_active_all[test_frame_pool].mean():.4f}")

    calib_idx = np.sort(rng.choice(calib_frame_pool, size=min(args.n_calib_frames, len(calib_frame_pool)),
                                    replace=False))
    lm_calib = lm_all[calib_idx]
    est_calib = est_all[calib_idx]
    true_calib = true_all[calib_idx]
    calib_burst_active = burst_active_all[calib_idx]  # known to us for diagnostics; NEVER fed to the algorithm
    print(f"  n_calib used: {len(calib_idx)}, burst-active among them: {calib_burst_active.sum()} "
          f"({calib_burst_active.mean():.4f})")

    print()
    print("Step 2: global lambda calibrated on this SAME mixed pool (new helper, reused read-only CoverageSet)")
    room = _build_room(d)
    lambda_list = np.linspace(0.0, 1.0, args.lambda_steps)
    lambdas_global = calibrate_global_lambda_from_arrays(lm_calib, est_calib, true_calib, room, lambda_list, args.alpha)
    print(f"  global lambdas (per speaker), mixed-calib: {lambdas_global}")

    print()
    print("Step 3: LCP calibration on the same mixed pool -- context features + scores S_i")
    X_calib_raw = compute_raw_context_features(lm_calib)  # (n_calib, K, 2)
    S_calib = compute_calibration_scores(lm_calib, est_calib, true_calib, lambda_list, nele, nazi)
    standardizers = [fit_standardizer(X_calib_raw[:, k, :]) for k in range(K)]
    for k in range(K):
        mean_k, std_k = standardizers[k]
        print(f"  speaker {k}: X mean={mean_k}, std={std_k}")

    print()
    print("=" * 70)
    print("Step 4: test pool (disjoint trajectories, same file)")
    print("=" * 70)
    valid_test_idx = test_frame_pool[gt_valid_all[test_frame_pool]]
    test_idx = np.sort(rng.choice(valid_test_idx, size=min(args.n_test_frames, len(valid_test_idx)), replace=False))
    print(f"  n_test frames used: {len(test_idx)} / {len(valid_test_idx)} valid frames in the test pool")

    print()
    print("=" * 70)
    print("Step 5: per-frame/speaker global CP vs LCP, plus burst-neighbor-weight diagnostic")
    print("=" * 70)

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

            # Post-hoc diagnostic only: what fraction of this test point's kernel
            # weight mass (over calib neighbors, excluding self) landed on
            # calibration examples that are themselves burst-active?
            calib_burst_mask_valid = calib_burst_active[valid_calib_mask]
            n_valid_calib = calib_burst_mask_valid.shape[0]
            weight_to_calib = w_row[:n_valid_calib]  # exclude the self-weight (last entry)
            weight_to_calib = weight_to_calib / weight_to_calib.sum()  # renormalize over calib-only mass
            mass_to_burst_neighbors = float(weight_to_calib[calib_burst_mask_valid].sum())

            records.append(dict(
                frame=int(t), speaker=k, burst_active=bool(burst_active_all[t]),
                raw_mean=float(X_test_raw[0]), raw_ptb=float(X_test_raw[1]),
                global_lambda=float(lambdas_global[k]), global_covered=global_covered, global_area=global_area,
                lcp_decision_stat_at_true=float(tildeS_test_map[true_idx_grid]),
                lcp_threshold_at_true=float(q_map[true_idx_grid]),
                lcp_covered=lcp_covered, lcp_area=lcp_area,
                mass_to_burst_neighbors=mass_to_burst_neighbors,
            ))

    print(f"  processed {len(records)} (frame, speaker) records")
    base_rate = float(calib_burst_active.mean())

    print()
    print("=" * 70)
    print("Step 6: coverage / area report (mixed calibration)")
    print("=" * 70)

    overall = records
    active = [r for r in records if r["burst_active"]]
    inactive = [r for r in records if not r["burst_active"]]

    header = f"{'bucket':<10s} {'n':>5s} {'global_cov':>11s} {'lcp_cov':>9s} {'global_area':>12s} {'lcp_area':>9s}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    rows_txt = [header, sep]
    for name, recs in [("overall", overall), ("active", active), ("inactive", inactive)]:
        if not recs:
            row = f"{name:<10s} {'(no frames in this bucket)':>50s}"
        else:
            n = len(recs)
            gcov = np.mean([r["global_covered"] for r in recs])
            lcov = np.mean([r["lcp_covered"] for r in recs])
            garea = np.mean([r["global_area"] for r in recs])
            larea = np.mean([r["lcp_area"] for r in recs])
            row = f"{name:<10s} {n:5d} {gcov:11.3f} {lcov:9.3f} {garea:12.1f} {larea:9.1f}"
        print(row)
        rows_txt.append(row)

    print()
    print("=" * 70)
    print("Step 7: burst-neighbor kernel-weight diagnostic")
    print("=" * 70)
    print(f"  calib pool burst-active base rate: {base_rate:.4f}")
    diag_rows = []
    for name, recs in [("active", active), ("inactive", inactive)]:
        if recs:
            mean_mass = np.mean([r["mass_to_burst_neighbors"] for r in recs])
            ratio = mean_mass / base_rate if base_rate > 0 else float("nan")
            line = (f"  {name:<9s} test frames: mean weight-mass to burst-active calib neighbors "
                    f"= {mean_mass:.4f}  (base rate {base_rate:.4f}, ratio {ratio:.2f}x)")
        else:
            line = f"  {name:<9s} test frames: (none)"
        print(line)
        diag_rows.append(line)

    print()
    print("=" * 70)
    print(f"Step 8: {args.n_examples_to_print} example rows")
    print("=" * 70)
    example_records = records[:args.n_examples_to_print]
    ex_header = (f"{'frame':>6s} {'spk':>3s} {'active':>6s} {'raw_mean':>9s} {'raw_ptb':>8s} "
                 f"{'g_lambda':>9s} {'lcp_stat':>9s} {'lcp_thresh':>10s} {'g_area':>7s} {'lcp_area':>8s} "
                 f"{'g_cov':>6s} {'lcp_cov':>8s} {'mass2burst':>10s}")
    print(ex_header)
    print("-" * len(ex_header))
    ex_rows_txt = [ex_header, "-" * len(ex_header)]
    for r in example_records:
        row = (f"{r['frame']:6d} {r['speaker']:3d} {str(r['burst_active']):>6s} "
               f"{r['raw_mean']:9.4f} {r['raw_ptb']:8.2f} {r['global_lambda']:9.4f} "
               f"{r['lcp_decision_stat_at_true']:9.4f} {r['lcp_threshold_at_true']:10.4f} "
               f"{r['global_area']:7d} {r['lcp_area']:8d} "
               f"{str(r['global_covered']):>6s} {str(r['lcp_covered']):>8s} {r['mass_to_burst_neighbors']:10.4f}")
        print(row)
        ex_rows_txt.append(row)

    out_txt = os.path.join(args.out_dir, "results_lcp_mixed_calib_experiment.txt")
    with open(out_txt, "w") as fh:
        fh.write("\n".join(rows_txt) + "\n\n" + "\n".join(diag_rows) + "\n\n" + "\n".join(ex_rows_txt) + "\n")
    print(f"\nSaved -> {out_txt}")


if __name__ == "__main__":
    main()
