"""
Global CP vs LCP (X = estimated speaker separation Delta_hat), repeated
scene-disjoint splits, on the paired angular-separation pilot dataset
(D = 5,10,15,20,30,45,60 deg, 25 paired base scenes).

Question this answers: does the EXISTING Algorithm-7.8 LCP implementation
(Code/two_speaker_tracking/lcp.py, unmodified) improve conditional
coverage/area when the context feature is the single scalar

    X_t = Delta_hat_t = great-circle separation between the two
          ESTIMATED speaker DOAs in frame t (all_estimated_positions[t,0]
          and [t,1]) -- no GT, no GT-based matching/reordering, both
          speaker records in a frame share the same X_t.

This is a two-diagnostics follow-up: diagnostic 1 (informativeness of
Delta_hat about true D) and diagnostic 2 (does P(S|Delta_hat) look
heterogeneous) both supported trying this. This script is the actual
Global-vs-LCP comparison they were building up to.

Reuses, unmodified:
  - eval_angular_separation.py: load_conditions, pool_scenes (scene-disjoint,
    D-paired sampling -- exact same split logic already validated there)
  - Code/two_speaker_tracking/lcp.py: compute_calibration_scores,
    calibrate_global_lambda_from_arrays, fit_standardizer,
    standardize_features, widest_path_score_map, localized_cp_decision,
    run_unit_checks -- the Algorithm 7.8 implementation itself
  - Code/crc_ssl.py: CoverageSet (region growing, GT<->estimated matching
    used ONLY for scoring/coverage checks, never for constructing X)

Does not modify lcp.py, crc_ssl.py, eval_angular_separation.py, the
tracker, or the npz export pipeline. Does not clip/cap/transform
Delta_hat and does not special-case the >63deg tail.
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))

import argparse
import json
import os
import time

import numpy as np

from Code.crc_ssl import CoverageSet
from Code.utilities import normalize
from Code.two_speaker_tracking.npz_adapter import radians_to_grid_index, _build_room
from Code.two_speaker_tracking.lcp import (
    compute_calibration_scores, calibrate_global_lambda_from_arrays,
    fit_standardizer, standardize_features, widest_path_score_map,
    localized_cp_decision, run_unit_checks,
)
from eval_angular_separation import load_conditions, pool_scenes, DEFAULT_SEPARATIONS

BASE = "/home/dsi/mayavb/PythonProjects/SRP-DNN/data/doa_sep_pilot_v2"
SEPARATIONS = DEFAULT_SEPARATIONS
DATA_PATHS = [f"{BASE}/D{int(D)}/Reverb_400_ms_SNR_15_dB/speakers_2_flat.npz" for D in SEPARATIONS]

# Same bin edges as Diagnostic 2, unchanged -- do NOT retune based on LCP results.
DELTA_BIN_EDGES = [0, 13, 18.5, 23.5, 33, 43, 53, 63, 200]
DELTA_BIN_LABELS = [f"[{DELTA_BIN_EDGES[i]},{DELTA_BIN_EDGES[i + 1]})" for i in range(len(DELTA_BIN_EDGES) - 1)]


def great_circle_sep_deg(est_pos_rad):
    """est_pos_rad: (n, K=2, 2) radians [polar-elevation, azimuth]. Returns (n,) degrees.
    Uses ONLY the two estimated positions -- no GT, no speaker matching/reordering
    (separation is symmetric under swapping which estimate is "0" vs "1")."""
    th0, phi0 = est_pos_rad[:, 0, 0], est_pos_rad[:, 0, 1]
    th1, phi1 = est_pos_rad[:, 1, 0], est_pos_rad[:, 1, 1]
    cos_delta = np.cos(th0) * np.cos(th1) + np.sin(th0) * np.sin(th1) * np.cos(phi0 - phi1)
    return np.degrees(np.arccos(np.clip(cos_delta, -1.0, 1.0)))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_splits", type=int, default=20)
    p.add_argument("--seed", type=int, default=0, help="split i uses seed+i, i.e. seeds 0..n_splits-1")
    p.add_argument("--calib_scene_frac", type=float, default=0.5)
    # Matches the ACTUAL calibration/test sampling used in the validated
    # eval_angular_separation.py run (its run_one_split / Step 2.3), i.e.
    # 10 frames/scene -- NOT the 25 frames/scene used there for the separate,
    # unrelated Step 2.1/2.2 descriptive E/V table. Preserved exactly per
    # instructions ("do not silently increase the effective calibration
    # sample size").
    p.add_argument("--n_calib_frames_per_scene", type=int, default=10)
    p.add_argument("--n_test_frames_per_scene", type=int, default=10)
    p.add_argument("--lambda_steps", type=int, default=500)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--lcp_bandwidth", type=float, default=1.0)
    p.add_argument("--out_dir", default="Results")
    p.add_argument("--worker_split", type=int, default=None,
                    help="Internal: compute ONLY this one split index (0-based), save its result to "
                         "<out_dir>/partial_split_<i>.json, and exit -- lets the n_splits repeats run as "
                         "independent parallel processes (each split is CPU-bound, embarrassingly "
                         "parallel, no shared state). Omit for normal sequential/aggregating mode, which "
                         "reuses any partial_split_<i>.json already on disk instead of recomputing it.")
    return p.parse_args()


def run_one_split(conds, separations, common_scenes, room, nele, nazi, lambda_list, args, split_seed):
    rng = np.random.default_rng(split_seed)
    scene_order = rng.permutation(common_scenes)
    n_calib_scenes = int(round(len(scene_order) * args.calib_scene_frac))
    calib_scenes, test_scenes = scene_order[:n_calib_scenes], scene_order[n_calib_scenes:]

    # Sanity C: calib/test base_scene_id sets disjoint.
    assert len(set(calib_scenes.tolist()) & set(test_scenes.tolist())) == 0, \
        "a base_scene_id leaked into both calib and test -- pairing invariant violated"

    # pool_scenes (reused, unmodified from eval_angular_separation.py) pulls ALL
    # `separations` for every scene in the given scene_ids -- sanity D: all 7 D
    # conditions for a base scene are on the same side of the split, by construction.
    lm_c, est_c, true_c, D_c = pool_scenes(conds, separations, calib_scenes,
                                            args.n_calib_frames_per_scene, rng)
    lm_t, est_t, true_t, D_t = pool_scenes(conds, separations, test_scenes,
                                            args.n_test_frames_per_scene, rng)
    K = lm_c.shape[1]

    # ---- Global CP: calibrated from the calib pool only ----
    lambdas_global = calibrate_global_lambda_from_arrays(lm_c, est_c, true_c, room, lambda_list, args.alpha)

    # ---- LCP: X, standardizer, S_calib -- all from the calib pool only ----
    # Sanity A: Delta_hat uses ONLY all_estimated_positions, no GT anywhere in this call.
    delta_hat_c = great_circle_sep_deg(est_c)
    delta_hat_t = great_circle_sep_deg(est_t)

    # Sanity F: both speaker slots in a frame get the identical Delta_hat value.
    X_calib = np.repeat(delta_hat_c[:, None], K, axis=1)[..., None]  # (n_c, K, 1)

    # Sanity B: fit_standardizer sees ONLY calibration X.
    standardizers = [fit_standardizer(X_calib[:, k, :]) for k in range(K)]

    # Unmodified reuse of the existing calibration-score construction (water-filling
    # lambda_star search via CoverageSet.neighbours_coverage_set, GT used here only to
    # know which true position each estimated slot's score is checked against -- never
    # to construct X).
    S_calib = compute_calibration_scores(lm_c, est_c, true_c, lambda_list, nele, nazi)

    records = []
    for i in range(lm_t.shape[0]):
        # GT-matching used ONLY to know which true position to score/check coverage
        # against for THIS record -- never touches X (Delta_hat), computed above from
        # est-only arrays before this loop even starts.
        true_order, est_order = CoverageSet._match_estimated_to_source(true_t[i], est_t[i])
        for true_s, est_s in zip(true_order, est_order):
            k = int(est_s)
            raw_map = lm_t[i, k]
            norm_map = normalize(raw_map)
            seed = tuple(radians_to_grid_index(est_t[i, k], nele, nazi).astype(int))
            true_idx_grid = tuple(radians_to_grid_index(true_t[i, true_s], nele, nazi).astype(int))

            # Global CP -- sanity E: identical (lm_t[i,k], seed, true_idx_grid) used for LCP below.
            global_region = CoverageSet.neighbours_coverage_set(
                norm_map, float(lambdas_global[k]), estimated_position=seed)
            global_covered = bool(global_region[true_idx_grid])
            global_area = int(global_region.sum())

            # LCP -- sanity G: speaker-specific S_calib[:,k]/standardizer[k]/decision call.
            X_test_raw = np.array([delta_hat_t[i]])
            mean_k, std_k = standardizers[k]
            X_test_std = standardize_features(X_test_raw, mean_k, std_k)
            X_calib_std_k = standardize_features(X_calib[:, k, :], mean_k, std_k)

            Lambda_map = widest_path_score_map(norm_map, seed)
            S_test_map = -Lambda_map

            valid_mask = np.isfinite(S_calib[:, k])
            # Sanity H: args.lcp_bandwidth fixed at 1.0, passed through unchanged, no search.
            region_lcp, _, _, _ = localized_cp_decision(
                S_calib[valid_mask, k], X_calib_std_k[valid_mask],
                S_test_map, X_test_std, args.lcp_bandwidth, args.alpha)

            lcp_covered = bool(region_lcp[true_idx_grid])
            lcp_area = int(region_lcp.sum())

            records.append(dict(D=float(D_t[i]), delta_hat=float(delta_hat_t[i]),
                                 global_covered=global_covered, global_area=global_area,
                                 lcp_covered=lcp_covered, lcp_area=lcp_area))

    by_D = {}
    for D in separations:
        recs = [r for r in records if r["D"] == D]
        if recs:
            by_D[D] = dict(
                n=len(recs),
                global_cov=float(np.mean([r["global_covered"] for r in recs])),
                global_area=float(np.mean([r["global_area"] for r in recs])),
                lcp_cov=float(np.mean([r["lcp_covered"] for r in recs])),
                lcp_area=float(np.mean([r["lcp_area"] for r in recs])),
            )

    overall = dict(
        n=len(records),
        global_cov=float(np.mean([r["global_covered"] for r in records])),
        global_area=float(np.mean([r["global_area"] for r in records])),
        lcp_cov=float(np.mean([r["lcp_covered"] for r in records])),
        lcp_area=float(np.mean([r["lcp_area"] for r in records])),
    )

    return dict(
        n_calib_scenes=len(calib_scenes), n_test_scenes=len(test_scenes),
        n_calib_frames=lm_c.shape[0], n_calib_records=lm_c.shape[0] * K,
        n_test_frames=lm_t.shape[0], n_test_records=len(records),
        by_D=by_D, overall=overall, records=records,
    )


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    print("=" * 100)
    print("Step 0: unit checks (Algorithm 7.8 implementation, read-only reuse)")
    print("=" * 100)
    run_unit_checks(verbose=True)

    print()
    print("=" * 100)
    print("Step 1: load 7 conditions, find paired scenes")
    print("=" * 100)
    conds, nele, nazi, common_scenes = load_conditions(DATA_PATHS, SEPARATIONS)
    room = _build_room(conds[SEPARATIONS[0]])
    lambda_list = np.linspace(0.0, 1.0, args.lambda_steps)
    n_calib_scenes_expected = int(round(len(common_scenes) * args.calib_scene_frac))
    n_test_scenes_expected = len(common_scenes) - n_calib_scenes_expected
    print(f"  {len(common_scenes)} common base scenes -> {n_calib_scenes_expected} calib / "
          f"{n_test_scenes_expected} test scenes per split (deterministic given calib_scene_frac)")
    print(f"  calibration sampling convention: {args.n_calib_frames_per_scene} frames/scene "
          f"(matches the validated eval_angular_separation.py run's Step 2.3 convention, "
          f"NOT the unrelated 25 frames/scene used there only for the descriptive E/V table)")
    print(f"  expected calib frames/split = {len(SEPARATIONS)} D x {n_calib_scenes_expected} scenes x "
          f"{args.n_calib_frames_per_scene} = {len(SEPARATIONS) * n_calib_scenes_expected * args.n_calib_frames_per_scene}"
          f"  (x K=2 speaker-records = "
          f"{len(SEPARATIONS) * n_calib_scenes_expected * args.n_calib_frames_per_scene * 2})")
    print(f"  expected test frames/split  = {len(SEPARATIONS)} x {n_test_scenes_expected} x "
          f"{args.n_test_frames_per_scene} = {len(SEPARATIONS) * n_test_scenes_expected * args.n_test_frames_per_scene}"
          f"  (x K=2 speaker-records = "
          f"{len(SEPARATIONS) * n_test_scenes_expected * args.n_test_frames_per_scene * 2})")

    def partial_path(i):
        return os.path.join(args.out_dir, f"partial_split_{i}.json")

    def save_partial(i, res):
        # by_D keys are floats -> stringify for JSON, restored on load.
        payload = dict(res)
        payload["by_D"] = {str(k): v for k, v in res["by_D"].items()}
        with open(partial_path(i), "w") as fh:
            json.dump(payload, fh)

    def load_partial(i):
        with open(partial_path(i)) as fh:
            payload = json.load(fh)
        payload["by_D"] = {float(k): v for k, v in payload["by_D"].items()}
        return payload

    # Worker mode: compute exactly one split, save it, exit -- lets n_splits
    # run as independent parallel OS processes (each split is CPU-bound and
    # shares no state with the others).
    if args.worker_split is not None:
        i = args.worker_split
        tS = time.time()
        res = run_one_split(conds, SEPARATIONS, common_scenes, room, nele, nazi, lambda_list,
                             args, split_seed=args.seed + i)
        save_partial(i, res)
        ov = res["overall"]
        print(f"  [worker split {i}] n={ov['n']} global_cov={ov['global_cov']:.3f} lcp_cov={ov['lcp_cov']:.3f} "
              f"({time.time() - tS:.1f}s) -> {partial_path(i)}")
        return

    print()
    print("=" * 100)
    print(f"Step 2: {args.n_splits} repeated scene-disjoint splits, Global CP vs LCP (h={args.lcp_bandwidth}, alpha={args.alpha})")
    print("=" * 100)
    split_results = []
    for i in range(args.n_splits):
        tS = time.time()
        if os.path.exists(partial_path(i)):
            res = load_partial(i)
            print(f"  split {i:3d}  (loaded from {partial_path(i)})", end="  ")
        else:
            res = run_one_split(conds, SEPARATIONS, common_scenes, room, nele, nazi, lambda_list,
                                 args, split_seed=args.seed + i)
            save_partial(i, res)
            print(f"  split {i:3d} ", end="  ")
        split_results.append(res)
        if i == 0:
            print(f"\n  [split 0 confirmed] calib_scenes={res['n_calib_scenes']} test_scenes={res['n_test_scenes']} "
                  f"calib_frames={res['n_calib_frames']} calib_records={res['n_calib_records']} "
                  f"test_frames={res['n_test_frames']} test_records={res['n_test_records']}")
        ov = res["overall"]
        print(f"n={ov['n']:5d}  global_cov={ov['global_cov']:.3f} lcp_cov={ov['lcp_cov']:.3f}  "
              f"global_area={ov['global_area']:7.1f} lcp_area={ov['lcp_area']:7.1f}  ({time.time() - tS:.1f}s)")

    print(f"\n  Step 2 total: {time.time() - t0:.1f}s")

    # -----------------------------------------------------------------
    # Section 7: primary report, by TRUE commanded D
    # -----------------------------------------------------------------
    print()
    print("=" * 100)
    print("Section 7: results by TRUE commanded D (mean +/- std across splits)")
    print("=" * 100)
    header = (f"{'D':>5s} {'global_cov':>11s} {'lcp_cov':>11s} {'d_cov(LCP-G)':>13s} {'n_split_LCP>G':>14s} "
              f"{'global_area':>12s} {'lcp_area':>12s} {'d_area(LCP-G)':>14s}")
    print(header)
    print("-" * len(header))
    out_lines = [header, "-" * len(header)]

    by_D_summary = {}
    for D in SEPARATIONS:
        gcovs = [r["by_D"][D]["global_cov"] for r in split_results if D in r["by_D"]]
        lcovs = [r["by_D"][D]["lcp_cov"] for r in split_results if D in r["by_D"]]
        gareas = [r["by_D"][D]["global_area"] for r in split_results if D in r["by_D"]]
        lareas = [r["by_D"][D]["lcp_area"] for r in split_results if D in r["by_D"]]
        d_cov = [l - g for l, g in zip(lcovs, gcovs)]
        d_area = [l - g for l, g in zip(lareas, gareas)]
        n_lcp_better = sum(1 for x in d_cov if x > 0)

        by_D_summary[D] = dict(
            global_cov_mean=float(np.mean(gcovs)), global_cov_std=float(np.std(gcovs)),
            lcp_cov_mean=float(np.mean(lcovs)), lcp_cov_std=float(np.std(lcovs)),
            global_area_mean=float(np.mean(gareas)), global_area_std=float(np.std(gareas)),
            lcp_area_mean=float(np.mean(lareas)), lcp_area_std=float(np.std(lareas)),
            d_cov_mean=float(np.mean(d_cov)), d_cov_std=float(np.std(d_cov)),
            d_area_mean=float(np.mean(d_area)), d_area_std=float(np.std(d_area)),
            n_splits_lcp_better_cov=n_lcp_better, n_splits=len(gcovs),
        )
        line = (f"{D:5.1f} "
                f"{np.mean(gcovs):6.3f}+-{np.std(gcovs):<4.3f} "
                f"{np.mean(lcovs):6.3f}+-{np.std(lcovs):<4.3f} "
                f"{np.mean(d_cov):+7.4f}   "
                f"{n_lcp_better:3d}/{len(gcovs):<3d}      "
                f"{np.mean(gareas):7.1f}+-{np.std(gareas):<6.1f} "
                f"{np.mean(lareas):7.1f}+-{np.std(lareas):<6.1f} "
                f"{np.mean(d_area):+8.2f}")
        print(line)
        out_lines.append(line)

    # overall / marginal
    gcovs_o = [r["overall"]["global_cov"] for r in split_results]
    lcovs_o = [r["overall"]["lcp_cov"] for r in split_results]
    gareas_o = [r["overall"]["global_area"] for r in split_results]
    lareas_o = [r["overall"]["lcp_area"] for r in split_results]
    d_cov_o = [l - g for l, g in zip(lcovs_o, gcovs_o)]
    d_area_o = [l - g for l, g in zip(lareas_o, gareas_o)]
    n_lcp_better_o = sum(1 for x in d_cov_o if x > 0)
    overall_summary = dict(
        global_cov_mean=float(np.mean(gcovs_o)), global_cov_std=float(np.std(gcovs_o)),
        lcp_cov_mean=float(np.mean(lcovs_o)), lcp_cov_std=float(np.std(lcovs_o)),
        global_area_mean=float(np.mean(gareas_o)), global_area_std=float(np.std(gareas_o)),
        lcp_area_mean=float(np.mean(lareas_o)), lcp_area_std=float(np.std(lareas_o)),
        d_cov_mean=float(np.mean(d_cov_o)), d_cov_std=float(np.std(d_cov_o)),
        d_area_mean=float(np.mean(d_area_o)), d_area_std=float(np.std(d_area_o)),
        n_splits_lcp_better_cov=n_lcp_better_o, n_splits=len(gcovs_o),
    )
    line = (f"{'ALL':>5s} "
            f"{np.mean(gcovs_o):6.3f}+-{np.std(gcovs_o):<4.3f} "
            f"{np.mean(lcovs_o):6.3f}+-{np.std(lcovs_o):<4.3f} "
            f"{np.mean(d_cov_o):+7.4f}   "
            f"{n_lcp_better_o:3d}/{len(gcovs_o):<3d}      "
            f"{np.mean(gareas_o):7.1f}+-{np.std(gareas_o):<6.1f} "
            f"{np.mean(lareas_o):7.1f}+-{np.std(lareas_o):<6.1f} "
            f"{np.mean(d_area_o):+8.2f}")
    print(line)
    out_lines.append(line)

    # -----------------------------------------------------------------
    # Section 8: secondary report, by Delta_hat bin, pooled across splits
    # -----------------------------------------------------------------
    print()
    print("=" * 100)
    print("Section 8: results by Delta_hat bin (SAME edges as Diagnostic 2, unchanged, incl. tail)")
    print("  Aggregation: records pooled across all 20 splits' test sets (per-split counts for the")
    print("  rarer bins are too small to average split-by-split; pooling is stated explicitly here).")
    print("=" * 100)
    all_records = [r for res in split_results for r in res["records"]]
    all_delta = np.array([r["delta_hat"] for r in all_records])
    all_gcov = np.array([r["global_covered"] for r in all_records])
    all_lcov = np.array([r["lcp_covered"] for r in all_records])
    all_garea = np.array([r["global_area"] for r in all_records])
    all_larea = np.array([r["lcp_area"] for r in all_records])
    bin_idx = np.digitize(all_delta, DELTA_BIN_EDGES) - 1

    header2 = (f"{'bin':<14s} {'n':>6s} {'global_cov':>11s} {'lcp_cov':>11s} "
               f"{'global_area_mn':>15s} {'lcp_area_mn':>13s} {'global_area_md':>15s} {'lcp_area_md':>13s}")
    print(header2)
    print("-" * len(header2))
    out_lines2 = [header2, "-" * len(header2)]
    bin_summary = {}
    for b, label in enumerate(DELTA_BIN_LABELS):
        m = bin_idx == b
        n = int(m.sum())
        if n == 0:
            line = f"{label:<14s}  (no data)"
            print(line)
            out_lines2.append(line)
            continue
        bin_summary[label] = dict(
            n=n, global_cov=float(all_gcov[m].mean()), lcp_cov=float(all_lcov[m].mean()),
            global_area_mean=float(all_garea[m].mean()), lcp_area_mean=float(all_larea[m].mean()),
            global_area_median=float(np.median(all_garea[m])), lcp_area_median=float(np.median(all_larea[m])),
        )
        line = (f"{label:<14s} {n:6d} {all_gcov[m].mean():11.3f} {all_lcov[m].mean():11.3f} "
                f"{all_garea[m].mean():15.1f} {all_larea[m].mean():13.1f} "
                f"{np.median(all_garea[m]):15.1f} {np.median(all_larea[m]):13.1f}")
        print(line)
        out_lines2.append(line)

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------
    out_txt = os.path.join(args.out_dir, "results_lcp_angular_separation.txt")
    with open(out_txt, "w") as fh:
        fh.write("Section 7: by true commanded D\n")
        fh.write("\n".join(out_lines) + "\n\n")
        fh.write("Section 8: by Delta_hat bin (pooled across splits)\n")
        fh.write("\n".join(out_lines2) + "\n")
    print(f"\nSaved text -> {out_txt}")

    out_json = os.path.join(args.out_dir, "lcp_angular_separation_raw.json")
    with open(out_json, "w") as fh:
        json.dump(dict(
            args=vars(args),
            by_D_summary={str(k): v for k, v in by_D_summary.items()},
            overall_summary=overall_summary,
            bin_summary=bin_summary,
        ), fh, indent=2)
    print(f"Saved raw json -> {out_json}")

    print(f"\nTOTAL WALL TIME: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
