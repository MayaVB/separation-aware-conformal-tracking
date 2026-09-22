"""
ORACLE Mondrian CP vs Global CP on the angular-separation pilot dataset,
using the TRUE commanded speaker separation D as the Mondrian category.

This is deliberately an ORACLE diagnostic. True D is unknown at inference
in the real problem -- this script does NOT propose a deployable method.
It answers one question only:

    If we knew the true source-separation regime, would calibrating CP
    separately within each separation group correct the strong
    conditional-coverage heterogeneity observed under Global CP?

Reference Global-CP numbers this is checked against (Results/results_angular_separation.txt):
    D=10: coverage 0.756 (severe undercoverage)
    D=15: coverage 0.828 (undercoverage)
    D=60: coverage 0.982 (overcoverage)
    overall (marginal): ~0.894

Method
------
Global CP:      one CoverageSet calibration pooling calibration frames across
                 ALL D, giving lambda_global[k] (k = speaker slot).
Oracle Mondrian: SEVEN independent CoverageSet calibrations, one per true-D
                 category, each using ONLY that category's calibration
                 frames (same frames Global would have pooled in -- never
                 resampled), giving lambda_mondrian[D, k].

Both are evaluated on the IDENTICAL test pool within each split; the ONLY
difference between the two methods is which calibration records determined
the threshold used for a given test record. No LCP, no RBF, no estimated
separation, no new CP/region-growing logic -- this script only decides
which lambda a test record is scored against, via
Code.crc_ssl.CoverageSet.neighbours_coverage_set (called exactly as
eval_angular_separation.py's run_one_split already calls it) and
Code.two_speaker_tracking.lcp.calibrate_global_lambda_from_arrays (called
exactly as eval_angular_separation.py already calls it, just on a
per-D-filtered calibration subset instead of the pooled one).

Splits are bit-identical to eval_angular_separation.py's run_one_split for
the same split_seed: same np.random.default_rng(split_seed), same
scene_order permutation, same calib/test scene split, same sequence of
pool_scenes() calls (imported unmodified) -- so calib/test scene
membership and the exact frames sampled per (D, scene) match the existing
Global-CP reference run exactly, given identical CLI args.

Finite-sample handling: Code.crc_ssl.CoverageSet._calc_conformal_risk_control
raises ValueError if 1/(n_calib+1) > alpha, i.e. n_calib <= 8 for alpha=0.1.
This is NOT worked around here -- a per-(split, D) Mondrian calibration
that hits this is caught, logged as INFEASIBLE, and that (split, D)
combination is excluded from Mondrian aggregation (Global is unaffected,
since its pooled calibration set is unaffected by the exclusion).
"""

import argparse
import json
import os

import numpy as np

from Code.crc_ssl import CoverageSet
from Code.utilities import normalize
from Code.two_speaker_tracking.npz_adapter import radians_to_grid_index, _build_room
from Code.two_speaker_tracking.lcp import calibrate_global_lambda_from_arrays

from eval_angular_separation import load_conditions, pool_scenes, DEFAULT_SEPARATIONS

# Defaults mirror the exact args recorded in Results/angular_separation_raw.json
# for the existing Global-CP reference run, so a bare invocation of this script
# reproduces identical splits/calibration/test pools.
DEFAULT_DATA_PATHS = [
    "/home/dsi/mayavb/PythonProjects/SRP-DNN/data/doa_sep_pilot_v2/D5/Reverb_400_ms_SNR_15_dB/speakers_2_flat.npz",
    "/home/dsi/mayavb/PythonProjects/SRP-DNN/data/doa_sep_pilot_v2/D10/Reverb_400_ms_SNR_15_dB/speakers_2_flat.npz",
    "/home/dsi/mayavb/PythonProjects/SRP-DNN/data/doa_sep_pilot_v2/D15/Reverb_400_ms_SNR_15_dB/speakers_2_flat.npz",
    "/home/dsi/mayavb/PythonProjects/SRP-DNN/data/doa_sep_pilot_v2/D20/Reverb_400_ms_SNR_15_dB/speakers_2_flat.npz",
    "/home/dsi/mayavb/PythonProjects/SRP-DNN/data/doa_sep_pilot_v2/D30/Reverb_400_ms_SNR_15_dB/speakers_2_flat.npz",
    "/home/dsi/mayavb/PythonProjects/SRP-DNN/data/doa_sep_pilot_v2/D45/Reverb_400_ms_SNR_15_dB/speakers_2_flat.npz",
    "/home/dsi/mayavb/PythonProjects/SRP-DNN/data/doa_sep_pilot_v2/D60/Reverb_400_ms_SNR_15_dB/speakers_2_flat.npz",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_paths", nargs="+", default=DEFAULT_DATA_PATHS,
                    help="One speakers_2_flat.npz path per --separations entry, same order")
    p.add_argument("--separations", nargs="+", type=float, default=DEFAULT_SEPARATIONS)
    p.add_argument("--n_splits", type=int, default=20)
    p.add_argument("--calib_scene_frac", type=float, default=0.5)
    p.add_argument("--n_calib_frames_per_scene", type=int, default=10)
    p.add_argument("--n_test_frames_per_scene", type=int, default=10)
    p.add_argument("--lambda_steps", type=int, default=500)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0, help="base seed; split i uses seed + i (must match Global run)")
    p.add_argument("--out_dir", default="Results")
    return p.parse_args()


def oracle_mondrian_split(conds, separations, common_scenes, room, nele, nazi, lambda_list, args, split_seed):
    """Bit-identical calib/test scene split and frame sampling to
    eval_angular_separation.run_one_split for the same split_seed (same rng
    call sequence: permutation, then pool_scenes(calib), then
    pool_scenes(test)). Returns per-record Global AND Oracle-Mondrian
    coverage/area, plus calibration diagnostics (n_calib per D, lambdas,
    any infeasibility)."""
    rng = np.random.default_rng(split_seed)
    scene_order = rng.permutation(common_scenes)
    n_calib = int(round(len(scene_order) * args.calib_scene_frac))
    calib_scenes, test_scenes = scene_order[:n_calib], scene_order[n_calib:]
    assert len(set(calib_scenes.tolist()) & set(test_scenes.tolist())) == 0, \
        "a base_scene_id leaked into both calib and test -- pairing invariant violated"

    lm_c, est_c, true_c, D_c = pool_scenes(conds, separations, calib_scenes,
                                            args.n_calib_frames_per_scene, rng)
    lm_t, est_t, true_t, D_t = pool_scenes(conds, separations, test_scenes,
                                            args.n_test_frames_per_scene, rng)

    # ---- Global CP: one calibration pooled across all D (unchanged from eval_angular_separation.py) ----
    lambdas_global = calibrate_global_lambda_from_arrays(lm_c, est_c, true_c, room, lambda_list, args.alpha)

    # ---- Oracle Mondrian CP: one independent calibration per true-D category ----
    lambdas_mondrian = {}
    n_calib_by_D = {}
    infeasible = []
    for D in separations:
        mask = (D_c == D)
        n_calib_by_D[D] = int(mask.sum())
        if n_calib_by_D[D] == 0:
            infeasible.append((D, "no calibration records for this D in this split"))
            continue
        try:
            lambdas_mondrian[D] = calibrate_global_lambda_from_arrays(
                lm_c[mask], est_c[mask], true_c[mask], room, lambda_list, args.alpha)
        except ValueError as e:
            infeasible.append((D, str(e)))

    # ---- Evaluate Global and Oracle Mondrian on IDENTICAL test records ----
    records_global, records_mondrian = [], []
    for i in range(lm_t.shape[0]):
        D_i = float(D_t[i])
        true_order, est_order = CoverageSet._match_estimated_to_source(true_t[i], est_t[i])
        for true_s, est_s in zip(true_order, est_order):
            k = int(est_s)
            norm_map = normalize(lm_t[i, k])
            seed = tuple(radians_to_grid_index(est_t[i, k], nele, nazi).astype(int))
            true_idx = tuple(radians_to_grid_index(true_t[i, true_s], nele, nazi).astype(int))

            region_g = CoverageSet.neighbours_coverage_set(norm_map, float(lambdas_global[k]), estimated_position=seed)
            records_global.append(dict(D=D_i, covered=bool(region_g[true_idx]), area=int(region_g.sum())))

            if D_i in lambdas_mondrian:
                region_m = CoverageSet.neighbours_coverage_set(
                    norm_map, float(lambdas_mondrian[D_i][k]), estimated_position=seed)
                records_mondrian.append(dict(D=D_i, covered=bool(region_m[true_idx]), area=int(region_m.sum())))

    return dict(
        n_calib_scenes=len(calib_scenes), n_test_scenes=len(test_scenes),
        n_calib_by_D=n_calib_by_D, infeasible=infeasible,
        lambdas_global=lambdas_global, lambdas_mondrian=lambdas_mondrian,
        records_global=records_global, records_mondrian=records_mondrian,
    )


def summarize_by_D(records, separations):
    out = {}
    for D in separations:
        recs = [r for r in records if r["D"] == D]
        if recs:
            out[D] = dict(coverage=float(np.mean([r["covered"] for r in recs])),
                           area=float(np.mean([r["area"] for r in recs])), n=len(recs))
    return out


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    separations = args.separations
    assert len(separations) == len(args.data_paths), "--separations and --data_paths must match in length"

    print("=" * 78)
    print("ORACLE Mondrian CP vs Global CP -- true commanded D as Mondrian category")
    print("ORACLE diagnostic: true D is NOT available at inference in the real problem.")
    print("=" * 78)

    conds, nele, nazi, common_scenes = load_conditions(args.data_paths, separations)
    room = _build_room(conds[separations[0]])
    n_grid_cells = nele * nazi
    print(f"\n{len(common_scenes)} paired base scenes, grid {nele}x{nazi}={n_grid_cells} cells, "
          f"separations={separations}")

    lambda_list = np.linspace(0.0, 1.0, args.lambda_steps)

    print(f"\nRunning {args.n_splits} scene-disjoint splits (seed={args.seed}, "
          f"calib_scene_frac={args.calib_scene_frac}, "
          f"n_calib_frames_per_scene={args.n_calib_frames_per_scene}, "
          f"n_test_frames_per_scene={args.n_test_frames_per_scene})")
    print("-" * 78)

    split_results = []
    all_infeasible = []
    for i in range(args.n_splits):
        split_seed = args.seed + i
        res = oracle_mondrian_split(conds, separations, common_scenes, room, nele, nazi,
                                     lambda_list, args, split_seed)
        split_results.append(res)

        n_calib_str = " ".join(f"D={D:.0f}:n={res['n_calib_by_D'][D]}" for D in separations)
        print(f"split {i:3d}  calib_scenes={res['n_calib_scenes']:2d} test_scenes={res['n_test_scenes']:2d}  {n_calib_str}")
        if res["infeasible"]:
            for D, reason in res["infeasible"]:
                all_infeasible.append((i, D, reason))
                print(f"    INFEASIBLE split={i} D={D}: {reason}")

    print("-" * 78)
    if all_infeasible:
        print(f"\n{len(all_infeasible)} (split, D) Mondrian calibrations were INFEASIBLE "
              f"(n_calib too small for alpha={args.alpha}) and were excluded from Mondrian "
              f"aggregation for that split/D only (Global CP unaffected):")
        for i, D, reason in all_infeasible:
            print(f"  split={i} D={D}: {reason}")
    else:
        print("\nNo finite-sample infeasibility encountered: every (split, D) Mondrian "
              f"calibration had n_calib > 1/{args.alpha:.2f} - 1 = {1/args.alpha - 1:.1f} records.")

    # ---- Per-D, per-split coverage/area summaries ----
    global_by_D_per_split = [summarize_by_D(r["records_global"], separations) for r in split_results]
    mondrian_by_D_per_split = [summarize_by_D(r["records_mondrian"], separations) for r in split_results]

    print("\n" + "=" * 78)
    print("PRIMARY RESULT TABLE -- Global CP vs ORACLE Mondrian CP, by true D")
    print("=" * 78)
    header = (f"{'D_deg':>6} {'n_g':>5} {'n_m':>5} "
              f"{'cov_global':>11} {'cov_mondrian':>13} {'d_cov':>8} "
              f"{'area_global':>12} {'area_mondrian':>14} {'d_area':>9} "
              f"{'area%_global':>13} {'area%_mondrian':>15}")
    print(header)

    table_rows = {}
    n_splits_mondrian_gt_global = {}
    n_splits_mondrian_closer_to_target = {}
    for D in separations:
        covs_g = [d[D]["coverage"] for d in global_by_D_per_split if D in d]
        areas_g = [d[D]["area"] for d in global_by_D_per_split if D in d]
        covs_m = [d[D]["coverage"] for d in mondrian_by_D_per_split if D in d]
        areas_m = [d[D]["area"] for d in mondrian_by_D_per_split if D in d]
        n_g = [d[D]["n"] for d in global_by_D_per_split if D in d]
        n_m = [d[D]["n"] for d in mondrian_by_D_per_split if D in d]

        # paired per-split diffs, only over splits where BOTH methods have data for this D
        splits_both = [s for s in range(len(split_results))
                       if D in global_by_D_per_split[s] and D in mondrian_by_D_per_split[s]]
        cov_diffs = [mondrian_by_D_per_split[s][D]["coverage"] - global_by_D_per_split[s][D]["coverage"]
                     for s in splits_both]
        area_diffs = [mondrian_by_D_per_split[s][D]["area"] - global_by_D_per_split[s][D]["area"]
                      for s in splits_both]
        n_gt = sum(1 for s in splits_both
                   if mondrian_by_D_per_split[s][D]["coverage"] > global_by_D_per_split[s][D]["coverage"])
        n_closer = sum(1 for s in splits_both
                       if abs(mondrian_by_D_per_split[s][D]["coverage"] - 0.90) <
                          abs(global_by_D_per_split[s][D]["coverage"] - 0.90))
        n_splits_mondrian_gt_global[D] = (n_gt, len(splits_both))
        n_splits_mondrian_closer_to_target[D] = (n_closer, len(splits_both))

        row = dict(
            cov_g_mean=float(np.mean(covs_g)), cov_g_std=float(np.std(covs_g)),
            cov_m_mean=float(np.mean(covs_m)) if covs_m else float("nan"),
            cov_m_std=float(np.std(covs_m)) if covs_m else float("nan"),
            area_g_mean=float(np.mean(areas_g)), area_g_std=float(np.std(areas_g)),
            area_m_mean=float(np.mean(areas_m)) if areas_m else float("nan"),
            area_m_std=float(np.std(areas_m)) if areas_m else float("nan"),
            d_cov_mean=float(np.mean(cov_diffs)) if cov_diffs else float("nan"),
            d_cov_std=float(np.std(cov_diffs)) if cov_diffs else float("nan"),
            d_area_mean=float(np.mean(area_diffs)) if area_diffs else float("nan"),
            d_area_std=float(np.std(area_diffs)) if area_diffs else float("nan"),
            n_splits=len(splits_both),
        )
        table_rows[D] = row

        area_pct_g = row["area_g_mean"] / n_grid_cells * 100
        area_pct_m = row["area_m_mean"] / n_grid_cells * 100 if covs_m else float("nan")
        print(f"{D:6.1f} {np.mean(n_g):5.0f} {(np.mean(n_m) if n_m else float('nan')):5.0f} "
              f"{row['cov_g_mean']:6.3f}+/-{row['cov_g_std']:4.3f} "
              f"{row['cov_m_mean']:7.3f}+/-{row['cov_m_std']:4.3f} "
              f"{row['d_cov_mean']:+7.3f} "
              f"{row['area_g_mean']:7.1f}+/-{row['area_g_std']:5.1f} "
              f"{row['area_m_mean']:8.1f}+/-{row['area_m_std']:5.1f} "
              f"{row['d_area_mean']:+8.1f} "
              f"{area_pct_g:12.2f} {area_pct_m:14.2f}")

    print("\nSplits where Mondrian coverage > Global, and where Mondrian is closer to 0.90:")
    for D in separations:
        gt, tot = n_splits_mondrian_gt_global[D]
        closer, tot2 = n_splits_mondrian_closer_to_target[D]
        print(f"  D={D:5.1f}deg  mondrian>global: {gt}/{tot}   mondrian_closer_to_0.90: {closer}/{tot2}")

    # ---- Section 8: thresholds ----
    print("\n" + "=" * 78)
    print("MONDRIAN CALIBRATION THRESHOLDS (mean/std across splits, per speaker slot k)")
    print("Lambda orientation: neighbours_coverage_set includes cell (i,j) iff "
          "normalized_map[i,j] >= lambda (Code/crc_ssl.py:335). SMALLER lambda -> "
          "MORE cells pass the threshold -> LARGER/more permissive region. "
          "LARGER lambda -> smaller/tighter region.")
    print("=" * 78)
    K = split_results[0]["lambdas_global"].shape[0]
    lambda_global_by_k = {k: [float(r["lambdas_global"][k]) for r in split_results] for k in range(K)}
    lambda_mondrian_by_Dk = {}
    for D in separations:
        for k in range(K):
            vals = [float(r["lambdas_mondrian"][D][k]) for r in split_results if D in r["lambdas_mondrian"]]
            lambda_mondrian_by_Dk[(D, k)] = vals

    for k in range(K):
        print(f"\nspeaker slot k={k}:  mean(lambda_global[k]) = "
              f"{np.mean(lambda_global_by_k[k]):.4f} +/- {np.std(lambda_global_by_k[k]):.4f}")
        for D in separations:
            vals = lambda_mondrian_by_Dk[(D, k)]
            if vals:
                print(f"    D={D:5.1f}deg  mean(lambda_mondrian) = {np.mean(vals):.4f} "
                      f"+/- {np.std(vals):.4f}  (n_splits={len(vals)})")
            else:
                print(f"    D={D:5.1f}deg  (no feasible splits)")

    # ---- Section 9: overall/marginal ----
    print("\n" + "=" * 78)
    print("OVERALL / MARGINAL (pooled across all D) -- reported for completeness, NOT the "
          "primary conclusion (conditional-within-D behavior is)")
    print("=" * 78)
    overall_cov_g = [np.mean([rr["covered"] for rr in r["records_global"]]) for r in split_results]
    overall_area_g = [np.mean([rr["area"] for rr in r["records_global"]]) for r in split_results]
    overall_cov_m = [np.mean([rr["covered"] for rr in r["records_mondrian"]]) for r in split_results
                      if r["records_mondrian"]]
    overall_area_m = [np.mean([rr["area"] for rr in r["records_mondrian"]]) for r in split_results
                       if r["records_mondrian"]]
    print(f"GLOBAL:           coverage = {np.mean(overall_cov_g):.4f} +/- {np.std(overall_cov_g):.4f}   "
          f"area = {np.mean(overall_area_g):.1f} +/- {np.std(overall_area_g):.1f}")
    print(f"ORACLE MONDRIAN:  coverage = {np.mean(overall_cov_m):.4f} +/- {np.std(overall_cov_m):.4f}   "
          f"area = {np.mean(overall_area_m):.1f} +/- {np.std(overall_area_m):.1f}")

    # ---- Section 10: interpretation, computed directly from the numbers above ----
    print("\n" + "=" * 78)
    print("INTERPRETATION (data-driven, from the table above)")
    print("=" * 78)

    def cov_g(D): return table_rows[D]["cov_g_mean"]
    def cov_m(D): return table_rows[D]["cov_m_mean"]
    def area_g(D): return table_rows[D]["area_g_mean"]
    def area_m(D): return table_rows[D]["area_m_mean"]

    print(f"1. D=10 closer to 0.90 under Mondrian? "
          f"Global={cov_g(10.0):.3f} (|.|={abs(cov_g(10.0)-0.9):.3f})  "
          f"Mondrian={cov_m(10.0):.3f} (|.|={abs(cov_m(10.0)-0.9):.3f})  -> "
          f"{'YES' if abs(cov_m(10.0)-0.9) < abs(cov_g(10.0)-0.9) else 'NO'}")
    print(f"2. D=15 closer to 0.90 under Mondrian? "
          f"Global={cov_g(15.0):.3f} (|.|={abs(cov_g(15.0)-0.9):.3f})  "
          f"Mondrian={cov_m(15.0):.3f} (|.|={abs(cov_m(15.0)-0.9):.3f})  -> "
          f"{'YES' if abs(cov_m(15.0)-0.9) < abs(cov_g(15.0)-0.9) else 'NO'}")
    print(f"3. D=60 overcoverage reduced toward 0.90 under Mondrian? "
          f"Global={cov_g(60.0):.3f}  Mondrian={cov_m(60.0):.3f}  -> "
          f"{'YES' if abs(cov_m(60.0)-0.9) < abs(cov_g(60.0)-0.9) else 'NO'}")
    print(f"4. Hard D (10,15) -- does improved coverage cost more area? "
          f"D=10: area_g={area_g(10.0):.1f} area_m={area_m(10.0):.1f} (d={area_m(10.0)-area_g(10.0):+.1f})  "
          f"D=15: area_g={area_g(15.0):.1f} area_m={area_m(15.0):.1f} (d={area_m(15.0)-area_g(15.0):+.1f})")
    print(f"5. Easy/overcovered D (60) -- can Mondrian shrink area while staying near target? "
          f"area_g={area_g(60.0):.1f} area_m={area_m(60.0):.1f} (d={area_m(60.0)-area_g(60.0):+.1f})  "
          f"cov_m={cov_m(60.0):.3f}")
    cov_spread_g = max(cov_g(D) for D in separations) - min(cov_g(D) for D in separations)
    cov_spread_m = max(cov_m(D) for D in separations) - min(cov_m(D) for D in separations)
    print(f"6. Conditional-coverage spread across all 7 D: "
          f"Global={cov_spread_g:.3f}  Mondrian={cov_spread_m:.3f}  -> "
          f"{'FLATTENED' if cov_spread_m < cov_spread_g else 'NOT flattened'}")
    print("7. Split-to-split variability from smaller per-category calibration sets "
          "(std across 20 splits, Global vs Mondrian, per D):")
    for D in separations:
        print(f"     D={D:5.1f}deg  cov_std_global={table_rows[D]['cov_g_std']:.4f}  "
              f"cov_std_mondrian={table_rows[D]['cov_m_std']:.4f}  "
              f"area_std_global={table_rows[D]['area_g_std']:.1f}  "
              f"area_std_mondrian={table_rows[D]['area_m_std']:.1f}")
    print("8. Are learned per-D thresholds consistent with observed difficulty? See "
          "the lambda table above -- compare D=10/15 (hard, undercovered under Global) "
          "vs D=60 (easy, overcovered under Global).")

    print("\n" + "=" * 78)
    print("SCIENTIFIC INTERPRETATION CONSTRAINT (per spec, section 11)")
    print("=" * 78)
    print("This is an ORACLE experiment using TRUE commanded D, unavailable at inference.")
    print("If this succeeds, the conclusion is NOT 'we solved conditional coverage using")
    print("true separation.' The conclusion is: 'the conditional heterogeneity induced by")
    print("speaker separation is calibratable when the separation regime is known.' This")
    print("motivates (but does NOT itself answer) whether an inference-time observable")
    print("category (e.g. estimated separation) can approximate these Mondrian groups --")
    print("that question is explicitly OUT OF SCOPE for this script.")

    # ---- save numeric results ----
    out_txt = os.path.join(args.out_dir, "results_oracle_mondrian_angular_separation.txt")
    with open(out_txt, "w") as fh:
        fh.write("ORACLE Mondrian CP vs Global CP, true commanded D as Mondrian category\n")
        fh.write("ORACLE diagnostic only -- true D is NOT available at inference in the real problem.\n\n")
        fh.write(f"{len(common_scenes)} paired base scenes, grid {nele}x{nazi}={n_grid_cells} cells, "
                 f"separations={separations}\n")
        fh.write(f"n_splits={args.n_splits} seed={args.seed} calib_scene_frac={args.calib_scene_frac} "
                 f"n_calib_frames_per_scene={args.n_calib_frames_per_scene} "
                 f"n_test_frames_per_scene={args.n_test_frames_per_scene} alpha={args.alpha}\n\n")
        if all_infeasible:
            fh.write(f"{len(all_infeasible)} (split, D) Mondrian calibrations INFEASIBLE (excluded):\n")
            for i, D, reason in all_infeasible:
                fh.write(f"  split={i} D={D}: {reason}\n")
        else:
            fh.write("No finite-sample infeasibility encountered.\n")
        fh.write("\n")
        fh.write(f"{'D_deg':>6} {'cov_global':>11} {'cov_mondrian':>13} {'d_cov':>8} "
                 f"{'area_global':>12} {'area_mondrian':>14} {'d_area':>9} "
                 f"{'area%_global':>13} {'area%_mondrian':>15}\n")
        for D in separations:
            row = table_rows[D]
            area_pct_g = row["area_g_mean"] / n_grid_cells * 100
            area_pct_m = row["area_m_mean"] / n_grid_cells * 100 if not np.isnan(row["area_m_mean"]) else float("nan")
            fh.write(f"{D:6.1f} {row['cov_g_mean']:6.3f}+/-{row['cov_g_std']:5.3f} "
                     f"{row['cov_m_mean']:7.3f}+/-{row['cov_m_std']:5.3f} "
                     f"{row['d_cov_mean']:+7.3f} "
                     f"{row['area_g_mean']:7.1f}+/-{row['area_g_std']:6.1f} "
                     f"{row['area_m_mean']:8.1f}+/-{row['area_m_std']:6.1f} "
                     f"{row['d_area_mean']:+8.1f} "
                     f"{area_pct_g:12.2f} {area_pct_m:14.2f}\n")
        fh.write(f"\noverall_global: coverage={np.mean(overall_cov_g):.4f}+/-{np.std(overall_cov_g):.4f} "
                 f"area={np.mean(overall_area_g):.1f}+/-{np.std(overall_area_g):.1f}\n")
        fh.write(f"overall_oracle_mondrian: coverage={np.mean(overall_cov_m):.4f}+/-{np.std(overall_cov_m):.4f} "
                 f"area={np.mean(overall_area_m):.1f}+/-{np.std(overall_area_m):.1f}\n")
    print(f"\nSaved -> {out_txt}")

    out_json = os.path.join(args.out_dir, "oracle_mondrian_angular_separation_raw.json")
    with open(out_json, "w") as fh:
        json.dump(dict(
            args=vars(args),
            n_grid_cells=n_grid_cells,
            infeasible=[dict(split=i, D=D, reason=reason) for i, D, reason in all_infeasible],
            table_rows={str(D): v for D, v in table_rows.items()},
            n_splits_mondrian_gt_global={str(D): v for D, v in n_splits_mondrian_gt_global.items()},
            n_splits_mondrian_closer_to_target={str(D): v for D, v in n_splits_mondrian_closer_to_target.items()},
            lambda_global_mean={str(k): float(np.mean(v)) for k, v in lambda_global_by_k.items()},
            lambda_global_std={str(k): float(np.std(v)) for k, v in lambda_global_by_k.items()},
            lambda_mondrian_mean={f"{D}_{k}": (float(np.mean(v)) if v else None)
                                   for (D, k), v in lambda_mondrian_by_Dk.items()},
            lambda_mondrian_std={f"{D}_{k}": (float(np.std(v)) if v else None)
                                  for (D, k), v in lambda_mondrian_by_Dk.items()},
            overall_global=dict(coverage_mean=float(np.mean(overall_cov_g)), coverage_std=float(np.std(overall_cov_g)),
                                 area_mean=float(np.mean(overall_area_g)), area_std=float(np.std(overall_area_g))),
            overall_oracle_mondrian=dict(coverage_mean=float(np.mean(overall_cov_m)), coverage_std=float(np.std(overall_cov_m)),
                                          area_mean=float(np.mean(overall_area_m)), area_std=float(np.std(overall_area_m))),
        ), fh, indent=2)
    print(f"Saved -> {out_json}")


if __name__ == "__main__":
    main()
