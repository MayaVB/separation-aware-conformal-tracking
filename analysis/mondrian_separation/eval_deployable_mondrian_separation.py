"""
DEPLOYABLE Mondrian CP using estimated (not true) speaker separation
Delta_hat as the Mondrian category.

Unlike eval_oracle_mondrian_angular_separation.py, this script's Mondrian
category assignment uses ONLY model output (the two estimated DOAs in a
frame) -- never true/commanded D. True D is used AFTER calibration and
category assignment, only to break down results for evaluation (section
B7 below), exactly mirroring how the Oracle experiment reports by true D.

Delta_hat definition (frame-level, uses BOTH estimated DOAs, no GT):
    cos(Delta_hat) = cos(th0)*cos(th1) + sin(th0)*sin(th1)*cos(phi0-phi1)
    Delta_hat = arccos(clip(cos(Delta_hat), -1, 1))
where [th, phi] = [elevation, azimuth] radians, from
all_estimated_positions[t,0,:] and [t,1,:] (raw estimated-slot order, NOT
the true<->estimated matched order used elsewhere for scoring).

Mondrian categories are FROZEN before this run (from Diagnostic 2, not
re-derived here): 8 bins over Delta_hat in degrees:
    G1=[0,13) G2=[13,18.5) G3=[18.5,23.5) G4=[23.5,33) G5=[33,43)
    G6=[43,53) G7=[53,63) G8=[63,200)
Not merged, split, or re-boundaried based on this script's own results.

Method: identical calibration machinery to Global CP and Oracle Mondrian
(Code.two_speaker_tracking.lcp.calibrate_global_lambda_from_arrays,
Code.crc_ssl.CoverageSet.neighbours_coverage_set, same matching
convention, same hole-filling). The ONLY thing this script adds is (a)
computing Delta_hat from estimated positions and (b) assigning the fixed
G_j bin above from it. No LCP, no RBF, no localized ranks, no GT in the
category assignment.

Splits are bit-identical to eval_angular_separation.run_one_split /
eval_oracle_mondrian_angular_separation.oracle_mondrian_split for the
same split_seed (same rng call sequence via the unmodified, imported
pool_scenes, always invoked with the FULL 7-condition separations list).

Finite-sample handling matches the Oracle script: Code.crc_ssl.CoverageSet
raises ValueError when a category's calibration set is too small for
alpha (n <= 8 for alpha=0.1). This is caught and reported as INFEASIBLE
per (split, category); no fallback, no borrowing across bins, no merging.

Oracle Mondrian numbers used in section B7's three-way table are loaded
from the existing, already-validated
Results/oracle_mondrian_angular_separation_raw.json (not recomputed --
that experiment used identical splits/calibration/test pools, so reusing
its saved per-D table is exact, not an approximation).
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
from Code.two_speaker_tracking.lcp import calibrate_global_lambda_from_arrays

from eval_angular_separation import load_conditions, pool_scenes, DEFAULT_SEPARATIONS
from eval_oracle_mondrian_angular_separation import DEFAULT_DATA_PATHS

BIN_EDGES_DEG = [0.0, 13.0, 18.5, 23.5, 33.0, 43.0, 53.0, 63.0, 200.0]
N_BINS = len(BIN_EDGES_DEG) - 1
BIN_LABELS = [f"G{j+1}=[{BIN_EDGES_DEG[j]:.1f},{BIN_EDGES_DEG[j+1]:.1f})" for j in range(N_BINS)]


def delta_hat_deg_batch(est_rad):
    """est_rad: (n, 2, 2) radians [ele, azi] for the two ESTIMATED (unmatched)
    speaker slots, straight from all_estimated_positions. No GT used."""
    th0, az0 = est_rad[:, 0, 0], est_rad[:, 0, 1]
    th1, az1 = est_rad[:, 1, 0], est_rad[:, 1, 1]
    cosD = np.cos(th0) * np.cos(th1) + np.sin(th0) * np.sin(th1) * np.cos(az0 - az1)
    cosD = np.clip(cosD, -1.0, 1.0)
    return np.degrees(np.arccos(cosD))


def assign_bins(delta_deg):
    idx = np.searchsorted(BIN_EDGES_DEG, delta_deg, side="right") - 1
    return np.clip(idx, 0, N_BINS - 1)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_paths", nargs="+", default=DEFAULT_DATA_PATHS)
    p.add_argument("--separations", nargs="+", type=float, default=DEFAULT_SEPARATIONS)
    p.add_argument("--n_splits", type=int, default=20)
    p.add_argument("--calib_scene_frac", type=float, default=0.5)
    p.add_argument("--n_calib_frames_per_scene", type=int, default=10)
    p.add_argument("--n_test_frames_per_scene", type=int, default=10)
    p.add_argument("--lambda_steps", type=int, default=500)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="Results")
    p.add_argument("--oracle_json", default="Results/oracle_mondrian_angular_separation_raw.json",
                    help="reused, not recomputed, for the Oracle Mondrian column of the B7 table")
    return p.parse_args()


def deployable_split(conds, separations, common_scenes, room, nele, nazi, lambda_list, args, split_seed):
    rng = np.random.default_rng(split_seed)
    scene_order = rng.permutation(common_scenes)
    n_calib = int(round(len(scene_order) * args.calib_scene_frac))
    calib_scenes, test_scenes = scene_order[:n_calib], scene_order[n_calib:]

    lm_c, est_c, true_c, D_c = pool_scenes(conds, separations, calib_scenes,
                                            args.n_calib_frames_per_scene, rng)
    lm_t, est_t, true_t, D_t = pool_scenes(conds, separations, test_scenes,
                                            args.n_test_frames_per_scene, rng)

    lambdas_global = calibrate_global_lambda_from_arrays(lm_c, est_c, true_c, room, lambda_list, args.alpha)

    Gc = assign_bins(delta_hat_deg_batch(est_c))
    Gt = assign_bins(delta_hat_deg_batch(est_t))

    lambdas_hatD = {}
    n_calib_by_G = {}
    infeasible = []
    for j in range(N_BINS):
        mask = (Gc == j)
        n_calib_by_G[j] = int(mask.sum())
        if n_calib_by_G[j] == 0:
            infeasible.append((j, "no calibration frames in this Delta_hat bin for this split"))
            continue
        try:
            lambdas_hatD[j] = calibrate_global_lambda_from_arrays(
                lm_c[mask], est_c[mask], true_c[mask], room, lambda_list, args.alpha)
        except ValueError as e:
            infeasible.append((j, str(e)))

    records_global, records_hatD = [], []
    for i in range(lm_t.shape[0]):
        D_i = float(D_t[i])
        Gj = int(Gt[i])
        true_order, est_order = CoverageSet._match_estimated_to_source(true_t[i], est_t[i])
        for true_s, est_s in zip(true_order, est_order):
            k = int(est_s)
            norm_map = normalize(lm_t[i, k])
            seed = tuple(radians_to_grid_index(est_t[i, k], nele, nazi).astype(int))
            true_idx = tuple(radians_to_grid_index(true_t[i, true_s], nele, nazi).astype(int))

            region_g = CoverageSet.neighbours_coverage_set(norm_map, float(lambdas_global[k]), estimated_position=seed)
            records_global.append(dict(D=D_i, G=Gj, covered=bool(region_g[true_idx]), area=int(region_g.sum())))

            if Gj in lambdas_hatD:
                region_h = CoverageSet.neighbours_coverage_set(
                    norm_map, float(lambdas_hatD[Gj][k]), estimated_position=seed)
                records_hatD.append(dict(D=D_i, G=Gj, covered=bool(region_h[true_idx]), area=int(region_h.sum())))

    return dict(
        n_calib_scenes=len(calib_scenes), n_test_scenes=len(test_scenes),
        n_calib_by_G=n_calib_by_G, infeasible=infeasible,
        lambdas_global=lambdas_global, lambdas_hatD=lambdas_hatD,
        records_global=records_global, records_hatD=records_hatD,
    )


def summarize_by_key(records, key, keys):
    out = {}
    for kv in keys:
        recs = [r for r in records if r[key] == kv]
        if recs:
            areas = [r["area"] for r in recs]
            out[kv] = dict(coverage=float(np.mean([r["covered"] for r in recs])),
                            area=float(np.mean(areas)), area_median=float(np.median(areas)), n=len(recs))
    return out


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    separations = args.separations
    conds, nele, nazi, common_scenes = load_conditions(args.data_paths, separations)
    room = _build_room(conds[separations[0]])
    n_grid_cells = nele * nazi
    lambda_list = np.linspace(0.0, 1.0, args.lambda_steps)

    with open(args.oracle_json) as fh:
        oracle = json.load(fh)
    oracle_rows = oracle["table_rows"]  # keys are str(D)

    print("=" * 90)
    print("PART B -- DEPLOYABLE Mondrian CP using estimated separation Delta_hat")
    print("Category assignment uses ONLY model estimates. True D used for evaluation only.")
    print("Frozen bins (from Diagnostic 2, NOT re-derived here):")
    for lbl in BIN_LABELS:
        print(f"  {lbl}")
    print("=" * 90)

    split_results = []
    for i in range(args.n_splits):
        res = deployable_split(conds, separations, common_scenes, room, nele, nazi, lambda_list, args, args.seed + i)
        split_results.append(res)
        n_str = " ".join(f"G{j+1}:n={res['n_calib_by_G'][j]}" for j in range(N_BINS))
        print(f"split {i:3d}  calib_scenes={res['n_calib_scenes']:2d}  {n_str}")
        for j, reason in res["infeasible"]:
            print(f"    INFEASIBLE split={i} G{j+1}: {reason}")

    all_infeasible = [(i, j, reason) for i, res in enumerate(split_results) for j, reason in res["infeasible"]]
    print("-" * 90)
    if all_infeasible:
        print(f"\n{len(all_infeasible)} (split, G) categories INFEASIBLE -- excluded from that split/category only:")
        for i, j, reason in all_infeasible:
            print(f"  split={i} G{j+1}: {reason}")
        feasible_all = False
    else:
        print("\nAll 8 categories were FEASIBLE in all 20 splits (n_calib > "
              f"1/{args.alpha:.2f} - 1 = {1/args.alpha - 1:.1f} in every case).")
        feasible_all = True

    # ---- B4: full n_calib matrix ----
    print("\n" + "=" * 90)
    print("B4. n_calib per split per Delta_hat category")
    print("=" * 90)
    header = "split " + " ".join(f"{('G'+str(j+1)):>6}" for j in range(N_BINS))
    print(header)
    for i, res in enumerate(split_results):
        print(f"{i:5d} " + " ".join(f"{res['n_calib_by_G'][j]:6d}" for j in range(N_BINS)))

    # ---- B7: primary three-way table by true D ----
    global_by_D = [summarize_by_key(r["records_global"], "D", separations) for r in split_results]
    hatD_by_D = [summarize_by_key(r["records_hatD"], "D", separations) for r in split_results]

    print("\n" + "=" * 90)
    print("B7. PRIMARY RESULT -- Global vs ORACLE Mondrian(true D) vs ESTIMATED-SEPARATION Mondrian, by true D")
    print("=" * 90)
    print(f"{'D':>5} {'cov_global':>11} {'cov_oracle':>11} {'cov_hatD':>10} {'d_cov(hatD-g)':>14} "
          f"{'area_global':>12} {'area_oracle':>12} {'area_hatD':>11} {'d_area':>9}")
    b7_rows = {}
    n_hatD_gt_global = {}
    n_hatD_closer = {}
    for D in separations:
        covs_g = [d[D]["coverage"] for d in global_by_D if D in d]
        areas_g = [d[D]["area"] for d in global_by_D if D in d]
        covs_h = [d[D]["coverage"] for d in hatD_by_D if D in d]
        areas_h = [d[D]["area"] for d in hatD_by_D if D in d]
        splits_both = [s for s in range(len(split_results)) if D in global_by_D[s] and D in hatD_by_D[s]]
        d_cov = [hatD_by_D[s][D]["coverage"] - global_by_D[s][D]["coverage"] for s in splits_both]
        d_area = [hatD_by_D[s][D]["area"] - global_by_D[s][D]["area"] for s in splits_both]
        n_gt = sum(1 for s in splits_both if hatD_by_D[s][D]["coverage"] > global_by_D[s][D]["coverage"])
        n_closer = sum(1 for s in splits_both
                       if abs(hatD_by_D[s][D]["coverage"] - 0.9) < abs(global_by_D[s][D]["coverage"] - 0.9))
        n_hatD_gt_global[D] = (n_gt, len(splits_both))
        n_hatD_closer[D] = (n_closer, len(splits_both))

        orc = oracle_rows[str(D)]
        row = dict(cov_g=float(np.mean(covs_g)), cov_g_std=float(np.std(covs_g)),
                   cov_o=orc["cov_m_mean"], cov_o_std=orc["cov_m_std"],
                   cov_h=float(np.mean(covs_h)) if covs_h else float("nan"),
                   cov_h_std=float(np.std(covs_h)) if covs_h else float("nan"),
                   area_g=float(np.mean(areas_g)), area_g_std=float(np.std(areas_g)),
                   area_o=orc["area_m_mean"], area_o_std=orc["area_m_std"],
                   area_h=float(np.mean(areas_h)) if areas_h else float("nan"),
                   area_h_std=float(np.std(areas_h)) if areas_h else float("nan"),
                   d_cov=float(np.mean(d_cov)) if d_cov else float("nan"),
                   d_area=float(np.mean(d_area)) if d_area else float("nan"))
        b7_rows[D] = row
        print(f"{D:5.1f} {row['cov_g']:6.3f}+/-{row['cov_g_std']:4.3f} "
              f"{row['cov_o']:6.3f}+/-{row['cov_o_std']:4.3f} "
              f"{row['cov_h']:6.3f}+/-{row['cov_h_std']:4.3f} "
              f"{row['d_cov']:+8.3f}      "
              f"{row['area_g']:7.1f}+/-{row['area_g_std']:5.1f} "
              f"{row['area_o']:7.1f}+/-{row['area_o_std']:5.1f} "
              f"{row['area_h']:7.1f}+/-{row['area_h_std']:5.1f} "
              f"{row['d_area']:+8.1f}")

    print("\nSplits hatD>global / hatD closer to 0.90 than global:")
    for D in separations:
        gt, tot = n_hatD_gt_global[D]
        cl, tot2 = n_hatD_closer[D]
        print(f"  D={D:5.1f}deg  hatD>global: {gt}/{tot}   hatD_closer_to_0.90: {cl}/{tot2}")

    # ---- B8: by observable Delta_hat category ----
    global_by_G = [summarize_by_key(r["records_global"], "G", list(range(N_BINS))) for r in split_results]
    hatD_by_G = [summarize_by_key(r["records_hatD"], "G", list(range(N_BINS))) for r in split_results]

    print("\n" + "=" * 90)
    print("B8. Evaluate directly by observable Delta_hat Mondrian category")
    print("=" * 90)
    print(f"{'cat':>4} {'n_test':>7} {'cov_global':>11} {'cov_hatD':>10} "
          f"{'area_g(mean/med)':>18} {'area_h(mean/med)':>18} {'n_calib(min/mean/max)':>22}")
    b8_rows = {}
    K = split_results[0]["lambdas_global"].shape[0]
    lambda_hatD_by_Gk = {(j, k): [] for j in range(N_BINS) for k in range(K)}
    for res in split_results:
        for j, lam in res["lambdas_hatD"].items():
            for k in range(K):
                lambda_hatD_by_Gk[(j, k)].append(float(lam[k]))

    for j in range(N_BINS):
        covs_g = [d[j]["coverage"] for d in global_by_G if j in d]
        covs_h = [d[j]["coverage"] for d in hatD_by_G if j in d]
        areas_g = [d[j]["area"] for d in global_by_G if j in d]
        areas_gm = [d[j]["area_median"] for d in global_by_G if j in d]
        areas_h = [d[j]["area"] for d in hatD_by_G if j in d]
        areas_hm = [d[j]["area_median"] for d in hatD_by_G if j in d]
        n_test = [d[j]["n"] for d in global_by_G if j in d]
        n_calib_list = [res["n_calib_by_G"][j] for res in split_results]
        b8_rows[j] = dict(
            cov_g=float(np.mean(covs_g)) if covs_g else float("nan"),
            cov_h=float(np.mean(covs_h)) if covs_h else float("nan"),
            area_g=float(np.mean(areas_g)) if areas_g else float("nan"),
            area_g_med=float(np.mean(areas_gm)) if areas_gm else float("nan"),
            area_h=float(np.mean(areas_h)) if areas_h else float("nan"),
            area_h_med=float(np.mean(areas_hm)) if areas_hm else float("nan"),
            n_test=float(np.mean(n_test)) if n_test else float("nan"),
            n_calib_min=min(n_calib_list), n_calib_mean=float(np.mean(n_calib_list)), n_calib_max=max(n_calib_list),
        )
        r = b8_rows[j]
        print(f"{('G'+str(j+1)):>4} {r['n_test']:7.0f} {r['cov_g']:11.3f} {r['cov_h']:10.3f} "
              f"{r['area_g']:8.1f}/{r['area_g_med']:7.1f} {r['area_h']:8.1f}/{r['area_h_med']:7.1f} "
              f"{r['n_calib_min']:5d}/{r['n_calib_mean']:6.1f}/{r['n_calib_max']:5d}")

    print("\nlambda_hatD[G_j,k] mean +/- std across splits:")
    for j in range(N_BINS):
        parts = []
        for k in range(K):
            vals = lambda_hatD_by_Gk[(j, k)]
            if vals:
                parts.append(f"k={k}: {np.mean(vals):.4f}+/-{np.std(vals):.4f} (n_splits={len(vals)})")
            else:
                parts.append(f"k={k}: (no feasible splits)")
        print(f"  {BIN_LABELS[j]:>22}  " + "  ".join(parts))

    # ---- B9: overall ----
    overall_cov_g = [np.mean([rr["covered"] for rr in r["records_global"]]) for r in split_results]
    overall_area_g = [np.mean([rr["area"] for rr in r["records_global"]]) for r in split_results]
    overall_cov_h = [np.mean([rr["covered"] for rr in r["records_hatD"]]) for r in split_results if r["records_hatD"]]
    overall_area_h = [np.mean([rr["area"] for rr in r["records_hatD"]]) for r in split_results if r["records_hatD"]]
    overall_o = oracle["overall_oracle_mondrian"]

    print("\n" + "=" * 90)
    print("B9. Overall / marginal (secondary)")
    print("=" * 90)
    print(f"GLOBAL:                     coverage={np.mean(overall_cov_g):.4f}+/-{np.std(overall_cov_g):.4f}  "
          f"area={np.mean(overall_area_g):.1f}+/-{np.std(overall_area_g):.1f}")
    print(f"ORACLE MONDRIAN (true D):   coverage={overall_o['coverage_mean']:.4f}+/-{overall_o['coverage_std']:.4f}  "
          f"area={overall_o['area_mean']:.1f}+/-{overall_o['area_std']:.1f}")
    print(f"ESTIMATED-SEP. MONDRIAN:    coverage={np.mean(overall_cov_h):.4f}+/-{np.std(overall_cov_h):.4f}  "
          f"area={np.mean(overall_area_h):.1f}+/-{np.std(overall_area_h):.1f}")

    # ---- B10: coverage heterogeneity ----
    cov_g_by_D = [b7_rows[D]["cov_g"] for D in separations]
    cov_o_by_D = [b7_rows[D]["cov_o"] for D in separations]
    cov_h_by_D = [b7_rows[D]["cov_h"] for D in separations]
    spread_g = max(cov_g_by_D) - min(cov_g_by_D)
    spread_o = max(cov_o_by_D) - min(cov_o_by_D)
    spread_h = max(cov_h_by_D) - min(cov_h_by_D)
    cov_h_by_G = [b8_rows[j]["cov_h"] for j in range(N_BINS) if not np.isnan(b8_rows[j]["cov_h"])]
    spread_h_byG = max(cov_h_by_G) - min(cov_h_by_G) if cov_h_by_G else float("nan")

    print("\n" + "=" * 90)
    print("B10. Coverage heterogeneity (max - min coverage)")
    print("=" * 90)
    print(f"  across true D:      Global={spread_g:.3f}   Oracle Mondrian={spread_o:.3f}   "
          f"Estimated-sep Mondrian={spread_h:.3f}")
    print(f"  across Delta_hat G categories (estimated-sep Mondrian only): {spread_h_byG:.3f}")

    # ---- B11: interpretation ----
    print("\n" + "=" * 90)
    print("B11. Interpretation (data-driven)")
    print("=" * 90)
    print(f"1/2/3. D=10: Global={b7_rows[10.0]['cov_g']:.3f} Oracle={b7_rows[10.0]['cov_o']:.3f} "
          f"hatD={b7_rows[10.0]['cov_h']:.3f}   |   "
          f"D=15: Global={b7_rows[15.0]['cov_g']:.3f} Oracle={b7_rows[15.0]['cov_o']:.3f} "
          f"hatD={b7_rows[15.0]['cov_h']:.3f}")
    print(f"4. D=5: Global={b7_rows[5.0]['cov_g']:.3f} Oracle={b7_rows[5.0]['cov_o']:.3f} "
          f"hatD={b7_rows[5.0]['cov_h']:.3f}")
    print(f"5. D=60: Global={b7_rows[60.0]['cov_g']:.3f} Oracle={b7_rows[60.0]['cov_o']:.3f} "
          f"hatD={b7_rows[60.0]['cov_h']:.3f}")
    print("6. Area cost/savings: see d_area column in B7 table above.")
    print(f"7. Within-category (Delta_hat) hatD coverage, all 8 G's: "
          + ", ".join(f"G{j+1}={b8_rows[j]['cov_h']:.3f}" for j in range(N_BINS)))
    print(f"8. True-D coverage spread: Global={spread_g:.3f} Oracle={spread_o:.3f} hatD={spread_h:.3f}")
    print(f"9. G8=[63,200) coverage: hatD={b8_rows[N_BINS-1]['cov_h']:.3f} "
          f"(n_test={b8_rows[N_BINS-1]['n_test']:.0f}, n_calib mean={b8_rows[N_BINS-1]['n_calib_mean']:.1f})")
    print("10. Performance loss vs Oracle: compare cov_o vs cov_h and area_o vs area_h per D in the B7 table.")

    # ---- save ----
    out_txt = os.path.join(args.out_dir, "results_deployable_mondrian_separation.txt")
    with open(out_txt, "w") as fh:
        fh.write("DEPLOYABLE Mondrian CP using estimated separation Delta_hat (frozen bins from Diagnostic 2)\n")
        fh.write("Category assignment uses ONLY model estimates; true D used for evaluation only.\n\n")
        fh.write("Bins: " + "; ".join(BIN_LABELS) + "\n\n")
        fh.write(f"all_feasible={feasible_all}  n_infeasible=({len(all_infeasible)})\n\n")
        fh.write("B7 -- by true D\n")
        fh.write(f"{'D':>5} {'cov_global':>11} {'cov_oracle':>11} {'cov_hatD':>10} {'d_cov':>8} "
                 f"{'area_global':>12} {'area_oracle':>12} {'area_hatD':>11} {'d_area':>9} "
                 f"{'area%_global':>13} {'area%_hatD':>11}\n")
        for D in separations:
            row = b7_rows[D]
            apg = row["area_g"] / n_grid_cells * 100
            aph = row["area_h"] / n_grid_cells * 100 if not np.isnan(row["area_h"]) else float("nan")
            fh.write(f"{D:5.1f} {row['cov_g']:6.3f}+/-{row['cov_g_std']:5.3f} "
                     f"{row['cov_o']:6.3f}+/-{row['cov_o_std']:5.3f} "
                     f"{row['cov_h']:6.3f}+/-{row['cov_h_std']:5.3f} "
                     f"{row['d_cov']:+7.3f} "
                     f"{row['area_g']:7.1f}+/-{row['area_g_std']:6.1f} "
                     f"{row['area_o']:7.1f}+/-{row['area_o_std']:6.1f} "
                     f"{row['area_h']:7.1f}+/-{row['area_h_std']:6.1f} "
                     f"{row['d_area']:+8.1f} {apg:12.2f} {aph:11.2f}\n")
        fh.write(f"\noverall_global: coverage={np.mean(overall_cov_g):.4f}+/-{np.std(overall_cov_g):.4f} "
                 f"area={np.mean(overall_area_g):.1f}+/-{np.std(overall_area_g):.1f}\n")
        fh.write(f"overall_oracle_mondrian: coverage={overall_o['coverage_mean']:.4f}+/-{overall_o['coverage_std']:.4f} "
                 f"area={overall_o['area_mean']:.1f}+/-{overall_o['area_std']:.1f}\n")
        fh.write(f"overall_estimated_sep_mondrian: coverage={np.mean(overall_cov_h):.4f}+/-{np.std(overall_cov_h):.4f} "
                 f"area={np.mean(overall_area_h):.1f}+/-{np.std(overall_area_h):.1f}\n")
        fh.write(f"\ncoverage_spread(max-min over true D): global={spread_g:.3f} oracle={spread_o:.3f} hatD={spread_h:.3f}\n")
        fh.write(f"coverage_spread(max-min over Delta_hat G categories, hatD): {spread_h_byG:.3f}\n")

        fh.write("\nB8 -- by observable Delta_hat category\n")
        fh.write(f"{'cat':>4} {'n_test':>7} {'cov_global':>11} {'cov_hatD':>10} "
                 f"{'area_g_mean':>12} {'area_g_med':>11} {'area_h_mean':>12} {'area_h_med':>11} "
                 f"{'n_calib_min':>11} {'n_calib_mean':>13} {'n_calib_max':>12}\n")
        for j in range(N_BINS):
            r = b8_rows[j]
            fh.write(f"{('G'+str(j+1)):>4} {r['n_test']:7.0f} {r['cov_g']:11.3f} {r['cov_h']:10.3f} "
                     f"{r['area_g']:12.1f} {r['area_g_med']:11.1f} {r['area_h']:12.1f} {r['area_h_med']:11.1f} "
                     f"{r['n_calib_min']:11d} {r['n_calib_mean']:13.1f} {r['n_calib_max']:12d}\n")
    print(f"\nSaved -> {out_txt}")

    out_json = os.path.join(args.out_dir, "deployable_mondrian_separation_raw.json")
    with open(out_json, "w") as fh:
        json.dump(dict(
            args=vars(args), bins_deg=BIN_EDGES_DEG,
            all_infeasible=[dict(split=i, G=j, reason=reason) for i, j, reason in all_infeasible],
            all_categories_feasible_all_splits=feasible_all,
            b7_by_true_D={str(D): v for D, v in b7_rows.items()},
            b8_by_category={str(j): v for j, v in b8_rows.items()},
            overall_global=dict(coverage_mean=float(np.mean(overall_cov_g)), coverage_std=float(np.std(overall_cov_g)),
                                 area_mean=float(np.mean(overall_area_g)), area_std=float(np.std(overall_area_g))),
            overall_estimated_sep_mondrian=dict(coverage_mean=float(np.mean(overall_cov_h)), coverage_std=float(np.std(overall_cov_h)),
                                                 area_mean=float(np.mean(overall_area_h)), area_std=float(np.std(overall_area_h))),
            coverage_spread_true_D=dict(global_=spread_g, oracle=spread_o, hatD=spread_h),
            coverage_spread_by_G_hatD=spread_h_byG,
        ), fh, indent=2)
    print(f"Saved -> {out_json}")


if __name__ == "__main__":
    main()
