"""
Part A diagnostic: sanity-check the two suspicious patterns in the ORACLE
Mondrian result (Results/results_oracle_mondrian_angular_separation.txt):

  A1. D=30 and D=45 have extremely large split-to-split AREA std
      (194.7+/-302.4 and 135.4+/-238.0).
  A2. D=60 has an extremely small mean region area (~2.9 cells).

Diagnostic only -- does NOT change the CP score, water-filling/region-growing
algorithm, calibration procedure, matching convention, or splits. Reuses,
unmodified:
  - eval_angular_separation.load_conditions / pool_scenes (bit-identical
    scene splits and frame sampling to the already-reported Oracle Mondrian
    run, for the same split_seed -- pool_scenes is always called with the
    FULL 7-condition `separations` list, exactly as the production run does,
    so the rng draw sequence -- and therefore which frames get sampled --
    is unchanged; we only filter to D in {30, 45, 60} AFTER sampling)
  - Code.two_speaker_tracking.lcp.calibrate_global_lambda_from_arrays (the
    same Mondrian-per-D calibration call the production run used)
  - Code.two_speaker_tracking.lcp.compute_calibration_scores (already-
    validated V = -lambda_star per calibration record; the standard
    per-record nonconformity score this codebase already uses elsewhere
    for the SAME calibration procedure -- used here only to describe the
    score distribution that the CRC threshold search operates over, not
    to select the threshold itself)
  - Code.crc_ssl.CoverageSet (region growing / matching, read-only)

This independently reproduces lambda_mondrian[D=60] from the production run
as a cross-check: if the two scripts disagree, that is itself evidence of a
bug; if they agree, it is a consistency check that the production numbers
are reproducible from the same deterministic splits.
"""

import argparse
import os

import numpy as np

from Code.crc_ssl import CoverageSet
from Code.utilities import normalize
from Code.two_speaker_tracking.npz_adapter import radians_to_grid_index, _build_room
from Code.two_speaker_tracking.lcp import calibrate_global_lambda_from_arrays, compute_calibration_scores

from eval_angular_separation import load_conditions, pool_scenes, DEFAULT_SEPARATIONS
from eval_oracle_mondrian_angular_separation import DEFAULT_DATA_PATHS

TARGET_DS = [30.0, 45.0, 60.0]


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
    return p.parse_args()


def run_split(conds, separations, common_scenes, room, nele, nazi, lambda_list, args, split_seed):
    """Bit-identical calib/test scene split + frame sampling to
    eval_oracle_mondrian_angular_separation.oracle_mondrian_split for the
    same split_seed (same rng call sequence). Only Mondrian calibration +
    test eval for D in TARGET_DS is computed here (Global and the other 4
    D's are skipped -- diagnostic only, doesn't affect what's sampled)."""
    rng = np.random.default_rng(split_seed)
    scene_order = rng.permutation(common_scenes)
    n_calib = int(round(len(scene_order) * args.calib_scene_frac))
    calib_scenes, test_scenes = scene_order[:n_calib], scene_order[n_calib:]

    lm_c, est_c, true_c, D_c = pool_scenes(conds, separations, calib_scenes,
                                            args.n_calib_frames_per_scene, rng)
    lm_t, est_t, true_t, D_t = pool_scenes(conds, separations, test_scenes,
                                            args.n_test_frames_per_scene, rng)

    out = {}
    for D in TARGET_DS:
        mask_c = (D_c == D)
        n_calib_D = int(mask_c.sum())
        lam_D = calibrate_global_lambda_from_arrays(
            lm_c[mask_c], est_c[mask_c], true_c[mask_c], room, lambda_list, args.alpha)
        V = compute_calibration_scores(lm_c[mask_c], est_c[mask_c], true_c[mask_c],
                                        lambda_list, nele, nazi)  # (n_calib_D, K)
        lambda_star = -V  # (n_calib_D, K), higher = easier (needs less permissive region)

        mask_t = (D_t == D)
        idxs_t = np.where(mask_t)[0]
        records = []
        for i in idxs_t:
            true_order, est_order = CoverageSet._match_estimated_to_source(true_t[i], est_t[i])
            for true_s, est_s in zip(true_order, est_order):
                k = int(est_s)
                norm_map = normalize(lm_t[i, k])
                seed = tuple(radians_to_grid_index(est_t[i, k], nele, nazi).astype(int))
                true_idx = tuple(radians_to_grid_index(true_t[i, true_s], nele, nazi).astype(int))
                region = CoverageSet.neighbours_coverage_set(norm_map, float(lam_D[k]), estimated_position=seed)
                records.append(dict(
                    k=k, area=int(region.sum()), covered=bool(region[true_idx]),
                    est_deg=np.degrees(est_t[i, k]).tolist(), true_deg=np.degrees(true_t[i, true_s]).tolist(),
                ))
        out[D] = dict(n_calib=n_calib_D, lam=lam_D, lambda_star=lambda_star, records=records)
    return out


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def main():
    args = parse_args()
    separations = args.separations
    conds, nele, nazi, common_scenes = load_conditions(args.data_paths, separations)
    room = _build_room(conds[separations[0]])
    lambda_list = np.linspace(0.0, 1.0, args.lambda_steps)

    print("=" * 90)
    print("PART A -- Oracle Mondrian sanity/variance diagnostic (D=30, D=45, D=60 only)")
    print("Diagnostic only. No algorithm/calibration/split change.")
    print("=" * 90)

    all_splits = []
    for i in range(args.n_splits):
        res = run_split(conds, separations, common_scenes, room, nele, nazi, lambda_list, args, args.seed + i)
        all_splits.append(res)
        print(f"  split {i:3d} done  "
              + "  ".join(f"D={D:.0f}:n_calib={res[D]['n_calib']}" for D in TARGET_DS))

    K = all_splits[0][TARGET_DS[0]]["lam"].shape[0]

    # =====================================================================
    # A1: D=30 / D=45 split-level tables
    # =====================================================================
    for D in [30.0, 45.0]:
        print("\n" + "=" * 90)
        print(f"A1. D={D:.0f} split-level table")
        print("=" * 90)
        for k in range(K):
            print(f"\n--- speaker slot k={k} ---")
            print(f"{'split':>5} {'lambda':>7} {'cov':>6} {'mean_area':>10} {'med_area':>9} {'area_p90':>9} "
                  f"{'ls_min':>7} {'ls_p10':>7} {'ls_med':>7} {'ls_max':>7} {'n_calib':>7}")
            rows = []
            for i, res in enumerate(all_splits):
                r = res[D]
                recs_k = [rr for rr in r["records"] if rr["k"] == k]
                areas = [rr["area"] for rr in recs_k]
                covs = [rr["covered"] for rr in recs_k]
                ls = r["lambda_star"][:, k]
                ls = ls[np.isfinite(ls)]
                row = dict(
                    split=i, lam=float(r["lam"][k]),
                    cov=float(np.mean(covs)) if covs else float("nan"),
                    mean_area=float(np.mean(areas)) if areas else float("nan"),
                    med_area=float(np.median(areas)) if areas else float("nan"),
                    p90_area=pct(areas, 90),
                    ls_min=float(ls.min()) if len(ls) else float("nan"),
                    ls_p10=pct(ls, 10),
                    ls_med=float(np.median(ls)) if len(ls) else float("nan"),
                    ls_max=float(ls.max()) if len(ls) else float("nan"),
                    n_calib=len(ls),
                )
                rows.append(row)
                print(f"{row['split']:5d} {row['lam']:7.4f} {row['cov']:6.3f} {row['mean_area']:10.1f} "
                      f"{row['med_area']:9.1f} {row['p90_area']:9.1f} {row['ls_min']:7.4f} {row['ls_p10']:7.4f} "
                      f"{row['ls_med']:7.4f} {row['ls_max']:7.4f} {row['n_calib']:7d}")

            top3 = sorted(rows, key=lambda r: r["mean_area"], reverse=True)[:3]
            print(f"\n  Top-3 highest-mean-area splits for D={D:.0f}, k={k}:")
            for row in top3:
                print(f"    split={row['split']:3d}  lambda={row['lam']:.4f}  coverage={row['cov']:.3f}  "
                      f"mean_area={row['mean_area']:.1f}  median_area={row['med_area']:.1f}  "
                      f"area_p90={row['p90_area']:.1f}  |  calib lambda_star: "
                      f"min={row['ls_min']:.4f} p10={row['ls_p10']:.4f} median={row['ls_med']:.4f} "
                      f"max={row['ls_max']:.4f} (n={row['n_calib']})")

    # =====================================================================
    # A2: D=60 tiny-region sanity check
    # =====================================================================
    print("\n" + "=" * 90)
    print("A2. D=60 tiny-region sanity check")
    print("=" * 90)

    D = 60.0
    all_records = []
    for res in all_splits:
        all_records.extend(res[D]["records"])
    areas_all = np.array([r["area"] for r in all_records])
    print(f"\nTotal D=60 test records across {args.n_splits} splits: {len(all_records)}")
    print(f"  overall: mean={areas_all.mean():.2f} median={np.median(areas_all):.2f} "
          f"P10={pct(areas_all,10):.2f} P50={pct(areas_all,50):.2f} P90={pct(areas_all,90):.2f}")
    for thresh in (1, 2, 3, 5, 10):
        frac = float(np.mean(areas_all <= thresh))
        print(f"  fraction area <= {thresh:2d} cells: {frac:.4f}")

    for k in range(K):
        areas_k = np.array([r["area"] for r in all_records if r["k"] == k])
        print(f"\n  speaker k={k}: n={len(areas_k)} mean={areas_k.mean():.2f} median={np.median(areas_k):.2f} "
              f"P10={pct(areas_k,10):.2f} P50={pct(areas_k,50):.2f} P90={pct(areas_k,90):.2f}")
        for thresh in (1, 2, 3, 5, 10):
            frac = float(np.mean(areas_k <= thresh))
            print(f"    fraction area <= {thresh:2d} cells: {frac:.4f}")

    print("\n  D=60 lambda_mondrian distribution across splits (cross-check vs production run):")
    for k in range(K):
        lam_k = np.array([res[D]["lam"][k] for res in all_splits])
        print(f"    k={k}: mean={lam_k.mean():.4f} std={lam_k.std():.4f} min={lam_k.min():.4f} max={lam_k.max():.4f}")
    print("  Production run reported: k=0 mean=0.9914 std=0.0076 ; k=1 mean=0.9464 std=0.0677")

    # representative examples
    print("\n  Representative D=60 examples:")
    covered_areas = [(r["area"], r) for r in all_records if r["covered"]]
    missed = [(r["area"], r) for r in all_records if not r["covered"]]
    med = float(np.median(areas_all))

    def fmt(tag, r, area):
        est = r["est_deg"]
        true = r["true_deg"]
        print(f"    [{tag}] k={r['k']} area={area} covered={r['covered']}  "
              f"est=[ele={est[0]:.1f}, azi={est[1]:.1f}]deg  true=[ele={true[0]:.1f}, azi={true[1]:.1f}]deg")

    if covered_areas:
        area_c, rec_c = min(covered_areas, key=lambda t: t[0])
        fmt("smallest COVERED region", rec_c, area_c)
    else:
        print("    (no covered D=60 records found)")
    if missed:
        area_m, rec_m = min(missed, key=lambda t: t[0])
        fmt("smallest MISSED region", rec_m, area_m)
    else:
        print("    (no missed/uncovered D=60 records found)")
    closest = min(all_records, key=lambda r: abs(r["area"] - med))
    fmt(f"near-median region (median={med:.1f})", closest, closest["area"])

    print("\n" + "=" * 90)
    print("Part A complete. See findings summary in the accompanying chat message.")
    print("=" * 90)


if __name__ == "__main__":
    main()
