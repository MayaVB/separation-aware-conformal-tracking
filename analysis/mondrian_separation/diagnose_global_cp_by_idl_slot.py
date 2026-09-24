"""
Decompose the existing Global-CP angular-separation experiment (Table 1,
Results/results_angular_separation.txt) by IDL detection slot k=0/k=1.

Reporting/diagnostic only. Reuses, unmodified:
  - eval_angular_separation.load_conditions / pool_scenes (identical scene
    splits and frame sampling for the same split_seed)
  - Code.two_speaker_tracking.lcp.calibrate_global_lambda_from_arrays
    (the SAME Global-CP calibration call Table 1 used -- one threshold per
    k, pooled across ALL 7 true-D conditions' calibration frames, exactly
    as eval_angular_separation.run_one_split does)
  - Code.crc_ssl.CoverageSet (matching + region growing, read-only)
  - Code.two_speaker_tracking.eval_metrics.wrap_azi_err_deg (the same DOA
    error definition used for Table 1's E_mean column)

No recalibration, no new algorithm, no change to matching/region
construction, no new Mondrian grouping. The only addition versus
eval_angular_separation.run_one_split is tagging each (frame, speaker)
test record with its IDL slot k = est_s (the raw estimated-array index,
i.e. k=0 = detected from the ORIGINAL spectrum, k=1 = detected from the
POST-CANCELLATION residual spectrum -- see Module.py's IDL loop) and its
DOA error, so the SAME test records already used for Table 1's
coverage/area can be broken down by slot.

IMPORTANT caveat on DOA error: Table 1's own E_mean/E_median/etc. columns
come from a DIFFERENT sampling procedure (eval_angular_separation.py's
Step 2.1/2.2, build_frame_records: one single global sample pooling ALL
common scenes with n_frames_per_scene=25, unrelated to the 20 calib/test
splits or to Global-CP thresholds at all -- it is pure descriptive DOA
error, no CP machinery involved). This script's DOA-error numbers are
instead computed on the SAME 20-split TEST records used for
coverage/area (n_test_frames_per_scene=10, test-scenes only), so they are
a closely related but not bit-identical quantity to Table 1's E_mean --
documented explicitly in the printed output, not silently conflated.
Coverage and area, by contrast, are computed via the EXACT SAME code path
as Table 1 and are verified to reproduce it exactly (see section "pooled
vs Table 1" in the output).
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
from Code.two_speaker_tracking.eval_metrics import wrap_azi_err_deg

from eval_angular_separation import load_conditions, pool_scenes, DEFAULT_SEPARATIONS
from eval_oracle_mondrian_angular_separation import DEFAULT_DATA_PATHS

# Table 1 reference values (Results/results_angular_separation.txt), for the
# numerical reproduction check.
TABLE1 = {
    5.0:  dict(cov=0.930, err=13.66, area=1051.4),
    10.0: dict(cov=0.756, err=10.53, area=649.2),
    15.0: dict(cov=0.828, err=9.27,  area=303.2),
    20.0: dict(cov=0.888, err=6.73,  area=169.6),
    30.0: dict(cov=0.932, err=3.59,  area=132.3),
    45.0: dict(cov=0.940, err=4.74,  area=177.0),
    60.0: dict(cov=0.982, err=3.05,  area=140.8),
}


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


def run_one_split_by_slot(conds, separations, common_scenes, room, nele, nazi, lambda_list, args, split_seed):
    """Bit-identical to eval_angular_separation.run_one_split's calibration
    and test-record computation (same rng sequence, same
    calibrate_global_lambda_from_arrays call, same matching, same
    neighbours_coverage_set call) -- the only addition is recording k and
    DOA error per test record."""
    rng = np.random.default_rng(split_seed)
    scene_order = rng.permutation(common_scenes)
    n_calib = int(round(len(scene_order) * args.calib_scene_frac))
    calib_scenes, test_scenes = scene_order[:n_calib], scene_order[n_calib:]

    lm_c, est_c, true_c, _ = pool_scenes(conds, separations, calib_scenes,
                                          args.n_calib_frames_per_scene, rng)
    lambdas_global = calibrate_global_lambda_from_arrays(lm_c, est_c, true_c, room, lambda_list, args.alpha)

    lm_t, est_t, true_t, D_t = pool_scenes(conds, separations, test_scenes,
                                            args.n_test_frames_per_scene, rng)

    records = []
    for i in range(lm_t.shape[0]):
        true_order, est_order = CoverageSet._match_estimated_to_source(true_t[i], est_t[i])
        for true_s, est_s in zip(true_order, est_order):
            k = int(est_s)
            norm_map = normalize(lm_t[i, k])
            seed = tuple(radians_to_grid_index(est_t[i, k], nele, nazi).astype(int))
            true_idx = tuple(radians_to_grid_index(true_t[i, true_s], nele, nazi).astype(int))
            region = CoverageSet.neighbours_coverage_set(norm_map, float(lambdas_global[k]), estimated_position=seed)
            e_deg = float(wrap_azi_err_deg(est_t[i, k, 1], true_t[i, true_s, 1]))
            records.append(dict(D=float(D_t[i]), k=k, covered=bool(region[true_idx]),
                                 area=int(region.sum()), e_deg=e_deg))

    return dict(lambdas_global=lambdas_global, records=records)


def per_split_summary(records, D, k=None):
    if k is None:
        recs = [r for r in records if r["D"] == D]
    else:
        recs = [r for r in records if r["D"] == D and r["k"] == k]
    if not recs:
        return None
    return dict(coverage=float(np.mean([r["covered"] for r in recs])),
                area=float(np.mean([r["area"] for r in recs])),
                e_deg=float(np.mean([r["e_deg"] for r in recs])),
                n=len(recs))


def agg(per_split_vals, key):
    vals = [v[key] for v in per_split_vals if v is not None]
    return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else (float("nan"), float("nan"), 0)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    separations = args.separations

    conds, nele, nazi, common_scenes = load_conditions(args.data_paths, separations)
    room = _build_room(conds[separations[0]])
    n_grid_cells = nele * nazi
    lambda_list = np.linspace(0.0, 1.0, args.lambda_steps)

    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 100)
    out("Global-CP Table 1 decomposition by IDL slot (k=0: original-spectrum detection; "
        "k=1: post-cancellation residual detection)")
    out(f"Same {args.n_splits} scene-disjoint splits / calibration protocol / Global-CP thresholds / "
        "matching / region construction as Table 1. Reporting only -- no recalibration, no new method.")
    out("=" * 100)

    split_results = []
    for i in range(args.n_splits):
        res = run_one_split_by_slot(conds, separations, common_scenes, room, nele, nazi,
                                     lambda_list, args, args.seed + i)
        split_results.append(res)
    out(f"\nCompleted {args.n_splits} splits.")

    K = split_results[0]["lambdas_global"].shape[0]

    # =====================================================================
    # pooled (k ignored) vs Table 1
    # =====================================================================
    out("\n" + "=" * 100)
    out("Pooled (k0+k1) vs Table 1 -- verifies this diagnostic reproduces the existing experiment")
    out("=" * 100)
    out(f"{'D':>5} {'cov_pooled':>11} {'cov_table1':>11} {'d_cov':>9} "
        f"{'area_pooled':>12} {'area_table1':>12} {'d_area':>9} "
        f"{'err_pooled(this test-split pop.)':>34} {'err_table1(diff sample)':>24}")
    pooled_rows = {}
    for D in separations:
        per_split = [per_split_summary(r["records"], D, k=None) for r in split_results]
        cov_m, cov_s, n = agg(per_split, "coverage")
        area_m, area_s, _ = agg(per_split, "area")
        err_m, err_s, _ = agg(per_split, "e_deg")
        pooled_rows[D] = dict(cov=cov_m, cov_std=cov_s, area=area_m, area_std=area_s, err=err_m, err_std=err_s)
        t1 = TABLE1[D]
        out(f"{D:5.1f} {cov_m:11.4f} {t1['cov']:11.4f} {cov_m-t1['cov']:+9.4f} "
            f"{area_m:12.2f} {t1['area']:12.2f} {area_m-t1['area']:+9.2f} "
            f"{err_m:34.3f} {t1['err']:24.3f}")
    out("\n(cov/area should match Table 1 to within float/lambda-grid rounding -- both use the exact "
        "same calibration call and region-growing code. err_pooled differs from Table 1's E_mean by "
        "construction -- see module docstring: different sampling procedure, not a discrepancy.)")

    # =====================================================================
    # by-slot compact table
    # =====================================================================
    out("\n" + "=" * 100)
    out("COMPACT TABLE -- Global CP by IDL slot, mean +/- std across splits")
    out("=" * 100)
    header = (f"{'D':>5} | {'Cov k0':>13} | {'Cov k1':>13} | {'DOAerr k0':>13} | {'DOAerr k1':>13} | "
              f"{'Area k0':>15} | {'Area k1':>15} | {'Area% k0':>9} | {'Area% k1':>9}")
    out(header)
    by_slot_rows = {}
    for D in separations:
        row = {}
        for k in range(K):
            per_split = [per_split_summary(r["records"], D, k=k) for r in split_results]
            cov_m, cov_s, n = agg(per_split, "coverage")
            area_m, area_s, _ = agg(per_split, "area")
            err_m, err_s, _ = agg(per_split, "e_deg")
            row[k] = dict(cov=cov_m, cov_std=cov_s, area=area_m, area_std=area_s,
                          err=err_m, err_std=err_s, n_splits=n)
        by_slot_rows[D] = row
        out(f"{D:5.1f} | {row[0]['cov']:6.3f}+/-{row[0]['cov_std']:5.3f} | "
            f"{row[1]['cov']:6.3f}+/-{row[1]['cov_std']:5.3f} | "
            f"{row[0]['err']:6.2f}+/-{row[0]['err_std']:5.2f} | "
            f"{row[1]['err']:6.2f}+/-{row[1]['err_std']:5.2f} | "
            f"{row[0]['area']:7.1f}+/-{row[0]['area_std']:6.1f} | "
            f"{row[1]['area']:7.1f}+/-{row[1]['area_std']:6.1f} | "
            f"{row[0]['area']/n_grid_cells*100:8.2f} | {row[1]['area']/n_grid_cells*100:8.2f}")

    # =====================================================================
    # lambda_global[k]
    # =====================================================================
    out("\n" + "=" * 100)
    out("Global CP threshold lambda[k], mean +/- std across splits (D-agnostic by construction -- "
        "Global CP calibrates ONE threshold per k, pooled across all 7 true-D conditions per split)")
    out("=" * 100)
    lam_by_k = {k: [float(r["lambdas_global"][k]) for r in split_results] for k in range(K)}
    for k in range(K):
        out(f"  k={k}: lambda_global = {np.mean(lam_by_k[k]):.4f} +/- {np.std(lam_by_k[k]):.4f}")

    # =====================================================================
    # interpretation
    # =====================================================================
    out("\n" + "=" * 100)
    out("INTERPRETATION")
    out("=" * 100)
    out("k=0/k=1 are IDL DETECTION ITERATIONS, not persistent physical speaker identities: k=0 is "
        "detected from the original spectrum, k=1 from the spectrum AFTER subtracting a fitted "
        "template for k=0 (see Module.py's IDL loop / diagnose_cancellation_tier0.py).")

    def cov(D, k): return by_slot_rows[D][k]["cov"]
    def area(D, k): return by_slot_rows[D][k]["area"]
    def areapct(D, k): return by_slot_rows[D][k]["area"] / n_grid_cells * 100

    out(f"\n1. D=10/D=15 undercoverage primarily k=1?  "
        f"D=10: cov_k0={cov(10.0,0):.3f} cov_k1={cov(10.0,1):.3f}  |  "
        f"D=15: cov_k0={cov(15.0,0):.3f} cov_k1={cov(15.0,1):.3f}")
    out(f"2. Is k=0 approximately D-agnostic / calibrated?  cov_k0 across D: "
        + ", ".join(f"D={D:.0f}:{cov(D,0):.3f}" for D in separations)
        + f"   (spread={max(cov(D,0) for D in separations)-min(cov(D,0) for D in separations):.3f})")
    out(f"   cov_k1 across D: " + ", ".join(f"D={D:.0f}:{cov(D,1):.3f}" for D in separations)
        + f"   (spread={max(cov(D,1) for D in separations)-min(cov(D,1) for D in separations):.3f})")
    out(f"3. D=5 huge area -- k0 vs k1?  area_k0={area(5.0,0):.1f} ({areapct(5.0,0):.2f}%)  "
        f"area_k1={area(5.0,1):.1f} ({areapct(5.0,1):.2f}%)")
    out("4. k1 systematically larger regions than k0 across all D?  area_k1 - area_k0 by D: "
        + ", ".join(f"D={D:.0f}:{area(D,1)-area(D,0):+.1f}" for D in separations))
    out("5. See numbers above -- summarized in the accompanying chat message.")

    # ---- save ----
    out_txt = os.path.join(args.out_dir, "global_cp_by_idl_slot.txt")
    with open(out_txt, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nSaved -> {out_txt}")

    out_json = os.path.join(args.out_dir, "global_cp_by_idl_slot_raw.json")
    with open(out_json, "w") as fh:
        json.dump(dict(
            args=vars(args), n_grid_cells=n_grid_cells,
            pooled_vs_table1={str(D): v for D, v in pooled_rows.items()},
            by_slot={str(D): {str(k): v for k, v in row.items()} for D, row in by_slot_rows.items()},
            lambda_global_mean={str(k): float(np.mean(v)) for k, v in lam_by_k.items()},
            lambda_global_std={str(k): float(np.std(v)) for k, v in lam_by_k.items()},
        ), fh, indent=2)
    print(f"Saved -> {out_json}")


if __name__ == "__main__":
    main()
