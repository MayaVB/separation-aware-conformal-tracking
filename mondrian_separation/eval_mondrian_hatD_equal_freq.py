"""
DEPLOYABLE Mondrian CP using equal-frequency bins of the observable estimated
angular separation hatD ("Mondrian-hatD"), swept over category count
M = 2, 3, 4, 5.

This is the "clean" Mondrian-hatD baseline, distinct from the two prior
Mondrian scripts in this repo:

  - eval_oracle_mondrian_angular_separation.py  ("Oracle Mondrian"): uses the
    TRUE commanded separation D as the category. Not deployable (D is
    unknown at inference). Kept only as a comparator, loaded read-only from
    its own saved Results/oracle_mondrian_angular_separation_raw.json --
    NOT recomputed here.
  - eval_deployable_mondrian_separation.py ("manual-hatD Mondrian"): also
    uses an observable estimated separation, but with 8 FIXED, manually
    chosen bin edges in degrees (frozen from an earlier ad hoc diagnostic).
    Kept only as a comparator, loaded read-only from its own saved
    Results/deployable_mondrian_separation_raw.json -- NOT recomputed here.

This script differs from both in the *category construction*: bin edges are
derived automatically, per split, as equal-frequency (quantile) cuts of the
hatD values observed in that split's CALIBRATION frames only -- never from
true D, never from lambda_star, never from test data. M in {2,3,4,5} is
swept and all four are reported; no "best M" is chosen here (see NOTE on
model selection below).

============================================================
1. THE MONDRIAN VARIABLE: hatD
============================================================
hatD is a frame-level, GT-free statistic: the great-circle angular
separation between the two IDL slots' ESTIMATED DOAs in that frame (slot 0
= detection off the original spectrum, slot 1 = detection off the
post-cancellation residual spectrum -- see Module.py's IDL loop and
diagnose_global_cp_by_idl_slot.py). It uses only
`all_estimated_positions[:, 0, :]` and `[:, 1, :]`, in raw (unmatched)
slot order -- never `speaker_pos` (ground truth) and never lambda_star.
hatD is shared by k=0 and k=1 (it is a property of the frame's two-source
estimate configuration, not of either slot individually).

The great-circle formula is REUSED, not reimplemented, from
eval_deployable_mondrian_separation.delta_hat_deg_batch (spherical law of
cosines): for [ele, azi] radians (th0,az0), (th1,az1),
    cos(hatD) = cos(th0)cos(th1) + sin(th0)sin(th1)cos(az0-az1)
    hatD = arccos(clip(cos(hatD), -1, 1))   [degrees]

============================================================
2. EQUAL-FREQUENCY BINS, TIE-SAFE (see compute_equal_freq_boundaries)
============================================================
For a given split and M, boundaries are derived ONLY from that split's
CALIBRATION hatD values (never test data, never true D, never lambda_star):

  1. unique_vals = sorted distinct values actually observed in hatD_calib.
  2. For each of the M-1 target equal-frequency cut fractions q_i = i/M
     (i = 1..M-1), compute the RAW quantile value via
     np.quantile(hatD_calib, q_i, method="linear") -- the standard
     "each category gets ~1/M of the calibration mass" definition.
  3. Snap q_i's raw quantile to the MIDPOINT of the gap between the two
     DISTINCT unique_vals that bracket it: j = searchsorted(unique_vals,
     raw_q, side="left"), clipped to [1, len(unique_vals)-1], boundary_i =
     (unique_vals[j-1] + unique_vals[j]) / 2. A boundary produced this way
     always falls strictly BETWEEN two distinct observed hatD values, so a
     group of frames sharing an identical hatD value is never split across
     two Mondrian categories.
  4. Deduplicate + sort the M-1 candidate boundaries. Two different target
     quantiles can snap to the same inter-value gap only when the
     calibration sample has too few distinct hatD values relative to M
     (heavy ties, or M close to the number of distinct values); when that
     happens the boundary is kept once and the REALIZED number of
     categories is < M. This is reported explicitly per split/M (see
     Section 7 diagnostics) -- never silently padded back to M.

The same boundaries (frozen after step 2 above) are applied to both
calibration (for calibrating lambda[m,k]) and test (for evaluation) hatD
values, and are shared by k=0 and k=1 as required.

============================================================
3. CALIBRATION
============================================================
Within each realized category m, the two IDL slots are calibrated
SEPARATELY: lambda[m,0], lambda[m,1] = calibrate_global_lambda_from_arrays(
lm_c[mask], est_c[mask], true_c[mask], room, lambda_list, alpha), called on
the category's calibration subset only. This is the EXACT SAME finite-sample
conformal calibration call (Code.two_speaker_tracking.lcp) used by Global CP
and both prior Mondrian scripts -- lambda_star computation, matching,
region-growing, alpha and the finite-sample correction are all unchanged.
The only methodological change from Global CP is indexing lambda by
category m in addition to slot k: lambda[k] -> lambda[m,k].

Finite-sample feasibility: Code.crc_ssl.CoverageSet._calc_conformal_risk_control
raises ValueError when a category's calibration pool is too small for alpha
(n_calib <= 8 for alpha=0.1, since the correction term 1/(n_calib+1) must be
<= alpha). This is caught per (split, M, category), reported, and that
(split, M, category) is EXCLUDED from Mondrian-hatD aggregation for that M --
no fallback to Global CP, no merging of categories, no silent skip.

============================================================
4. SPLITS
============================================================
Uses the exact same 20 scene-disjoint splits (same np.random.default_rng
per-split seeding, same scene_order permutation, same pool_scenes() calls,
imported unmodified) as eval_angular_separation.run_one_split,
eval_oracle_mondrian_angular_separation.oracle_mondrian_split, and
eval_deployable_mondrian_separation.deployable_split. For a given split_seed
the calibration/test pools (lm_c/est_c/true_c/D_c, lm_t/est_t/true_t/D_t) and
Global CP's lambda_global[k] are IDENTICAL across all M (M only affects how
the calibration pool is subdivided into categories) -- so pools and Global CP
are computed ONCE per split and reused for every M in {2,3,4,5}.

============================================================
5/6. REPORTING
============================================================
True D is used ONLY for the by-true-D breakdown tables below (never for
category construction). For every D in {5,10,15,20,30,45,60}, and
separately for pooled (k0+k1), k=0 only, and k=1 only (per the newly
confirmed IDL-slot asymmetry -- see Results/global_cp_by_idl_slot.txt),
this script reports coverage/area/area% for Global CP and for
Mondrian-hatD at M=2,3,4,5. Section 6 additionally reports, for each of
pooled/k0/k1 and each of Global/M=2/3/4/5: overall (marginal) coverage,
overall area, and the true-D coverage spread (max_D - min_D).

============================================================
9. NO MODEL SELECTION
============================================================
"Best M" is NOT chosen here based on test coverage or any other test-set
criterion. M=2,3,4,5 are reported side by side only. How M should be
selected in a methodologically clean way (e.g. via a separate validation
split, or a fixed a-priori rule) is left for a later, separate decision.

Not implemented here (out of scope, per spec): learned/UAI-style groups,
true-D-based category construction, lambda_star-based or test-data-based
boundary selection, LCP/Algorithm 7.8, cancellation features.
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))

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
from eval_deployable_mondrian_separation import delta_hat_deg_batch  # exact same great-circle hatD, reused not reimplemented

M_VALUES = [2, 3, 4, 5]
VARIANTS = ["pooled", "k0", "k1"]  # k=0/k=1 breakdown requested given the confirmed IDL-slot asymmetry


# ---------------------------------------------------------------------------
# Section 2: equal-frequency, tie-safe boundary construction
# ---------------------------------------------------------------------------

def compute_equal_freq_boundaries(hatD_calib, M):
    """Equal-frequency Mondrian-category boundaries over hatD, from
    CALIBRATION hatD values only. See module docstring Section 2 for the
    exact procedure. Returns a sorted, strictly increasing np.ndarray of
    length <= M-1 (< M-1 only if categories collapse due to insufficient
    distinct hatD values -- reported by the caller, never silently padded).
    """
    hatD_calib = np.asarray(hatD_calib, dtype=float)
    unique_vals = np.unique(hatD_calib)
    if len(unique_vals) < 2:
        return np.array([])  # every calibration frame has the same hatD -- cannot split at all

    boundaries = []
    for i in range(1, M):
        q = i / M
        raw_q = np.quantile(hatD_calib, q, method="linear")
        j = np.searchsorted(unique_vals, raw_q, side="left")
        j = int(np.clip(j, 1, len(unique_vals) - 1))
        boundaries.append((unique_vals[j - 1] + unique_vals[j]) / 2.0)

    return np.unique(np.array(boundaries, dtype=float))  # sorts + dedups collapsed boundaries


def assign_bins(hatD, boundaries):
    """Category index 0..len(boundaries) via the frozen boundaries. Boundaries
    were derived from calibration hatD only; this function is applied
    identically to calibration hatD (for calibrating lambda[m,k]) and test
    hatD (for evaluation)."""
    if len(boundaries) == 0:
        return np.zeros(len(hatD), dtype=int)
    return np.searchsorted(boundaries, hatD, side="right")


# ---------------------------------------------------------------------------
# Splits: pools + Global CP computed once per split, reused across all M
# ---------------------------------------------------------------------------

def build_split_pools(conds, separations, common_scenes, room, lambda_list, args, split_seed):
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
    lambdas_global = calibrate_global_lambda_from_arrays(lm_c, est_c, true_c, room, lambda_list, args.alpha)

    return dict(n_calib_scenes=len(calib_scenes), n_test_scenes=len(test_scenes),
                lm_c=lm_c, est_c=est_c, true_c=true_c, D_c=D_c,
                lm_t=lm_t, est_t=est_t, true_t=true_t, D_t=D_t,
                lambdas_global=lambdas_global)


def eval_global_records(pools, nele, nazi):
    """Global CP test records (D, k, covered, area) -- M-independent, computed
    once per split. Reproduces eval_angular_separation.run_one_split /
    diagnose_global_cp_by_idl_slot.run_one_split_by_slot exactly (same
    calibration call, same matching, same region-growing), with k retained
    per record for the pooled/k0/k1 breakdown."""
    lm_t, est_t, true_t, D_t = pools["lm_t"], pools["est_t"], pools["true_t"], pools["D_t"]
    lambdas_global = pools["lambdas_global"]
    records = []
    for i in range(lm_t.shape[0]):
        D_i = float(D_t[i])
        true_order, est_order = CoverageSet._match_estimated_to_source(true_t[i], est_t[i])
        for true_s, est_s in zip(true_order, est_order):
            k = int(est_s)
            norm_map = normalize(lm_t[i, k])
            seed = tuple(radians_to_grid_index(est_t[i, k], nele, nazi).astype(int))
            true_idx = tuple(radians_to_grid_index(true_t[i, true_s], nele, nazi).astype(int))
            region = CoverageSet.neighbours_coverage_set(norm_map, float(lambdas_global[k]), estimated_position=seed)
            records.append(dict(D=D_i, k=k, covered=bool(region[true_idx]), area=int(region.sum())))
    return records


def eval_mondrian_hatD_for_M(pools, room, nele, nazi, lambda_list, args, M):
    """Given the FIXED (M-independent) calib/test pools from build_split_pools,
    construct equal-frequency hatD boundaries from calibration hatD only,
    calibrate lambda[m,k] per realized category, and evaluate on the test
    pool. True D (D_t) is attached to test records for reporting only -- it
    plays no role in category construction or calibration."""
    lm_c, est_c, true_c = pools["lm_c"], pools["est_c"], pools["true_c"]
    lm_t, est_t, true_t, D_t = pools["lm_t"], pools["est_t"], pools["true_t"], pools["D_t"]

    hatD_c = delta_hat_deg_batch(est_c)
    hatD_t = delta_hat_deg_batch(est_t)

    boundaries = compute_equal_freq_boundaries(hatD_c, M)
    n_cat = len(boundaries) + 1

    Gc = assign_bins(hatD_c, boundaries)
    Gt = assign_bins(hatD_t, boundaries)

    lambdas_by_cat = {}
    n_calib_by_cat = {}
    infeasible = []
    for m in range(n_cat):
        mask = (Gc == m)
        n_calib_by_cat[m] = int(mask.sum())
        if n_calib_by_cat[m] == 0:
            infeasible.append((m, "no calibration frames in this hatD category for this split"))
            continue
        try:
            lambdas_by_cat[m] = calibrate_global_lambda_from_arrays(
                lm_c[mask], est_c[mask], true_c[mask], room, lambda_list, args.alpha)
        except ValueError as e:
            infeasible.append((m, str(e)))

    records = []
    n_test_by_cat = {m: 0 for m in range(n_cat)}
    for i in range(lm_t.shape[0]):
        D_i = float(D_t[i])
        m_i = int(Gt[i])
        n_test_by_cat[m_i] += 1
        if m_i not in lambdas_by_cat:
            continue  # category infeasible or empty at calibration time -- excluded, not backfilled from Global
        true_order, est_order = CoverageSet._match_estimated_to_source(true_t[i], est_t[i])
        for true_s, est_s in zip(true_order, est_order):
            k = int(est_s)
            norm_map = normalize(lm_t[i, k])
            seed = tuple(radians_to_grid_index(est_t[i, k], nele, nazi).astype(int))
            true_idx = tuple(radians_to_grid_index(true_t[i, true_s], nele, nazi).astype(int))
            region = CoverageSet.neighbours_coverage_set(
                norm_map, float(lambdas_by_cat[m_i][k]), estimated_position=seed)
            records.append(dict(D=D_i, k=k, m=m_i, covered=bool(region[true_idx]), area=int(region.sum())))

    return dict(boundaries=boundaries, n_categories_realized=n_cat,
                n_calib_by_cat=n_calib_by_cat, n_test_by_cat=n_test_by_cat,
                infeasible=infeasible, lambdas_by_cat=lambdas_by_cat, records=records)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def summarize(records, D=None, k=None):
    recs = records
    if D is not None:
        recs = [r for r in recs if r["D"] == D]
    if k is not None:
        recs = [r for r in recs if r["k"] == k]
    if not recs:
        return None
    return dict(coverage=float(np.mean([r["covered"] for r in recs])),
                area=float(np.mean([r["area"] for r in recs])), n=len(recs))


def agg(values):
    values = [v for v in values if v is not None]
    return (float(np.mean(values)), float(np.std(values)), len(values)) if values else (float("nan"), float("nan"), 0)


def variant_kwargs(variant):
    return dict(k=None) if variant == "pooled" else dict(k=int(variant[1:]))


def summarize_boundaries(boundary_arrays, M):
    """boundary_arrays: list of np.ndarray, one per split. Returns (stats or
    None, n_degenerate) where stats covers only the splits that realized the
    full M-1 boundaries (no collapse); degenerate (collapsed) splits are
    counted separately and excluded from the mean/std/range."""
    full = [b for b in boundary_arrays if len(b) == M - 1]
    n_degenerate = len(boundary_arrays) - len(full)
    if not full:
        return None, n_degenerate
    arr = np.stack(full, axis=0)  # (n_full_splits, M-1)
    return dict(mean=arr.mean(axis=0), std=arr.std(axis=0),
                min=arr.min(axis=0), max=arr.max(axis=0), n_full_splits=len(full)), n_degenerate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    p.add_argument("--seed", type=int, default=0, help="base seed; split i uses seed + i (must match Global/Oracle/manual-hatD runs)")
    p.add_argument("--out_dir", default="Results")
    p.add_argument("--oracle_json", default="Results/oracle_mondrian_angular_separation_raw.json",
                    help="Comparator B (true-D Oracle Mondrian), reused read-only, NOT recomputed")
    p.add_argument("--manual_hatD_json", default="Results/deployable_mondrian_separation_raw.json",
                    help="Comparator C (8-bin manually-binned hatD Mondrian), reused read-only, NOT recomputed")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    separations = args.separations
    assert len(separations) == len(args.data_paths), "--separations and --data_paths must match in length"

    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 100)
    out("Mondrian-hatD (equal-frequency, tie-safe bins), M = 2,3,4,5 -- clean deployable baseline")
    out("Category construction uses ONLY calibration-frame hatD (estimated separation). No true D,")
    out("no lambda_star, no test data enters bin-boundary construction. See module docstring.")
    out("=" * 100)

    conds, nele, nazi, common_scenes = load_conditions(args.data_paths, separations)
    room = _build_room(conds[separations[0]])
    n_grid_cells = nele * nazi
    lambda_list = np.linspace(0.0, 1.0, args.lambda_steps)
    out(f"\n{len(common_scenes)} paired base scenes, grid {nele}x{nazi}={n_grid_cells} cells, separations={separations}")
    out(f"n_splits={args.n_splits} seed={args.seed} calib_scene_frac={args.calib_scene_frac} "
        f"n_calib_frames_per_scene={args.n_calib_frames_per_scene} n_test_frames_per_scene={args.n_test_frames_per_scene} "
        f"alpha={args.alpha}")

    # ---- run all splits once; Mondrian-hatD computed for every M on the SAME pools ----
    global_records_by_split = []
    mondrian_by_M = {M: [] for M in M_VALUES}
    for i in range(args.n_splits):
        split_seed = args.seed + i
        pools = build_split_pools(conds, separations, common_scenes, room, lambda_list, args, split_seed)
        global_records_by_split.append(eval_global_records(pools, nele, nazi))
        row = [f"calib_scenes={pools['n_calib_scenes']}"]
        for M in M_VALUES:
            res = eval_mondrian_hatD_for_M(pools, room, nele, nazi, lambda_list, args, M)
            mondrian_by_M[M].append(res)
            row.append(f"M={M}:cats={res['n_categories_realized']}"
                       + ("" if res['n_categories_realized'] == M else "(COLLAPSED)")
                       + (f" INFEASIBLE={len(res['infeasible'])}" if res['infeasible'] else ""))
        out(f"split {i:3d}  " + "  ".join(row))

    # ---- Section 7 feasibility summary ----
    out("\n" + "=" * 100)
    out("SECTION 7a: category feasibility across all (split, M)")
    out("=" * 100)
    any_infeasible = False
    for M in M_VALUES:
        infeas = [(i, m, reason) for i, res in enumerate(mondrian_by_M[M]) for m, reason in res["infeasible"]]
        collapsed = [i for i, res in enumerate(mondrian_by_M[M]) if res["n_categories_realized"] < M]
        if infeas:
            any_infeasible = True
            out(f"M={M}: {len(infeas)} (split, category) INFEASIBLE -- excluded from aggregation, NOT backfilled from Global CP:")
            for i, m, reason in infeas:
                out(f"    split={i} category={m}: {reason}")
        else:
            out(f"M={M}: all realized categories FEASIBLE in all {args.n_splits} splits "
                f"(n_calib > 1/alpha - 1 = {1/args.alpha - 1:.1f} in every case).")
        if collapsed:
            out(f"M={M}: {len(collapsed)}/{args.n_splits} splits realized FEWER than {M} categories "
                f"(collapsed boundaries -- too few distinct hatD values for {M-1} equal-frequency cuts): "
                f"splits {collapsed}")
    if not any_infeasible:
        out("\n=> No infeasible categories at any M. All reported Mondrian-hatD numbers below use a real, "
            "conformally valid per-category calibration -- nothing was silently substituted from Global CP.")

    # ---- Section 7b: boundary diagnostics ----
    out("\n" + "=" * 100)
    out("SECTION 7b: learned hatD boundaries (degrees), mean/std/range across splits")
    out("=" * 100)
    for M in M_VALUES:
        boundary_arrays = [res["boundaries"] for res in mondrian_by_M[M]]
        stats, n_degenerate = summarize_boundaries(boundary_arrays, M)
        out(f"\nM={M} ({M-1} boundaries expected):")
        if stats is None:
            out(f"  ALL {args.n_splits} splits collapsed below {M} categories -- no full-M split to summarize.")
            continue
        if n_degenerate:
            out(f"  ({n_degenerate}/{args.n_splits} splits collapsed to <{M} categories, excluded from this summary)")
        for b in range(M - 1):
            out(f"  boundary[{b}]: mean={stats['mean'][b]:7.2f}  std={stats['std'][b]:6.2f}  "
                f"range=[{stats['min'][b]:7.2f}, {stats['max'][b]:7.2f}]  (n={stats['n_full_splits']})")

    # ---- Section 7c: n_calib / n_test / lambda[m,k] per category ----
    out("\n" + "=" * 100)
    out("SECTION 7c: per-category calibration/test support and lambda[m,k], mean +/- std across splits")
    out("=" * 100)
    K = mondrian_by_M[M_VALUES[0]][0]["lambdas_by_cat"][0].shape[0] if 0 in mondrian_by_M[M_VALUES[0]][0]["lambdas_by_cat"] else 2
    for M in M_VALUES:
        out(f"\nM={M}:")
        out(f"  {'cat':>4} {'n_calib(min/mean/max)':>24} {'n_test(min/mean/max)':>23} "
            + "  ".join(f"lambda[m,{k}]" for k in range(K)))
        for m in range(M):
            n_calib_list = [res["n_calib_by_cat"].get(m, 0) for res in mondrian_by_M[M]]
            n_test_list = [res["n_test_by_cat"].get(m, 0) for res in mondrian_by_M[M]]
            lam_strs = []
            for k in range(K):
                vals = [float(res["lambdas_by_cat"][m][k]) for res in mondrian_by_M[M] if m in res["lambdas_by_cat"]]
                lam_strs.append(f"{np.mean(vals):.4f}+/-{np.std(vals):.4f}(n={len(vals)})" if vals else "(none feasible)")
            out(f"  {m:>4} {min(n_calib_list):4d}/{np.mean(n_calib_list):6.1f}/{max(n_calib_list):4d}          "
                f"{min(n_test_list):4d}/{np.mean(n_test_list):6.1f}/{max(n_test_list):4d}         "
                + "  ".join(lam_strs))

    # ---- Section 5: primary results by true D, pooled / k0 / k1 ----
    out("\n" + "=" * 100)
    out("SECTION 5: PRIMARY RESULTS BY TRUE D -- Global CP vs Mondrian-hatD (M=2,3,4,5)")
    out("(True D used for reporting/grouping ONLY -- never for category construction)")
    out("=" * 100)

    by_D_tables = {}  # variant -> D -> dict with global + per-M cov/area/area%
    for variant in VARIANTS:
        vk = variant_kwargs(variant)
        out(f"\n--- variant: {variant} " + ("(k=0+k=1 combined)" if variant == "pooled" else "") + " ---")
        header = f"{'D':>5} {'cov_G':>7} {'area_G':>9} {'area%_G':>8}"
        for M in M_VALUES:
            header += f" | {'cov_M'+str(M):>8} {'area_M'+str(M):>9} {'area%_M'+str(M):>9}"
        out(header)
        by_D_tables[variant] = {}
        for D in separations:
            covs_g = [summarize(recs, D=D, **vk) for recs in global_records_by_split]
            cov_g, cov_g_std, _ = agg([s["coverage"] if s else None for s in covs_g])
            area_g, area_g_std, _ = agg([s["area"] if s else None for s in covs_g])
            row = dict(cov_g=cov_g, cov_g_std=cov_g_std, area_g=area_g, area_g_std=area_g_std)
            line = f"{D:5.1f} {cov_g:7.3f} {area_g:9.1f} {area_g/n_grid_cells*100:8.2f}"
            for M in M_VALUES:
                covs_m = [summarize(res["records"], D=D, **vk) for res in mondrian_by_M[M]]
                cov_m, cov_m_std, _ = agg([s["coverage"] if s else None for s in covs_m])
                area_m, area_m_std, _ = agg([s["area"] if s else None for s in covs_m])
                row[f"cov_M{M}"] = cov_m; row[f"cov_M{M}_std"] = cov_m_std
                row[f"area_M{M}"] = area_m; row[f"area_M{M}_std"] = area_m_std
                line += f" | {cov_m:8.3f} {area_m:9.1f} {area_m/n_grid_cells*100:9.2f}"
            out(line)
            by_D_tables[variant][D] = row

    # ---- Section 6: overall results ----
    out("\n" + "=" * 100)
    out("SECTION 6: OVERALL RESULTS -- overall coverage/area, true-D coverage spread (max-min)")
    out("=" * 100)
    overall_tables = {}
    for variant in VARIANTS:
        vk = variant_kwargs(variant)
        out(f"\n--- variant: {variant} ---")
        out(f"{'method':>10} {'overall_cov':>13} {'overall_area':>14} {'D_coverage_spread':>18}")
        overall_tables[variant] = {}

        g_overall_cov = [summarize(recs, **vk) for recs in global_records_by_split]
        cov_g_o, cov_g_o_std, _ = agg([s["coverage"] if s else None for s in g_overall_cov])
        area_g_o, area_g_o_std, _ = agg([s["area"] if s else None for s in g_overall_cov])
        cov_g_by_D = [by_D_tables[variant][D]["cov_g"] for D in separations]
        spread_g = max(cov_g_by_D) - min(cov_g_by_D)
        out(f"{'Global':>10} {cov_g_o:6.3f}+/-{cov_g_o_std:5.3f} {area_g_o:7.1f}+/-{area_g_o_std:6.1f}      {spread_g:14.3f}")
        overall_tables[variant]["Global"] = dict(cov=cov_g_o, cov_std=cov_g_o_std, area=area_g_o, area_std=area_g_o_std, spread=spread_g)

        for M in M_VALUES:
            m_overall = [summarize(res["records"], **vk) for res in mondrian_by_M[M]]
            cov_m_o, cov_m_o_std, _ = agg([s["coverage"] if s else None for s in m_overall])
            area_m_o, area_m_o_std, _ = agg([s["area"] if s else None for s in m_overall])
            cov_m_by_D = [by_D_tables[variant][D][f"cov_M{M}"] for D in separations]
            spread_m = max(cov_m_by_D) - min(cov_m_by_D)
            out(f"{'M='+str(M):>10} {cov_m_o:6.3f}+/-{cov_m_o_std:5.3f} {area_m_o:7.1f}+/-{area_m_o_std:6.1f}      {spread_m:14.3f}")
            overall_tables[variant][f"M{M}"] = dict(cov=cov_m_o, cov_std=cov_m_o_std, area=area_m_o, area_std=area_m_o_std, spread=spread_m)

    # ---- Section 8: comparators (pooled only -- Oracle-D / manual-hatD JSONs have no k-breakdown saved) ----
    out("\n" + "=" * 100)
    out("SECTION 8: COMPARATORS (pooled k0+k1 only -- Oracle-D and manual-hatD Mondrian were not run with a")
    out("k-breakdown, so only their existing pooled numbers are shown here; loaded read-only, NOT recomputed)")
    out("=" * 100)
    comparators_loaded = {}
    try:
        with open(args.oracle_json) as fh:
            oracle = json.load(fh)
        comparators_loaded["oracle"] = oracle
    except FileNotFoundError:
        oracle = None
        out(f"  (oracle_json not found at {args.oracle_json} -- Oracle-D comparator column skipped)")
    try:
        with open(args.manual_hatD_json) as fh:
            manual = json.load(fh)
        comparators_loaded["manual_hatD"] = manual
    except FileNotFoundError:
        manual = None
        out(f"  (manual_hatD_json not found at {args.manual_hatD_json} -- manual-hatD comparator column skipped)")

    header = f"{'D':>5} {'cov_Global':>11}"
    if oracle: header += f" {'cov_OracleD':>12}"
    if manual: header += f" {'cov_manualhatD':>15}"
    for M in M_VALUES:
        header += f" {'cov_M'+str(M):>9}"
    out(header)
    for D in separations:
        line = f"{D:5.1f} {by_D_tables['pooled'][D]['cov_g']:11.3f}"
        if oracle:
            line += f" {oracle['table_rows'][str(D)]['cov_m_mean']:12.3f}"
        if manual:
            line += f" {manual['b7_by_true_D'][str(D)]['cov_h']:15.3f}"
        for M in M_VALUES:
            line += f" {by_D_tables['pooled'][D][f'cov_M{M}']:9.3f}"
        out(line)

    out(f"\n{'method':>14} {'overall_coverage':>17} {'overall_area':>14}")
    out(f"{'Global':>14} {overall_tables['pooled']['Global']['cov']:8.4f}+/-{overall_tables['pooled']['Global']['cov_std']:6.4f} "
        f"{overall_tables['pooled']['Global']['area']:7.1f}+/-{overall_tables['pooled']['Global']['area_std']:6.1f}")
    if oracle:
        oo = oracle["overall_oracle_mondrian"]
        out(f"{'Oracle-D':>14} {oo['coverage_mean']:8.4f}+/-{oo['coverage_std']:6.4f} {oo['area_mean']:7.1f}+/-{oo['area_std']:6.1f}")
    if manual:
        mo = manual["overall_estimated_sep_mondrian"]
        out(f"{'manual-hatD':>14} {mo['coverage_mean']:8.4f}+/-{mo['coverage_std']:6.4f} {mo['area_mean']:7.1f}+/-{mo['area_std']:6.1f}")
    for M in M_VALUES:
        ot = overall_tables["pooled"][f"M{M}"]
        out(f"{'M='+str(M):>14} {ot['cov']:8.4f}+/-{ot['cov_std']:6.4f} {ot['area']:7.1f}+/-{ot['area_std']:6.1f}")

    out("\n" + "=" * 100)
    out("NOTE (spec section 9): no 'best M' is selected here based on these test-coverage numbers. "
        "M=2,3,4,5 are reported side by side for later, separately-decided model selection.")
    out("=" * 100)

    # ---- save ----
    out_txt = os.path.join(args.out_dir, "results_mondrian_hatD_equal_freq.txt")
    with open(out_txt, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nSaved -> {out_txt}")

    def jsonify_records_summary(table):
        return {str(D): v for D, v in table.items()}

    out_json = os.path.join(args.out_dir, "mondrian_hatD_equal_freq_raw.json")
    with open(out_json, "w") as fh:
        json.dump(dict(
            args=vars(args), n_grid_cells=n_grid_cells, M_values=M_VALUES,
            by_D={variant: jsonify_records_summary(by_D_tables[variant]) for variant in VARIANTS},
            overall={variant: overall_tables[variant] for variant in VARIANTS},
            boundaries_summary={
                str(M): (lambda s_nd: dict(
                    mean=s_nd[0]["mean"].tolist(), std=s_nd[0]["std"].tolist(),
                    min=s_nd[0]["min"].tolist(), max=s_nd[0]["max"].tolist(),
                    n_full_splits=s_nd[0]["n_full_splits"], n_degenerate=s_nd[1],
                ) if s_nd[0] is not None else dict(n_degenerate=s_nd[1], all_collapsed=True)
                )(summarize_boundaries([res["boundaries"] for res in mondrian_by_M[M]], M))
                for M in M_VALUES
            },
            n_infeasible_by_M={str(M): sum(len(res["infeasible"]) for res in mondrian_by_M[M]) for M in M_VALUES},
            n_collapsed_splits_by_M={str(M): sum(1 for res in mondrian_by_M[M] if res["n_categories_realized"] < M)
                                     for M in M_VALUES},
        ), fh, indent=2)
    print(f"Saved -> {out_json}")


if __name__ == "__main__":
    main()
