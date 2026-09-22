"""
Stage 2 of the angular-separation heterogeneity experiment (see
/home/mayavb/.claude/plans/piped-swinging-tarjan.md). Tests whether the true angular
separation D between the two simultaneously-active speakers changes the DOA
estimation-error / nonconformity-score distribution -- i.e. P(V | D) != P(V) -- using
data produced by SRP-DNN-CP's --doa-pair-mode (see its RunSRPDNN.py/Dataset.py/
OptSRPDNN.py, and CLAUDE.md there for the flags). Does NOT run LCP/Algorithm 7.8
(never imports localized_cp_decision/widest_path_score_map) -- this is strictly
Global-CP + raw-score analysis, to decide WHETHER separation-conditioned localization
is even worth building.

Expects one npz file per separation condition D (same schema as the existing
speakers_2_flat.npz files, PLUS --doa-pair-mode's extra fields:
base_scene_id_per_frame, doa_center_deg_per_frame, doa_separation_deg_per_frame),
generated with the SAME --seed at every D so the same base_scene_id refers to the
same underlying room/utterances/noise/SNR/center-angle across all conditions --
only the two derived speaker azimuths differ (see Dataset.py's RandomMicSigDataset
doa_pair_mode docstring). base_scene_id is the pairing/cluster key for every
statistic below: splits and bootstrap resampling operate on base_scene_id, never on
frame or on speaker.

Nonconformity score reused as-is from the project's own LCP work (not reinvented):
V = -lambda_star, via Code.two_speaker_tracking.lcp.compute_calibration_scores
(read-only reuse, exactly as eval_lcp_repeated_splits.py already does for LCP's own
calibration scores). DOA error E reused from Code.two_speaker_tracking.eval_metrics.

Because compute_calibration_scores does a per-frame lambda-grid region-growing
search (the same cost as one CoverageSet calibration point), scoring every frame of
every scene is prohibitively expensive for J~100 scenes x 7 conditions x ~100 frames
x K=2 speakers -- --n_frames_per_scene subsamples a fixed number of frames per
(scene, D) for V, following the same subsampling pattern eval_lcp_repeated_splits.py
already uses for calibration/test frame pools.
"""

import argparse
import json
import os

import numpy as np

from Code.crc_ssl import CoverageSet
from Code.utilities import normalize
from Code.two_speaker_tracking.npz_adapter import radians_to_grid_index, _build_room
from Code.two_speaker_tracking.eval_metrics import wrap_azi_err_deg
from Code.two_speaker_tracking.lcp import (
    compute_calibration_scores, calibrate_global_lambda_from_arrays, run_unit_checks,
)

DEFAULT_SEPARATIONS = [5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_paths", nargs="+", required=True,
                    help="One speakers_2_flat.npz path per --separations entry, same order")
    p.add_argument("--separations", nargs="+", type=float, default=DEFAULT_SEPARATIONS,
                    help="D values (deg), matching --data_paths order (default: 5 10 15 20 30 45 60)")
    p.add_argument("--n_frames_per_scene", type=int, default=10,
                    help="frames subsampled per (scene, D) for V/E (default: 10)")
    p.add_argument("--n_splits", type=int, default=20)
    p.add_argument("--calib_scene_frac", type=float, default=0.5)
    p.add_argument("--n_calib_frames_per_scene", type=int, default=5,
                    help="frames subsampled per calib scene for the Global-CP lambda fit (default: 5)")
    p.add_argument("--n_test_frames_per_scene", type=int, default=5,
                    help="frames subsampled per test scene for Global-CP coverage/area (default: 5)")
    p.add_argument("--lambda_steps", type=int, default=500)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--n_bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0, help="base seed; split i uses seed + i")
    p.add_argument("--out_dir", default="Results")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loading / pairing
# ---------------------------------------------------------------------------

def load_conditions(data_paths, separations):
    conds = {}
    for D, path in zip(separations, data_paths):
        d = np.load(path, allow_pickle=True)
        for key in ("base_scene_id_per_frame", "doa_separation_deg_per_frame"):
            assert key in d, (f"{path} is missing '{key}' -- was it generated with "
                               f"--doa-pair-mode? (see SRP-DNN-CP's RunSRPDNN.py)")
        conds[D] = d
    nele = int(conds[separations[0]]["nele"])
    nazi = int(conds[separations[0]]["nazi"])
    for D in separations:
        assert int(conds[D]["nele"]) == nele and int(conds[D]["nazi"]) == nazi, \
            "grid resolution differs across condition files"
    common_scenes = None
    for D in separations:
        ids = set(np.unique(conds[D]["base_scene_id_per_frame"]).tolist())
        common_scenes = ids if common_scenes is None else (common_scenes & ids)
    common_scenes = np.array(sorted(common_scenes), dtype=np.int64)
    assert len(common_scenes) > 0, "no base_scene_id is present in all conditions -- check pairing"
    return conds, nele, nazi, common_scenes


def matched_errors(est_sub, true_sub):
    """(n,K,2) est/true radians -> (n,K) wrapped azimuth error in degrees, indexed by
    ESTIMATED speaker slot (same matching convention compute_calibration_scores uses)."""
    n, K = est_sub.shape[:2]
    E = np.full((n, K), np.nan)
    for i in range(n):
        true_order, est_order = CoverageSet._match_estimated_to_source(true_sub[i], est_sub[i])
        for true_s, est_s in zip(true_order, est_order):
            E[i, est_s] = wrap_azi_err_deg(est_sub[i, est_s, 1], true_sub[i, true_s, 1])
    return E


def scene_frame_indices(d, scene_id):
    sidx = d["base_scene_id_per_frame"]
    gtv = d["gt_valid_per_frame"].astype(bool)
    return np.where((sidx == scene_id) & gtv)[0]


# ---------------------------------------------------------------------------
# Step 2.1/2.2: per-(D, scene) frame-level E and V
# ---------------------------------------------------------------------------

def build_frame_records(conds, separations, common_scenes, nele, nazi, lambda_list,
                         n_frames_per_scene, rng):
    """Returns:
      frame_records: list of dicts {D, scene_id, E_deg, V} -- one per sampled (frame,speaker)
      scene_summary: dict D -> scene_id -> {'median_V': float, 'median_E': float, 'n': int}
    """
    frame_records = []
    scene_summary = {D: {} for D in separations}
    for D in separations:
        d = conds[D]
        lm_all, est_all, true_all = d["all_likelihood_maps"], d["all_estimated_positions"], d["speaker_pos"]
        for j in common_scenes:
            idxs = scene_frame_indices(d, j)
            if len(idxs) == 0:
                continue
            sub = rng.choice(idxs, size=min(n_frames_per_scene, len(idxs)), replace=False)
            lm_sub, est_sub, true_sub = lm_all[sub], est_all[sub], true_all[sub]
            S_sub = compute_calibration_scores(lm_sub, est_sub, true_sub, lambda_list, nele, nazi)  # (n,K) = V
            E_sub = matched_errors(est_sub, true_sub)  # (n,K) deg
            valid = np.isfinite(S_sub) & np.isfinite(E_sub)
            v_vals = S_sub[valid]
            e_vals = E_sub[valid]
            for v, e in zip(v_vals, e_vals):
                frame_records.append(dict(D=D, scene_id=int(j), E_deg=float(e), V=float(v)))
            if len(v_vals) > 0:
                scene_summary[D][int(j)] = dict(
                    median_V=float(np.median(v_vals)), median_E=float(np.median(e_vals)),
                    n=int(len(v_vals)))
    return frame_records, scene_summary


# ---------------------------------------------------------------------------
# Step 2.3: Global-CP conditional coverage, repeated scene-level splits
# ---------------------------------------------------------------------------

def pool_scenes(conds, separations, scene_ids, n_per_scene, rng):
    """Concatenate n_per_scene subsampled frames per scene, per D, across all
    separations in `separations` -- returns (lm, est, true, D_of_frame) pooled arrays."""
    lm_list, est_list, true_list, D_list = [], [], [], []
    for D in separations:
        d = conds[D]
        for j in scene_ids:
            idxs = scene_frame_indices(d, j)
            if len(idxs) == 0:
                continue
            sub = rng.choice(idxs, size=min(n_per_scene, len(idxs)), replace=False)
            lm_list.append(d["all_likelihood_maps"][sub])
            est_list.append(d["all_estimated_positions"][sub])
            true_list.append(d["speaker_pos"][sub])
            D_list.extend([D] * len(sub))
    return (np.concatenate(lm_list, axis=0), np.concatenate(est_list, axis=0),
            np.concatenate(true_list, axis=0), np.array(D_list, dtype=float))


def run_one_split(conds, separations, common_scenes, room, nele, nazi, lambda_list, args, split_seed):
    """Splits base_scene_id ONCE (never per-D) into calib/test pools; pool_scenes()
    below then pulls ALL `separations` for every scene in whichever pool it's given,
    so all 7 D-versions of a base scene always land together in calib or in test,
    never split across them -- preserves the paired design end to end. No separate
    D-stratification is needed: since every base scene already contributes to every
    D by construction, a single scene-level split is automatically balanced across D."""
    rng = np.random.default_rng(split_seed)
    scene_order = rng.permutation(common_scenes)
    n_calib = int(round(len(scene_order) * args.calib_scene_frac))
    calib_scenes, test_scenes = scene_order[:n_calib], scene_order[n_calib:]
    assert len(set(calib_scenes.tolist()) & set(test_scenes.tolist())) == 0, \
        "a base_scene_id leaked into both calib and test -- pairing invariant violated"

    lm_c, est_c, true_c, _ = pool_scenes(conds, separations, calib_scenes,
                                          args.n_calib_frames_per_scene, rng)
    lambdas_global = calibrate_global_lambda_from_arrays(lm_c, est_c, true_c, room, lambda_list, args.alpha)

    lm_t, est_t, true_t, D_t = pool_scenes(conds, separations, test_scenes,
                                            args.n_test_frames_per_scene, rng)
    K = lm_t.shape[1]
    records = []
    for i in range(lm_t.shape[0]):
        true_order, est_order = CoverageSet._match_estimated_to_source(true_t[i], est_t[i])
        for true_s, est_s in zip(true_order, est_order):
            k = int(est_s)
            norm_map = normalize(lm_t[i, k])
            seed = tuple(radians_to_grid_index(est_t[i, k], nele, nazi).astype(int))
            true_idx = tuple(radians_to_grid_index(true_t[i, true_s], nele, nazi).astype(int))
            region = CoverageSet.neighbours_coverage_set(norm_map, float(lambdas_global[k]), estimated_position=seed)
            records.append(dict(D=float(D_t[i]), covered=bool(region[true_idx]), area=int(region.sum())))

    overall_cov = float(np.mean([r["covered"] for r in records])) if records else float("nan")
    by_D = {}
    for D in separations:
        recs = [r for r in records if r["D"] == D]
        if recs:
            by_D[D] = dict(coverage=float(np.mean([r["covered"] for r in recs])),
                            area=float(np.mean([r["area"] for r in recs])), n=len(recs))
    return dict(overall_coverage=overall_cov, by_D=by_D)


# ---------------------------------------------------------------------------
# Cluster (base-scene) bootstrap, jointly across all D per the paired design
# ---------------------------------------------------------------------------

def cluster_bootstrap_ci(scene_summary, separations, common_scenes, metric_key, quantile, n_bootstrap, rng):
    """Cluster bootstrap over base_scene_id, jointly across all D (per the paired
    design): each iteration draws ONE resampled set of scene ids and reuses it for
    every D's CI in that iteration (rather than resampling each D condition's scene
    pool independently), preserving the pairing the same way the calib/test split
    does. Operates on the per-scene `metric_key` summary (e.g. median_V/median_E from
    build_frame_records) -- the requested `quantile` (or the mean, if None) is taken
    across scenes' summary values for each D, on each bootstrap draw."""
    boot = {D: [] for D in separations}
    scenes = np.array(common_scenes)
    for _ in range(n_bootstrap):
        resampled = rng.choice(scenes, size=len(scenes), replace=True)  # shared across every D below
        for D in separations:
            vals = [scene_summary[D][j][metric_key] for j in resampled if j in scene_summary[D]]
            if vals:
                boot[D].append(np.quantile(vals, quantile) if quantile is not None else np.mean(vals))
    ci = {}
    for D in separations:
        if boot[D]:
            ci[D] = (float(np.percentile(boot[D], 2.5)), float(np.percentile(boot[D], 97.5)))
        else:
            ci[D] = (float("nan"), float("nan"))
    return ci


# ---------------------------------------------------------------------------
# Step 2.2 paired analysis: within-scene V(D_small) vs V(D_large)
# ---------------------------------------------------------------------------

def paired_compare(scene_summary, D_small, D_large, common_scenes):
    pairs = [(scene_summary[D_small][j]["median_V"], scene_summary[D_large][j]["median_V"])
             for j in common_scenes if j in scene_summary[D_small] and j in scene_summary[D_large]]
    if not pairs:
        return None
    diffs = np.array([a - b for a, b in pairs])
    n_small_worse = int(np.sum(diffs > 0))  # V_small > V_large: small separation is harder
    return dict(n=len(diffs), mean_diff=float(diffs.mean()), std_diff=float(diffs.std()),
                frac_small_gt_large=n_small_worse / len(diffs))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    separations = args.separations
    assert len(separations) == len(args.data_paths), "--separations and --data_paths must match in length"

    print("=" * 70)
    print("Step 0: unit checks (LCP module reused read-only for V = -lambda_star)")
    print("=" * 70)
    run_unit_checks(verbose=True)

    print()
    print("=" * 70)
    print(f"Step 1: load {len(separations)} condition files, find paired base_scene_id set")
    print("=" * 70)
    conds, nele, nazi, common_scenes = load_conditions(args.data_paths, separations)
    room = _build_room(conds[separations[0]])
    print(f"  {len(common_scenes)} base scenes present in every condition (of "
          f"{[len(np.unique(conds[D]['base_scene_id_per_frame'])) for D in separations]} per-file)")

    print()
    print("=" * 70)
    print("Step 2.1/2.2: per-(D, scene) DOA error E and nonconformity score V")
    print("=" * 70)
    lambda_list = np.linspace(0.0, 1.0, args.lambda_steps)
    rng = np.random.default_rng(args.seed)
    frame_records, scene_summary = build_frame_records(
        conds, separations, common_scenes, nele, nazi, lambda_list, args.n_frames_per_scene, rng)

    e_stats, v_stats = {}, {}
    for D in separations:
        recs = [r for r in frame_records if r["D"] == D]
        if not recs:
            continue
        E = np.array([r["E_deg"] for r in recs])
        V = np.array([r["V"] for r in recs])
        e_stats[D] = dict(mean=float(E.mean()), median=float(np.median(E)), std=float(E.std()),
                           p90=float(np.percentile(E, 90)), p95=float(np.percentile(E, 95)), n=len(E))
        v_stats[D] = dict(median=float(np.median(V)), q90=float(np.percentile(V, 90)),
                           q95=float(np.percentile(V, 95)), n=len(V))
        print(f"  D={D:5.1f}deg  E: mean={e_stats[D]['mean']:6.2f} median={e_stats[D]['median']:6.2f} "
              f"p90={e_stats[D]['p90']:6.2f} p95={e_stats[D]['p95']:6.2f}  |  "
              f"V: median={v_stats[D]['median']:7.4f} Q90={v_stats[D]['q90']:7.4f} Q95={v_stats[D]['q95']:7.4f}  "
              f"(n={e_stats[D]['n']})")

    print()
    print("Cluster (base-scene) bootstrap 95% CIs, jointly resampled across D:")
    boot_rng = np.random.default_rng(args.seed + 10_000)
    v_median_ci = cluster_bootstrap_ci(scene_summary, separations, common_scenes, "median_V", 0.5,
                                        args.n_bootstrap, boot_rng)
    v_q90_ci = cluster_bootstrap_ci(scene_summary, separations, common_scenes, "median_V", 0.9,
                                     args.n_bootstrap, boot_rng)
    for D in separations:
        print(f"  D={D:5.1f}deg  median(V) 95% CI={v_median_ci[D]}  Q90(scene-median V) 95% CI={v_q90_ci[D]}")

    print()
    print("Paired within-scene comparison, smallest vs largest D "
          f"({separations[0]} vs {separations[-1]} deg):")
    pc = paired_compare(scene_summary, separations[0], separations[-1], common_scenes)
    if pc:
        print(f"  n={pc['n']}  mean(V_small - V_large)={pc['mean_diff']:+.4f} (std={pc['std_diff']:.4f})  "
              f"frac(V_small > V_large)={pc['frac_small_gt_large']:.3f}")

    print()
    print("=" * 70)
    print(f"Step 2.3: {args.n_splits} repeated scene-level splits, Global-CP conditional coverage")
    print("=" * 70)
    split_results = []
    for i in range(args.n_splits):
        res = run_one_split(conds, separations, common_scenes, room, nele, nazi, lambda_list,
                             args, split_seed=args.seed + i)
        split_results.append(res)
        row = "  ".join(f"D={D:.0f}:cov={res['by_D'].get(D, {}).get('coverage', float('nan')):.3f}"
                         for D in separations)
        print(f"  split {i:3d} overall_cov={res['overall_coverage']:.3f}  {row}")

    overall_cov_mean = float(np.mean([r["overall_coverage"] for r in split_results]))
    overall_cov_std = float(np.std([r["overall_coverage"] for r in split_results]))
    cov_by_D, area_by_D = {}, {}
    for D in separations:
        covs = [r["by_D"][D]["coverage"] for r in split_results if D in r["by_D"]]
        areas = [r["by_D"][D]["area"] for r in split_results if D in r["by_D"]]
        cov_by_D[D] = (float(np.mean(covs)), float(np.std(covs)), len(covs)) if covs else None
        area_by_D[D] = (float(np.mean(areas)), float(np.std(areas)), len(areas)) if areas else None

    print()
    print(f"Overall (marginal) Global-CP coverage: {overall_cov_mean:.3f} +/- {overall_cov_std:.3f} "
          f"(target 1-alpha={1 - args.alpha:.2f})")
    print("Conditional coverage / area by D:")
    for D in separations:
        c, a = cov_by_D[D], area_by_D[D]
        c_str = f"{c[0]:.3f}+/-{c[1]:.3f}" if c else "(no data)"
        a_str = f"{a[0]:.1f}+/-{a[1]:.1f}" if a else "(no data)"
        print(f"  D={D:5.1f}deg  coverage={c_str}  area={a_str}")

    # ---- headline plots ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        Ds = separations
        axes[0].plot(Ds, [v_stats[D]["median"] for D in Ds], "o-")
        axes[0].set_xlabel("Separation D (deg)"); axes[0].set_ylabel("median(V)")
        axes[0].set_title("Nonconformity score median vs D")

        axes[1].plot(Ds, [v_stats[D]["q90"] for D in Ds], "o-", color="tab:orange")
        axes[1].set_xlabel("Separation D (deg)"); axes[1].set_ylabel("Q_0.9(V)")
        axes[1].set_title("Nonconformity score 90th pct vs D")

        cov_means = [cov_by_D[D][0] if cov_by_D[D] else np.nan for D in Ds]
        cov_stds = [cov_by_D[D][1] if cov_by_D[D] else 0 for D in Ds]
        axes[2].errorbar(Ds, cov_means, yerr=cov_stds, fmt="o-", color="tab:green")
        axes[2].axhline(1 - args.alpha, ls="--", color="gray", label=f"target {1-args.alpha:.2f}")
        axes[2].axhline(overall_cov_mean, ls=":", color="black", label=f"overall {overall_cov_mean:.3f}")
        axes[2].set_xlabel("Separation D (deg)"); axes[2].set_ylabel("Global-CP coverage")
        axes[2].set_title("Conditional coverage vs D"); axes[2].legend(fontsize=8)

        fig.tight_layout()
        fig_path = os.path.join(args.out_dir, "angular_separation_headline.png")
        fig.savefig(fig_path, dpi=150)
        print(f"\nSaved headline figure -> {fig_path}")

        fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.5))
        axes2[0].errorbar(Ds, [e_stats[D]["mean"] for D in Ds],
                           yerr=[e_stats[D]["std"] for D in Ds], fmt="o-")
        axes2[0].plot(Ds, [e_stats[D]["p90"] for D in Ds], "s--", label="P90")
        axes2[0].set_xlabel("Separation D (deg)"); axes2[0].set_ylabel("DOA error E (deg)")
        axes2[0].set_title("DOA error vs D"); axes2[0].legend(fontsize=8)

        area_means = [area_by_D[D][0] if area_by_D[D] else np.nan for D in Ds]
        axes2[1].plot(Ds, area_means, "o-", color="tab:red")
        axes2[1].set_xlabel("Separation D (deg)"); axes2[1].set_ylabel("Global-CP region area")
        axes2[1].set_title("Region area vs D")

        fig2.tight_layout()
        fig2_path = os.path.join(args.out_dir, "angular_separation_supporting.png")
        fig2.savefig(fig2_path, dpi=150)
        print(f"Saved supporting figure -> {fig2_path}")
    except ImportError:
        print("\nmatplotlib not available -- skipped figures, numeric results still saved below")

    # ---- save numeric results ----
    out_txt = os.path.join(args.out_dir, "results_angular_separation.txt")
    with open(out_txt, "w") as fh:
        fh.write(f"{len(common_scenes)} paired base scenes, separations={separations}\n\n")
        fh.write("D_deg  E_mean  E_median  E_std  E_p90  E_p95  V_median  V_Q90  V_Q95  "
                 "cov_mean  cov_std  area_mean  area_std\n")
        for D in separations:
            c, a = cov_by_D[D], area_by_D[D]
            fh.write(f"{D:5.1f}  {e_stats[D]['mean']:6.2f}  {e_stats[D]['median']:6.2f}  "
                     f"{e_stats[D]['std']:6.2f}  {e_stats[D]['p90']:6.2f}  {e_stats[D]['p95']:6.2f}  "
                     f"{v_stats[D]['median']:8.4f}  {v_stats[D]['q90']:8.4f}  {v_stats[D]['q95']:8.4f}  "
                     f"{(c[0] if c else float('nan')):8.3f}  {(c[1] if c else float('nan')):7.3f}  "
                     f"{(a[0] if a else float('nan')):9.1f}  {(a[1] if a else float('nan')):8.1f}\n")
        fh.write(f"\noverall_coverage_mean={overall_cov_mean:.4f} std={overall_cov_std:.4f} "
                 f"(target={1-args.alpha:.2f})\n")
        if pc:
            fh.write(f"\npaired D={separations[0]} vs D={separations[-1]}: n={pc['n']} "
                     f"mean_diff={pc['mean_diff']:+.4f} std_diff={pc['std_diff']:.4f} "
                     f"frac(V_small>V_large)={pc['frac_small_gt_large']:.3f}\n")
    print(f"\nSaved -> {out_txt}")

    out_json = os.path.join(args.out_dir, "angular_separation_raw.json")
    with open(out_json, "w") as fh:
        json.dump(dict(args=vars(args), e_stats=e_stats, v_stats=v_stats,
                        v_median_ci=v_median_ci, v_q90_ci=v_q90_ci, paired_compare=pc,
                        overall_coverage_mean=overall_cov_mean, overall_coverage_std=overall_cov_std,
                        cov_by_D={str(k): v for k, v in cov_by_D.items()},
                        area_by_D={str(k): v for k, v in area_by_D.items()}),
                  fh, indent=2)
    print(f"Saved -> {out_json}")

    # ---- decision gate ----
    print()
    print("=" * 70)
    print("Decision gate")
    print("=" * 70)
    v_q90_vals = [v_stats[D]["q90"] for D in separations]
    v_q90_spread = max(v_q90_vals) - min(v_q90_vals)
    cov_vals = [cov_by_D[D][0] for D in separations if cov_by_D[D]]
    cov_spread = (max(cov_vals) - min(cov_vals)) if cov_vals else float("nan")
    print(f"  Q_0.9(V) spread across D: {v_q90_spread:.4f}")
    print(f"  Conditional coverage spread across D: {cov_spread:.4f} "
          f"(overall coverage: {overall_cov_mean:.3f})")
    print("  Judge this against the CIs/std above -- if both spreads are small relative to "
          "their bootstrap CIs / split std, P(V|D) ~= P(V) and the recommendation is to STOP "
          "here rather than proceed to feature engineering/LCP on angular separation.")


if __name__ == "__main__":
    main()
