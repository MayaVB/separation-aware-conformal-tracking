"""
Diagnostic: does the CP-weighted tracker's fusion weight w respond to burst
corruption, without touching the tracker, CP regions, descriptors, or lambdas?

This is read-only w.r.t. CPWeightedFusionTracker, cp_features.py, and
cp_weight.py -- it runs the tracker exactly as implemented, with the fixed
defaults below, and uses the npz's existing `burst_active_per_frame` array
ONLY to label/stratify the collected records after the fact. Burst metadata
never enters CP-region construction or w computation.

Fixed parameters (per user instruction -- not tuned here):
    lambda_var  = 1.0
    lambda_size = 2.0
    lambda_span = 1.0
    sigma_el    = 2.0
    sigma_az    = 2.0

Active/inactive labeling reuses the exact same convention as
eval_tracker_conditions.py's existing LCP active/inactive evaluation:
`burst_active_per_frame` is a per-FRAME boolean (not per-speaker); both
speakers in an active frame are labeled "active". Only frames with
gt_valid_per_frame == True are included (same gating eval_tracker_conditions.py
uses before computing any per-frame metric), and frame 0 of every trajectory
is excluded (w is undefined there -- no prior belief yet).

Uses the full burst_test dataset (100 trajectories, 10400 frames) -- this is
cheap: CP-region growing is a flood-fill per frame, not a model forward pass.

Run from repo root:
    python examples/run_cp_weight_burst_diagnostic.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from Code.two_speaker_tracking.cp_weighted_tracker import CPWeightedFusionTracker
from Code.two_speaker_tracking.npz_adapter import (
    calibrate_lambda_thresholds, build_cp_regions_for_frames, positions_to_grid_indices,
)

DATA_ROOT = "/src/data"
CALIB_PATH = f"{DATA_ROOT}/npz_output_tracking_no_burst_calib/Reverb_400_ms_SNR_15_dB/speakers_2_flat.npz"
BURST_PATH = f"{DATA_ROOT}/npz_output_tracking_burst_test/Reverb_400_ms_SNR_15_dB/speakers_2_flat.npz"
FRAMES_PER_TRAJECTORY = 104
ALPHA = 0.1
LAMBDA_STEPS = 200      # same dev-speed tradeoff as run_two_speaker_cp_weighted_test.py
N_CALIB_FRAMES = 3000
SEED = 1234567890

# Fixed tracker parameters -- NOT tuned, per instruction.
SIGMA_EL, SIGMA_AZ = 2.0, 2.0
LAMBDA_VAR, LAMBDA_SIZE, LAMBDA_SPAN = 1.0, 2.0, 1.0


def _stats(arr):
    arr = np.asarray(arr, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "n": int(arr.size),
    }


def main():
    print(f"[1/3] Calibrating lambda thresholds from {CALIB_PATH} "
          f"(alpha={ALPHA}, lambda_steps={LAMBDA_STEPS}, n_calib_frames={N_CALIB_FRAMES})")
    lambdas_by_alpha = calibrate_lambda_thresholds(
        CALIB_PATH, [ALPHA], lambda_steps=LAMBDA_STEPS,
        n_calib_frames=N_CALIB_FRAMES, seed=SEED)
    lambdas = lambdas_by_alpha[ALPHA]
    print(f"    lambdas={lambdas}")

    print(f"[2/3] Loading burst dataset {BURST_PATH}")
    d = np.load(BURST_PATH, allow_pickle=True)
    nele, nazi = int(d["nele"]), int(d["nazi"])
    lm_flat = d["all_likelihood_maps"]
    est_flat = d["all_estimated_positions"]
    gt_valid_flat = d["gt_valid_per_frame"]
    active_flat = d["burst_active_per_frame"].astype(bool)
    N_flat, K = lm_flat.shape[:2]
    n_traj = N_flat // FRAMES_PER_TRAJECTORY
    assert n_traj * FRAMES_PER_TRAJECTORY == N_flat

    print(f"    n_trajectories={n_traj}  N_flat={N_flat}  K={K}")

    records = {
        "active": {k: [] for k in ("size_norm", "span_norm", "var_norm", "w")},
        "inactive": {k: [] for k in ("size_norm", "span_norm", "var_norm", "w")},
    }

    print(f"[3/3] Running CPWeightedFusionTracker over all {n_traj} trajectories "
          f"(sigma_el={SIGMA_EL}, sigma_az={SIGMA_AZ}, "
          f"lambda_var={LAMBDA_VAR}, lambda_size={LAMBDA_SIZE}, lambda_span={LAMBDA_SPAN})")
    for i in range(n_traj):
        s, e = i * FRAMES_PER_TRAJECTORY, (i + 1) * FRAMES_PER_TRAJECTORY
        lm_traj = lm_flat[s:e]
        est_rad_traj = est_flat[s:e]
        valid_traj = gt_valid_flat[s:e]
        active_traj = active_flat[s:e]  # per-frame, not per-speaker

        cp_regions = build_cp_regions_for_frames(lm_traj, est_rad_traj, lambdas, nele, nazi)
        est_grid_traj = positions_to_grid_indices(est_rad_traj, nele, nazi)

        tracker = CPWeightedFusionTracker(
            n_speakers=K, sigma_el=SIGMA_EL, sigma_az=SIGMA_AZ,
            lambda_var=LAMBDA_VAR, lambda_size=LAMBDA_SIZE, lambda_span=LAMBDA_SPAN,
        )
        result = tracker.run(lm_traj, cp_regions.astype(float), est_grid_traj)
        dbg = result["debug"]  # each of w/size_norm/span_norm/var_norm: (T, K)

        for t in range(1, FRAMES_PER_TRAJECTORY):  # exclude frame 0: w undefined (no prior belief)
            if not valid_traj[t]:
                continue
            label = "active" if active_traj[t] else "inactive"
            for k in range(K):
                records[label]["size_norm"].append(dbg["size_norm"][t, k])
                records[label]["span_norm"].append(dbg["span_norm"][t, k])
                records[label]["var_norm"].append(dbg["var_norm"][t, k])
                records[label]["w"].append(dbg["w"][t, k])
        if (i + 1) % 20 == 0 or i == n_traj - 1:
            print(f"    trajectory {i + 1}/{n_traj} done")

    n_active = len(records["active"]["w"])
    n_inactive = len(records["inactive"]["w"])
    n_total = n_active + n_inactive
    print(f"\nRecord counts (frame/speaker pairs, frame 0 and gt-invalid frames excluded):")
    print(f"  active   : {n_active:>6d}  ({100 * n_active / n_total:.2f}%)")
    print(f"  inactive : {n_inactive:>6d}  ({100 * n_inactive / n_total:.2f}%)")
    print(f"  total    : {n_total:>6d}")

    # ------------------------------------------------------------------
    # Derived contributions and total_penalty
    # ------------------------------------------------------------------
    for label in ("active", "inactive"):
        r = records[label]
        r["size_contrib"] = (LAMBDA_SIZE * np.asarray(r["size_norm"])).tolist()
        r["span_contrib"] = (LAMBDA_SPAN * np.asarray(r["span_norm"])).tolist()
        r["var_contrib"] = (LAMBDA_VAR * np.asarray(r["var_norm"])).tolist()
        r["total_penalty"] = (np.asarray(r["size_contrib"])
                               + np.asarray(r["span_contrib"])
                               + np.asarray(r["var_contrib"])).tolist()

    quantities = ["size_norm", "span_norm", "var_norm",
                  "size_contrib", "span_contrib", "var_contrib",
                  "total_penalty", "w"]

    print("\n" + "=" * 78)
    print("Per-quantity stats, active vs inactive")
    print("=" * 78)
    stats = {label: {} for label in ("active", "inactive")}
    for q in quantities:
        print(f"\n{q}:")
        for label in ("active", "inactive"):
            s = _stats(records[label][q])
            stats[label][q] = s
            print(f"  {label:<9s} mean={s['mean']:.4f}  median={s['median']:.4f}  "
                  f"std={s['std']:.4f}  p10={s['p10']:.4f}  p90={s['p90']:.4f}  n={s['n']}")

    print("\n" + "=" * 78)
    print("Compact comparison table")
    print("=" * 78)
    print(f"{'':<22s} {'inactive':>12s} {'active':>12s}")
    print(f"{'size_norm median':<22s} {stats['inactive']['size_norm']['median']:>12.4f} "
          f"{stats['active']['size_norm']['median']:>12.4f}")
    print(f"{'span_norm median':<22s} {stats['inactive']['span_norm']['median']:>12.4f} "
          f"{stats['active']['span_norm']['median']:>12.4f}")
    print(f"{'var_norm median':<22s} {stats['inactive']['var_norm']['median']:>12.4f} "
          f"{stats['active']['var_norm']['median']:>12.4f}")
    print(f"{'total_penalty median':<22s} {stats['inactive']['total_penalty']['median']:>12.4f} "
          f"{stats['active']['total_penalty']['median']:>12.4f}")
    print(f"{'w mean':<22s} {stats['inactive']['w']['mean']:>12.4f} "
          f"{stats['active']['w']['mean']:>12.4f}")
    print(f"{'w median':<22s} {stats['inactive']['w']['median']:>12.4f} "
          f"{stats['active']['w']['median']:>12.4f}")

    mean_diff = stats["active"]["w"]["mean"] - stats["inactive"]["w"]["mean"]
    median_diff = stats["active"]["w"]["median"] - stats["inactive"]["w"]["median"]
    print(f"\nmean(w_active) - mean(w_inactive)     = {mean_diff:+.4f}")
    print(f"median(w_active) - median(w_inactive) = {median_diff:+.4f}")

    # ------------------------------------------------------------------
    # Which descriptor drives the difference (compare contribution deltas)?
    # ------------------------------------------------------------------
    print("\nContribution deltas (active median - inactive median):")
    for contrib in ("size_contrib", "span_contrib", "var_contrib"):
        delta = stats["active"][contrib]["median"] - stats["inactive"][contrib]["median"]
        print(f"  {contrib:<14s} {delta:+.4f}")

    # ------------------------------------------------------------------
    # Figure: active vs inactive distribution of w
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        axes[0].hist(records["inactive"]["w"], bins=50, alpha=0.6, density=True, label="inactive")
        axes[0].hist(records["active"]["w"], bins=50, alpha=0.6, density=True, label="active")
        axes[0].set_xlabel("w")
        axes[0].set_ylabel("density")
        axes[0].set_title("w distribution: burst-active vs inactive")
        axes[0].legend()

        axes[1].boxplot([records["inactive"]["w"], records["active"]["w"]],
                         labels=["inactive", "active"], showmeans=True)
        axes[1].set_ylabel("w")
        axes[1].set_title("w boxplot: burst-active vs inactive")

        plt.tight_layout()
        fig_path = os.path.join(os.path.dirname(__file__), "debug_cp_weight_burst_diagnostic.png")
        plt.savefig(fig_path, dpi=100)
        print(f"\nFigure saved: {fig_path}")
        plt.close()
    except ImportError:
        print("\n(matplotlib not available -- skipping figure)")

    print("\nDone.")


if __name__ == "__main__":
    main()
