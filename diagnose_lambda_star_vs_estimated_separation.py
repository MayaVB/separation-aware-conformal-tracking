"""
ONE diagnostic: visualize lambda_star vs the inference-observable estimated
speaker separation Delta_hat, on calibration data from a single fixed
scene-disjoint split (split/seed 0), to see whether lambda_star has
distinct regimes as a function of Delta_hat BEFORE deciding how any future
Mondrian grouping should be formed.

Does NOT change the CP/Mondrian/LCP algorithm, does NOT learn or propose
new groups, does NOT optimize bin boundaries. Read-only analysis of
quantities already produced by the existing, validated calibration code.

Delta_hat (frame-level, ESTIMATED positions only, no GT):
    cos(Delta_hat) = cos(th0)*cos(th1) + sin(th0)*sin(th1)*cos(phi0-phi1)
    Delta_hat = arccos(clip(cos(Delta_hat), -1, 1))
from all_estimated_positions[t,0,:], [t,1,:], same definition and
implementation as eval_deployable_mondrian_separation.delta_hat_deg_batch
(imported, not reimplemented, to guarantee it's the exact same quantity).

lambda_star (per calibration frame, per estimated-speaker slot k): the
existing, already-validated Code.two_speaker_tracking.lcp.compute_lambda_star
via compute_calibration_scores (S = -lambda_star), unmodified. Indexed by
estimated-speaker slot (est_s from CoverageSet._match_estimated_to_source,
which preserves raw estimated-array order), so lambda_star[:, k] lines up
with the same k used for Delta_hat's est[:, k, :].

Calibration data selection: the SAME split_seed=0 scene permutation and
calib_scene_frac used throughout (eval_angular_separation.run_one_split /
eval_oracle_mondrian_angular_separation.oracle_mondrian_split /
eval_deployable_mondrian_separation.deployable_split), and the SAME
pool_scenes(..., n_calib_frames_per_scene=10) call (imported unmodified)
-- so the calibration scenes/frames here are bit-identical to split 0 of
every prior experiment. Only the calibration pool is used; no test data
is sampled or touched.

The 8 Delta_hat bins from Diagnostic 2 / eval_deployable_mondrian_separation
are used ONLY for the reporting table in section 7 -- explicitly not a
proposed grouping.
"""

import argparse
import os

import numpy as np

from Code.two_speaker_tracking.lcp import compute_calibration_scores

from eval_angular_separation import load_conditions, pool_scenes, DEFAULT_SEPARATIONS
from eval_oracle_mondrian_angular_separation import DEFAULT_DATA_PATHS
from eval_deployable_mondrian_separation import delta_hat_deg_batch, BIN_EDGES_DEG, BIN_LABELS, N_BINS, assign_bins

D_COLORS = {5.0: "#1f77b4", 10.0: "#ff7f0e", 15.0: "#2ca02c", 20.0: "#d62728",
            30.0: "#9467bd", 45.0: "#8c564b", 60.0: "#e377c2"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_paths", nargs="+", default=DEFAULT_DATA_PATHS)
    p.add_argument("--separations", nargs="+", type=float, default=DEFAULT_SEPARATIONS)
    p.add_argument("--calib_scene_frac", type=float, default=0.5)
    p.add_argument("--n_calib_frames_per_scene", type=int, default=10)
    p.add_argument("--lambda_steps", type=int, default=500)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--window_deg", type=float, default=5.0)
    p.add_argument("--min_n_local", type=int, default=20)
    p.add_argument("--out_dir", default="Results")
    return p.parse_args()


def local_quantiles(delta_vals, lambda_vals, window, min_n):
    """Sliding-window local median/Q25/Q10 vs Delta_hat, evaluated at every
    distinct observed Delta_hat value (rounded to 2 decimals to merge
    float noise from arccos, not to smooth over real quantization steps).
    Only kept where n_local >= min_n. No interpolation/fitting."""
    d_round = np.round(delta_vals, 2)
    uniq = np.unique(d_round)
    out = []
    for d in uniq:
        mask = np.abs(delta_vals - d) <= window
        n = int(mask.sum())
        if n >= min_n:
            vals = lambda_vals[mask]
            out.append(dict(d=float(d), n=n,
                             median=float(np.median(vals)),
                             q25=float(np.percentile(vals, 25)),
                             q10=float(np.percentile(vals, 10))))
    return out


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    separations = args.separations

    conds, nele, nazi, common_scenes = load_conditions(args.data_paths, separations)
    lambda_list = np.linspace(0.0, 1.0, args.lambda_steps)

    # ---- exact same calib scene split as split_seed=0 in every prior experiment ----
    rng = np.random.default_rng(args.split_seed)
    scene_order = rng.permutation(common_scenes)
    n_calib = int(round(len(scene_order) * args.calib_scene_frac))
    calib_scenes = scene_order[:n_calib]

    lm_c, est_c, true_c, D_c = pool_scenes(conds, separations, calib_scenes,
                                            args.n_calib_frames_per_scene, rng)

    n_frames = lm_c.shape[0]
    K = lm_c.shape[1]
    print("=" * 90)
    print("Delta_hat vs lambda_star diagnostic -- calibration data, split_seed="
          f"{args.split_seed}, calib_scene_frac={args.calib_scene_frac}")
    print("=" * 90)
    print(f"\ncalibration scene IDs (n={len(calib_scenes)}): {sorted(calib_scenes.tolist())}")
    print(f"number of calibration frames: {n_frames}")
    print(f"number of speaker records (frames x K={K}): {n_frames * K}")

    # ---- Delta_hat (estimated only) and lambda_star (existing, unmodified) ----
    delta_hat_deg = delta_hat_deg_batch(est_c)  # (n_frames,)
    S = compute_calibration_scores(lm_c, est_c, true_c, lambda_list, nele, nazi)  # (n_frames, K) = -lambda_star
    lambda_star = -S  # (n_frames, K)

    # flatten to (frame, speaker) records
    delta_flat = np.repeat(delta_hat_deg, K)          # (n_frames*K,)
    lambda_flat = lambda_star.reshape(-1)             # (n_frames*K,)  order: [frame0_k0,frame0_k1,frame1_k0,...]
    k_flat = np.tile(np.arange(K), n_frames)
    D_flat = np.repeat(D_c, K)

    print(f"\nDelta_hat range: [{delta_hat_deg.min():.2f}, {delta_hat_deg.max():.2f}] deg, "
          f"{len(np.unique(np.round(delta_hat_deg, 2)))} distinct (rounded) values")
    print(f"lambda_star range: [{lambda_star.min():.4f}, {lambda_star.max():.4f}]")

    # =====================================================================
    # Local quantile curves
    # =====================================================================
    lq_pooled = local_quantiles(delta_flat, lambda_flat, args.window_deg, args.min_n_local)
    lq_by_k = {k: local_quantiles(delta_flat[k_flat == k], lambda_flat[k_flat == k],
                                   args.window_deg, args.min_n_local) for k in range(K)}

    # =====================================================================
    # Figures
    # =====================================================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- 3. main scatter + pooled local quantiles ----
    fig, ax = plt.subplots(figsize=(9, 6))
    for k, marker, color in [(0, "o", "tab:blue"), (1, "^", "tab:orange")]:
        m = k_flat == k
        ax.scatter(delta_flat[m], lambda_flat[m], s=14, alpha=0.18, marker=marker, color=color,
                   label=f"speaker k={k}", linewidths=0)
    if lq_pooled:
        ds = [r["d"] for r in lq_pooled]
        ax.plot(ds, [r["median"] for r in lq_pooled], "-", color="black", lw=1.8, label="pooled median")
        ax.plot(ds, [r["q25"] for r in lq_pooled], "--", color="black", lw=1.4, label="pooled Q25")
        ax.plot(ds, [r["q10"] for r in lq_pooled], ":", color="crimson", lw=2.0, label="pooled Q10")
    for edge in BIN_EDGES_DEG[1:-1]:
        ax.axvline(edge, color="gray", lw=0.5, ls=":", alpha=0.5)
    ax.set_xlabel("Estimated speaker separation $\\hat{\\Delta}$ [deg]")
    ax.set_ylabel("lambda_star")
    ax.set_title(f"lambda_star vs $\\hat{{\\Delta}}$ -- calibration data, split_seed={args.split_seed}",
                 fontsize=12)
    fig.text(0.5, 0.955,
              f"local stats: |$\\hat{{\\Delta}}-d$| <= {args.window_deg} deg, n>={args.min_n_local}; "
              "gray dotted = reporting-bin edges from section 7, not a proposed grouping",
              ha="center", fontsize=8, color="dimgray")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(args.out_dir, "lambda_star_vs_hatD.png"), dpi=150)
    fig.savefig(os.path.join(args.out_dir, "lambda_star_vs_hatD.pdf"))
    plt.close(fig)
    print(f"\nSaved -> {args.out_dir}/lambda_star_vs_hatD.png / .pdf")

    # ---- 4. per-speaker local quantiles, second figure ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for k, ax, color, marker in [(0, axes[0], "tab:blue", "o"), (1, axes[1], "tab:orange", "^")]:
        m = k_flat == k
        ax.scatter(delta_flat[m], lambda_flat[m], s=14, alpha=0.18, marker=marker, color=color, linewidths=0)
        lq = lq_by_k[k]
        if lq:
            ds = [r["d"] for r in lq]
            ax.plot(ds, [r["median"] for r in lq], "-", color="black", lw=1.8, label="median")
            ax.plot(ds, [r["q25"] for r in lq], "--", color="black", lw=1.4, label="Q25")
            ax.plot(ds, [r["q10"] for r in lq], ":", color="crimson", lw=2.0, label="Q10")
        for edge in BIN_EDGES_DEG[1:-1]:
            ax.axvline(edge, color="gray", lw=0.5, ls=":", alpha=0.5)
        ax.set_xlabel("Estimated speaker separation $\\hat{\\Delta}$ [deg]")
        ax.set_title(f"speaker k={k}")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("lambda_star")
    fig.suptitle(f"lambda_star vs $\\hat{{\\Delta}}$ by speaker slot -- calibration data, split_seed={args.split_seed}")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "lambda_star_vs_hatD_by_speaker.png"), dpi=150)
    fig.savefig(os.path.join(args.out_dir, "lambda_star_vs_hatD_by_speaker.pdf"))
    plt.close(fig)
    print(f"Saved -> {args.out_dir}/lambda_star_vs_hatD_by_speaker.png / .pdf")

    # ---- 5. support/density plot (frames, not speaker records) ----
    fig, ax = plt.subplots(figsize=(9, 4.5))
    max_d = float(np.ceil(delta_hat_deg.max()))
    bins = np.arange(0, max_d + 2, 1.0)
    ax.hist(delta_hat_deg, bins=bins, color="tab:gray", edgecolor="black", linewidth=0.3)
    for edge in BIN_EDGES_DEG[1:-1]:
        ax.axvline(edge, color="crimson", lw=0.8, ls=":", alpha=0.7)
    ax.set_xlabel("Estimated speaker separation $\\hat{\\Delta}$ [deg]")
    ax.set_ylabel("number of calibration frames")
    ax.set_title(f"Calibration support vs $\\hat{{\\Delta}}$ (1 deg bins), split_seed={args.split_seed}\n"
                 "(red dotted = reporting-bin edges from section 7)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "hatD_calibration_support.png"), dpi=150)
    fig.savefig(os.path.join(args.out_dir, "hatD_calibration_support.pdf"))
    plt.close(fig)
    print(f"Saved -> {args.out_dir}/hatD_calibration_support.png / .pdf")

    # ---- 6. optional true-D colored scatter (diagnostic/interpretation only) ----
    fig, ax = plt.subplots(figsize=(9, 6))
    for D in separations:
        m = D_flat == D
        ax.scatter(delta_flat[m], lambda_flat[m], s=14, alpha=0.35, color=D_COLORS.get(D, "black"),
                   label=f"D={D:.0f} deg", linewidths=0)
    for edge in BIN_EDGES_DEG[1:-1]:
        ax.axvline(edge, color="gray", lw=0.5, ls=":", alpha=0.5)
    ax.set_xlabel("Estimated speaker separation $\\hat{\\Delta}$ [deg]")
    ax.set_ylabel("lambda_star")
    ax.set_title("lambda_star vs $\\hat{\\Delta}$, colored by TRUE commanded D (interpretation only --\n"
                 "true D not used to compute Delta_hat/lambda_star/local stats)")
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "lambda_star_vs_hatD_colored_trueD.png"), dpi=150)
    fig.savefig(os.path.join(args.out_dir, "lambda_star_vs_hatD_colored_trueD.pdf"))
    plt.close(fig)
    print(f"Saved -> {args.out_dir}/lambda_star_vs_hatD_colored_trueD.png / .pdf")

    # =====================================================================
    # 7. Numerical summary over the frozen (reporting-only) Delta_hat bins
    # 8. heavy-tail fractions
    # =====================================================================
    Gc = assign_bins(delta_hat_deg)  # per-frame bin index, using the SAME frozen bins (import, not redefinition)

    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    out("\n" + "=" * 90)
    out("7/8. Numerical summary over the previous FIXED Delta_hat bins (REPORTING ONLY, not a proposed grouping)")
    out("=" * 90)
    header = (f"{'bin':>22} {'n_frm':>6} {'n_rec':>6} {'mean':>7} {'med':>7} {'Q25':>7} {'Q10':>7} {'Q05':>7} "
              f"{'min':>7} {'Q10_k0':>7} {'Q10_k1':>7}")
    out(header)
    for j in range(N_BINS):
        mask_frame = (Gc == j)
        n_frm = int(mask_frame.sum())
        rec_mask = np.repeat(mask_frame, K)
        vals = lambda_flat[rec_mask]
        n_rec = len(vals)
        if n_rec == 0:
            out(f"{BIN_LABELS[j]:>22} {n_frm:6d} {n_rec:6d}   (no records)")
            continue
        vals_k = {k: lambda_flat[rec_mask & (k_flat == k)] for k in range(K)}
        q10_k0 = np.percentile(vals_k[0], 10) if len(vals_k[0]) else float("nan")
        q10_k1 = np.percentile(vals_k[1], 10) if len(vals_k[1]) else float("nan")
        out(f"{BIN_LABELS[j]:>22} {n_frm:6d} {n_rec:6d} {vals.mean():7.4f} {np.median(vals):7.4f} "
            f"{np.percentile(vals,25):7.4f} {np.percentile(vals,10):7.4f} {np.percentile(vals,5):7.4f} "
            f"{vals.min():7.4f} {q10_k0:7.4f} {q10_k1:7.4f}")

    out("\nHeavy-tail / bimodality fractions per bin, per speaker:")
    out(f"{'bin':>22} {'k':>2} {'n':>5} {'frac>=0.95':>11} {'frac<=0.10':>11} {'frac<=0.30':>11}")
    for j in range(N_BINS):
        mask_frame = (Gc == j)
        rec_mask = np.repeat(mask_frame, K)
        for k in range(K):
            vals = lambda_flat[rec_mask & (k_flat == k)]
            if len(vals) == 0:
                out(f"{BIN_LABELS[j]:>22} {k:2d}     0        n/a         n/a         n/a")
                continue
            f95 = float(np.mean(vals >= 0.95))
            f10 = float(np.mean(vals <= 0.10))
            f30 = float(np.mean(vals <= 0.30))
            out(f"{BIN_LABELS[j]:>22} {k:2d} {len(vals):5d} {f95:11.4f} {f10:11.4f} {f30:11.4f}")

    out("\nLocal Q10(lambda_star | Delta_hat), pooled (window=+/-{:.0f} deg, min_n={}):".format(
        args.window_deg, args.min_n_local))
    for r in lq_pooled:
        out(f"  d={r['d']:6.2f}deg  n={r['n']:4d}  median={r['median']:.4f}  Q25={r['q25']:.4f}  Q10={r['q10']:.4f}")

    out_txt = os.path.join(args.out_dir, "lambda_star_vs_hatD_summary.txt")
    with open(out_txt, "w") as fh:
        fh.write(f"Delta_hat vs lambda_star diagnostic -- calibration data only, split_seed={args.split_seed}\n")
        fh.write(f"calibration scenes (n={len(calib_scenes)}): {sorted(calib_scenes.tolist())}\n")
        fh.write(f"n_calib_frames={n_frames}  n_speaker_records={n_frames*K}\n")
        fh.write(f"Delta_hat range: [{delta_hat_deg.min():.2f}, {delta_hat_deg.max():.2f}] deg\n")
        fh.write(f"lambda_star range: [{lambda_star.min():.4f}, {lambda_star.max():.4f}]\n\n")
        fh.write("\n".join(lines) + "\n")
    print(f"\nSaved -> {out_txt}")


if __name__ == "__main__":
    main()
