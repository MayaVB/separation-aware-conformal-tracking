"""
Tier-0 cancellation-mechanism diagnostic: test whether the k=1 heavy tail
(established in diagnose_lambda_star_vs_estimated_separation.py) is
associated with observable consequences/proxies of IDL's successive
interference-cancellation mechanism (Module.py's SourceDetectLocalize,
meth_mode='IDL': k=0 = argmax of the raw spectrum; k=1 = argmax of the
spectrum AFTER subtracting a fitted template for k=0 -- see
Module.py:1191-1229).

Tier 0 ONLY: no source-code changes to SRP-DNN, no model rerun, no new
inference. Uses only:
  (A) all_likelihood_maps[:,0]/[:,1], already in the existing npz (the
      raw pre-subtraction map and the post-subtraction residual map,
      respectively -- iter_maps_list captures `map` BEFORE the argmax/
      subtraction step of EACH iteration, so index 1 is exactly the
      spectrum IDL saw after removing its fitted k=0 template).
  (B) an OFFLINE reconstruction of the DPIPD template used internally
      by SRP-DNN (Module.py's DPIPD class, Module.py:1068-1157), built
      here from the same closed-form formula, same fixed microphone
      geometry (Dataset.py's benchmark2_array_setup.mic_pos, the 12-ch
      '3D' array RunSRPDNN.py selects via `array='12ch'`), and the same
      STFT/frequency-bin configuration (nfft=512, fs=16000,
      fre_used_ratio=1 -> fre_range_used=range(1,257), ch_mode='MM' ->
      all C(12,2)=66 mic pairs) -- copied as fixed constants/formula,
      not imported, to avoid any dependency on SRP-DNN/code/Dataset.py's
      module-level gpuRIR/webrtcvad imports (irrelevant to this
      diagnostic and not guaranteed importable in this analysis
      environment). The formula is transcribed verbatim from
      Module.py:1082-1099 (per-grid-cell ITD/IPD) and Module.py:1143-1157
      (data_adjust, ch_mode='MM': the ordered upper-triangular mic-pair
      extraction) -- see build_dpipd_vector() docstring for the exact
      steps and Module.py line references.

Nothing here is claimed as a validated causal cancellation-quality
metric -- see section E of the request this script answers. r_peak/
r_mean are "residual-to-initial map-strength ratios"; template_overlap
is a normalized-inner-product geometric similarity between the two
detected cells' ideal IPD templates. Both are proxies/observable
consequences of the IDL mechanism, not proofs of it.

Calibration data: split_seed=0, calib_scene_frac=0.5,
n_calib_frames_per_scene=10 -- identical to
diagnose_lambda_star_vs_estimated_separation.py (12 scenes, 840 frames,
1680 speaker records), via the same imported, unmodified pool_scenes.
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..")))

import argparse
import os

import numpy as np
from scipy.stats import spearmanr

from Code.two_speaker_tracking.lcp import compute_calibration_scores

from eval_angular_separation import load_conditions, pool_scenes, DEFAULT_SEPARATIONS
from eval_oracle_mondrian_angular_separation import DEFAULT_DATA_PATHS
from eval_deployable_mondrian_separation import delta_hat_deg_batch, BIN_EDGES_DEG
from diagnose_lambda_star_vs_estimated_separation import local_quantiles

# ---------------------------------------------------------------------------
# Fixed geometry/config constants, transcribed verbatim (not imported) from:
#   SRP-DNN/code/Dataset.py:176-190   (benchmark2_array_setup.mic_pos)
#   SRP-DNN/code/RunSRPDNN.py:54,64-69,193-196,305  (fs, array='12ch', nfft,
#                                                     fre_used_ratio, ch_mode)
#   SRP-DNN/code/LearnerSRPDNN.py:11,28-32 (c=343.0 default, DPIPD ctor args)
# ---------------------------------------------------------------------------
MIC_POS = np.array((
    (-0.028,  0.030, -0.040), ( 0.006,  0.057,  0.000), ( 0.022,  0.022, -0.046),
    (-0.055, -0.024, -0.025), (-0.031,  0.023,  0.042), (-0.032,  0.011,  0.046),
    (-0.025, -0.003,  0.051), (-0.036, -0.027,  0.038), (-0.035, -0.043,  0.025),
    ( 0.029, -0.048, -0.012), ( 0.034, -0.030,  0.037), ( 0.035,  0.025,  0.039),
))  # (12, 3), meters -- SRP-DNN's benchmark2_array_setup ('12ch', 3D)
NFFT = 512
FS = 16000.0
FRE_MAX = FS / 2.0          # = 8000.0, Module.py's `fre_max = fs / 2`
NF = int(NFFT / 2) + 1      # = 257, DPIPD's `nf` (LearnerSRPDNN.py: nf=int(self.nfft/2)+1)
FRE_RANGE_USED = range(1, int(NFFT / 2 * 1) + 1, 1)  # = range(1,257): LearnerSRPDNN.py's
                                                      # fre_used_ratio==1 branch, drops DC bin 0
SPEED = 343.0                # SourceTrackingFromSTFTLearner's default c=343.0


def build_dpipd_vector(ele, azi, mic_pos=MIC_POS, fre_max=FRE_MAX, nf=NF,
                        fre_range_used=FRE_RANGE_USED, speed=SPEED):
    """The ideal (noise-free) IPD template for ONE (ele, azi) grid point, in
    the exact representation SourceDetectLocalize.forward correlates the
    predicted IPD against (Module.py:1182-1184's `dpipd_template`).

    Formula, transcribed from Module.py:1082-1099 (per-mic-pair ITD/IPD,
    ch_mode='MM' ordering from data_adjust, Module.py:1143-1157) and
    LearnerSRPDNN.py:143-144 (fre_range_used slicing + real/imag concat):

      1. unit look-direction vector r = [sin(ele)cos(azi), sin(ele)sin(azi), cos(ele)]
      2. for every mic pair (m1<m2), m1,m2 in 0..11 (66 pairs, ch_mode='MM'
         keeps every pair, not just vs. a reference mic):
           ITD(m1,m2) = r . (mic_pos[m2] - mic_pos[m1]) / speed
           IPD(m1,m2,f) = -2*pi*f*ITD(m1,m2)  for f in linspace(0, fre_max, nf)
      3. complex template T(m1,m2,f) = exp(1j * IPD(m1,m2,f))  (unit modulus)
      4. drop the DC bin (f-index 0) -- fre_range_used
      5. concatenate [Re(T), Im(T)] along the frequency axis (matches
         LearnerSRPDNN.py:143's np.concatenate((...).real, (...).imag), axis=2))
      6. flatten -> a single real vector of length 2*(nf-1)*n_pairs = 2*256*66 = 33792

    Because every (mic pair, freq) entry has |exp(i*theta)|=1, every grid
    cell's vector has the SAME L2 norm (sqrt((nf-1)*n_pairs), since the
    real/imag split redistributes each entry's unit squared-magnitude
    across two components without changing the total) regardless of
    (ele, azi) -- so the natural normalized inner product for this
    representation IS a plain dot-product cosine similarity; no
    cell-dependent renormalization is needed. Verified empirically at
    runtime (see main(), norm sanity check).
    """
    nmic = mic_pos.shape[0]
    r = np.array([np.sin(ele) * np.cos(azi), np.sin(ele) * np.sin(azi), np.cos(ele)])
    fre_range = np.linspace(0.0, fre_max, nf)  # (nf,)
    pairs = []
    for m1 in range(nmic - 1):
        for m2 in range(m1 + 1, nmic):
            itd = float(np.dot(r, mic_pos[m2] - mic_pos[m1]) / speed)
            ipd = -2.0 * np.pi * fre_range * itd  # (nf,)
            pairs.append(np.exp(1j * ipd))
    T = np.stack(pairs, axis=-1)  # (nf, n_pairs=66)
    idx = list(fre_range_used)
    T = T[idx, :]  # (nf_used=256, 66)
    T_ri = np.concatenate([T.real, T.imag], axis=0)  # (2*nf_used, 66)
    return T_ri.reshape(-1)  # (2*nf_used*n_pairs,) = (33792,)


class TemplateCache:
    def __init__(self):
        self._cache = {}

    def get(self, ele, azi):
        key = (round(float(ele), 6), round(float(azi), 6))
        if key not in self._cache:
            self._cache[key] = build_dpipd_vector(ele, azi)
        return self._cache[key]

    def __len__(self):
        return len(self._cache)


def cosine_sim(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb))


def group_stats(vals, label, out):
    vals = np.asarray(vals)
    out(f"  {label}: n={len(vals)}  mean={vals.mean():.4f}  median={np.median(vals):.4f}  "
        f"Q10={np.percentile(vals,10):.4f}  Q90={np.percentile(vals,90):.4f}")


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
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--out_dir", default="Results")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    separations = args.separations

    conds, nele, nazi, common_scenes = load_conditions(args.data_paths, separations)
    lambda_list = np.linspace(0.0, 1.0, args.lambda_steps)

    rng = np.random.default_rng(args.split_seed)
    scene_order = rng.permutation(common_scenes)
    n_calib = int(round(len(scene_order) * args.calib_scene_frac))
    calib_scenes = scene_order[:n_calib]
    lm_c, est_c, true_c, D_c = pool_scenes(conds, separations, calib_scenes,
                                            args.n_calib_frames_per_scene, rng)
    n_frames = lm_c.shape[0]

    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 90)
    out(f"Tier-0 cancellation diagnostic -- split_seed={args.split_seed}, "
        f"{len(calib_scenes)} scenes, {n_frames} frames, {n_frames*2} speaker records")
    out("=" * 90)

    S = compute_calibration_scores(lm_c, est_c, true_c, lambda_list, nele, nazi)  # (n,K) = -lambda_star
    lambda_star = -S
    lambda_star_k1 = lambda_star[:, 1]
    delta_hat_deg = delta_hat_deg_batch(est_c)

    # =====================================================================
    # A. residual-map strength
    # =====================================================================
    out("\n" + "=" * 90)
    out("A. Residual-map strength (r_peak, r_mean)")
    out("=" * 90)
    P0 = lm_c[:, 0].reshape(n_frames, -1).max(axis=1)
    P1 = lm_c[:, 1].reshape(n_frames, -1).max(axis=1)
    M0 = lm_c[:, 0].reshape(n_frames, -1).mean(axis=1)
    M1 = lm_c[:, 1].reshape(n_frames, -1).mean(axis=1)
    r_peak = P1 / (P0 + args.eps)
    r_mean = M1 / (M0 + args.eps)
    out(f"r_peak: mean={r_peak.mean():.4f} median={np.median(r_peak):.4f} "
        f"range=[{r_peak.min():.4f},{r_peak.max():.4f}]")
    out(f"r_mean: mean={r_mean.mean():.4f} median={np.median(r_mean):.4f} "
        f"range=[{r_mean.min():.4f},{r_mean.max():.4f}]")

    rho_rpeak_hatD, p_rpeak_hatD = spearmanr(r_peak, delta_hat_deg)
    rho_rpeak_lam, p_rpeak_lam = spearmanr(r_peak, lambda_star_k1)
    rho_rmean_hatD, p_rmean_hatD = spearmanr(r_mean, delta_hat_deg)
    rho_rmean_lam, p_rmean_lam = spearmanr(r_mean, lambda_star_k1)
    out(f"\nSpearman(r_peak, hatD)          = {rho_rpeak_hatD:+.4f}  (p={p_rpeak_hatD:.2e})")
    out(f"Spearman(r_peak, lambda_star_k1) = {rho_rpeak_lam:+.4f}  (p={p_rpeak_lam:.2e})")
    out(f"Spearman(r_mean, hatD)          = {rho_rmean_hatD:+.4f}  (p={p_rmean_hatD:.2e})")
    out(f"Spearman(r_mean, lambda_star_k1) = {rho_rmean_lam:+.4f}  (p={p_rmean_lam:.2e})")

    bad = lambda_star_k1 <= 0.30
    good = lambda_star_k1 >= 0.95
    out(f"\nBAD (lambda_star_k1<=0.30) vs GOOD (lambda_star_k1>=0.95), r_peak/r_mean:")
    group_stats(r_peak[bad], "r_peak  BAD ", out)
    group_stats(r_peak[good], "r_peak  GOOD", out)
    group_stats(r_mean[bad], "r_mean  BAD ", out)
    group_stats(r_mean[good], "r_mean  GOOD", out)

    lq_rpeak_hatD = local_quantiles(delta_hat_deg, r_peak, args.window_deg, args.min_n_local)
    lq_lam_hatD = local_quantiles(delta_hat_deg, lambda_star_k1, args.window_deg, args.min_n_local)

    # =====================================================================
    # B. offline IPD-template overlap
    # =====================================================================
    out("\n" + "=" * 90)
    out("B. Offline IPD-template overlap")
    out("=" * 90)
    out("template_overlap(y0,y1) = <T(y0),T(y1)> / (||T(y0)|| * ||T(y1)||)")
    out("where T(ele,azi) is the flattened real [Re,Im]-concatenated DPIPD template")
    out(f"(nf_used={len(list(FRE_RANGE_USED))} freq bins x n_pairs=66 mic pairs -> "
        f"{2*len(list(FRE_RANGE_USED))*66}-dim vector), built offline via build_dpipd_vector() "
        "using SRP-DNN's fixed 12-mic geometry and STFT config (see module docstring). "
        "Evaluated at the two ESTIMATED (already on-grid) DOAs -- no GT used.")

    cache = TemplateCache()
    template_overlap = np.empty(n_frames)
    for i in range(n_frames):
        T0 = cache.get(est_c[i, 0, 0], est_c[i, 0, 1])
        T1 = cache.get(est_c[i, 1, 0], est_c[i, 1, 1])
        template_overlap[i] = cosine_sim(T0, T1)
    out(f"\nunique grid cells needed: {len(cache)} (of {n_frames*2} lookups)")

    norms = [np.linalg.norm(v) for v in list(cache._cache.values())[:10]]
    n_used = len(list(FRE_RANGE_USED))
    out(f"norm sanity check (first 10 cached templates): {['%.3f'%v for v in norms]} "
        f"(expected constant = sqrt(nf_used*n_pairs) = sqrt({n_used}*66) = "
        f"{np.sqrt(n_used*66):.3f} -- each of the nf_used*n_pairs unit-modulus complex "
        "template entries contributes exactly 1 to the total sum-of-squares regardless of "
        "how the real/imag split distributes it across the flattened vector)")

    out(f"\ntemplate_overlap: mean={template_overlap.mean():.4f} median={np.median(template_overlap):.4f} "
        f"range=[{template_overlap.min():.4f},{template_overlap.max():.4f}]")

    # =====================================================================
    # C. test the mechanism
    # =====================================================================
    out("\n" + "=" * 90)
    out("C. template_overlap vs hatD, lambda_star_k1, r_peak")
    out("=" * 90)
    rho_to_hatD, p_to_hatD = spearmanr(template_overlap, delta_hat_deg)
    rho_to_lam, p_to_lam = spearmanr(template_overlap, lambda_star_k1)
    rho_to_rpeak, p_to_rpeak = spearmanr(template_overlap, r_peak)
    out(f"Spearman(template_overlap, hatD)          = {rho_to_hatD:+.4f}  (p={p_to_hatD:.2e})")
    out(f"Spearman(template_overlap, lambda_star_k1) = {rho_to_lam:+.4f}  (p={p_to_lam:.2e})")
    out(f"Spearman(template_overlap, r_peak)          = {rho_to_rpeak:+.4f}  (p={p_to_rpeak:.2e})")

    out("\nBAD (lambda_star_k1<=0.30) vs GOOD (lambda_star_k1>=0.95), template_overlap:")
    group_stats(template_overlap[bad], "template_overlap  BAD ", out)
    group_stats(template_overlap[good], "template_overlap  GOOD", out)

    # =====================================================================
    # D. non-monotonic structure
    # =====================================================================
    out("\n" + "=" * 90)
    out("D. Local template_overlap vs hatD -- compared to local Q10(lambda_star_k1|hatD)")
    out("=" * 90)
    lq_to_hatD = local_quantiles(delta_hat_deg, template_overlap, args.window_deg, args.min_n_local)
    for r in lq_to_hatD:
        out(f"  d={r['d']:6.2f}deg  n={r['n']:4d}  median(overlap)={r['median']:.4f}  "
            f"Q25={r['q25']:.4f}  Q10={r['q10']:.4f}")

    # monotonicity check on the local median sequence (sorted by d, by construction)
    med_seq = np.array([r["median"] for r in lq_to_hatD])
    d_seq = np.array([r["d"] for r in lq_to_hatD])
    diffs = np.diff(med_seq)
    tol = 0.01
    n_up = int(np.sum(diffs > tol))
    n_down = int(np.sum(diffs < -tol))
    n_flat = len(diffs) - n_up - n_down
    reversals = int(np.sum(np.diff(np.sign(np.where(np.abs(diffs) > tol, diffs, 0))) != 0))
    minority = min(n_up, n_down)
    if minority == 0:
        direction = "non-decreasing" if n_down == 0 else "non-increasing"
        verdict = f"MONOTONIC ({direction})"
    elif minority <= 2:
        direction = "non-decreasing" if n_down < n_up else "non-increasing"
        verdict = f"APPROXIMATELY MONOTONIC ({direction}, {minority} minority-direction step(s))"
    else:
        verdict = "CLEARLY NON-MONOTONIC"
    out(f"\nlocal median(template_overlap) sequence: {n_up} up-steps, {n_down} down-steps, "
        f"{n_flat} ~flat (tol={tol}) out of {len(diffs)} consecutive gaps; "
        f"~{reversals} direction reversals -> {verdict}")

    # =====================================================================
    # figures
    # =====================================================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.scatter(delta_hat_deg, r_peak, s=12, alpha=0.25, color="tab:blue", linewidths=0)
    if lq_rpeak_hatD:
        ds = [r["d"] for r in lq_rpeak_hatD]
        ax.plot(ds, [r["median"] for r in lq_rpeak_hatD], "-", color="black", lw=1.6, label="local median")
        ax.plot(ds, [r["q10"] for r in lq_rpeak_hatD], ":", color="crimson", lw=1.8, label="local Q10")
    for edge in BIN_EDGES_DEG[1:-1]:
        ax.axvline(edge, color="gray", lw=0.4, ls=":", alpha=0.5)
    ax.set_xlabel("$\\hat{\\Delta}$ [deg]"); ax.set_ylabel("r_peak = P1/P0")
    ax.set_title("A. residual/initial peak ratio vs $\\hat{\\Delta}$")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.scatter(lambda_star_k1, r_peak, s=12, alpha=0.25, color="tab:orange", linewidths=0)
    ax.set_xlabel("lambda_star (k=1)"); ax.set_ylabel("r_peak = P1/P0")
    ax.set_title(f"A. r_peak vs lambda_star_k1 (Spearman={rho_rpeak_lam:+.3f})")

    ax = axes[1, 0]
    ax.scatter(delta_hat_deg, template_overlap, s=12, alpha=0.25, color="tab:green", linewidths=0)
    if lq_to_hatD:
        ds = [r["d"] for r in lq_to_hatD]
        ax.plot(ds, [r["median"] for r in lq_to_hatD], "-", color="black", lw=1.6, label="local median")
        ax.plot(ds, [r["q10"] for r in lq_to_hatD], ":", color="crimson", lw=1.8, label="local Q10")
    for edge in BIN_EDGES_DEG[1:-1]:
        ax.axvline(edge, color="gray", lw=0.4, ls=":", alpha=0.5)
    ax.set_xlabel("$\\hat{\\Delta}$ [deg]"); ax.set_ylabel("template_overlap")
    ax.set_title("B/D. IPD-template cosine overlap vs $\\hat{\\Delta}$")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax2 = ax.twinx()
    if lq_to_hatD:
        ds = [r["d"] for r in lq_to_hatD]
        l1, = ax.plot(ds, [r["median"] for r in lq_to_hatD], "-", color="tab:green", lw=2.0,
                       label="local median(template_overlap)")
    if lq_lam_hatD:
        ds2 = [r["d"] for r in lq_lam_hatD]
        l2, = ax2.plot(ds2, [r["q10"] for r in lq_lam_hatD], ":", color="crimson", lw=2.0,
                        label="local Q10(lambda_star_k1)")
    for edge in BIN_EDGES_DEG[1:-1]:
        ax.axvline(edge, color="gray", lw=0.4, ls=":", alpha=0.5)
    ax.set_xlabel("$\\hat{\\Delta}$ [deg]")
    ax.set_ylabel("template_overlap (median)", color="tab:green")
    ax2.set_ylabel("Q10(lambda_star_k1)", color="crimson")
    ax.set_title("D. alignment: template_overlap vs Q10(lambda_star_k1)")
    lines_ = [l1, l2] if lq_to_hatD and lq_lam_hatD else []
    if lines_:
        ax.legend(lines_, [l.get_label() for l in lines_], fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "cancellation_tier0_diagnostic.png"), dpi=150)
    fig.savefig(os.path.join(args.out_dir, "cancellation_tier0_diagnostic.pdf"))
    plt.close(fig)
    out(f"\nSaved -> {args.out_dir}/cancellation_tier0_diagnostic.png / .pdf")

    out_txt = os.path.join(args.out_dir, "cancellation_tier0_summary.txt")
    with open(out_txt, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nSaved -> {out_txt}")


if __name__ == "__main__":
    main()
