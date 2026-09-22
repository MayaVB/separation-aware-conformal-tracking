"""
Step 2 follow-up: the original 6-candidate diagnostic (analyze_lcp_feature_diagnostic.py)
found that raw_mean/raw_peak/raw_median/raw_std are pairwise correlated at
0.89-0.99 -- essentially one "magnitude" axis -- and that peak_over_median was
degenerate (near-zero separation, driven by outliers from a near-zero
denominator). This script re-runs the diagnostic with the 4 magnitude
features kept PLUS 4 new dimensionless/contrast features:

    peak_over_mean        = raw_peak / (raw_mean + eps)
    std_over_mean         = raw_std / (raw_mean + eps)
    median_over_mean      = raw_median / (raw_mean + eps)
    log_peak_over_median  = log((raw_peak + eps) / (raw_median + eps))

All features are computed from the RAW (pre-normalize()) likelihood map, per
(frame, speaker). burst_active_per_frame is used ONLY for this offline
diagnostic (never as an LCP input, never for h-selection). Does not touch
Code/two_speaker_tracking/lcp.py, eval_lcp_h_selection.py, or the running
background h-selection job.

Epsilon note: a direct scan of this dataset found raw_mean/raw_median can go
slightly NEGATIVE (min ~ -0.011, in 30/35 of 20800 records respectively,
overwhelmingly in burst-INACTIVE frames -- 29/30 and 34/35), evidently
near-silent-frame numerical noise in the raw SRP-DNN map, not a burst
artifact. raw_peak is always positive (min 0.0065). To keep every ratio/log
denominator positive (required for log_peak_over_median to avoid NaN), eps is
set to 0.02 -- about 2x the most negative observed raw_mean/raw_median value,
comfortably larger than the ~0.15% of records where the raw denominator is
near/below zero, while still small relative to the typical raw_mean scale
(~0.14-0.19). This eps is applied uniformly to every denominator below.
"""

import numpy as np

DATA_ROOT = "/src/data"
DEFAULT_PATH = f"{DATA_ROOT}/npz_output_tracking_burst5_test/Reverb_400_ms_SNR_15_dB/speakers_2_flat.npz"
EPS = 0.02

MAGNITUDE_FEATURES = ["raw_mean", "raw_median", "raw_peak", "raw_std"]
CONTRAST_FEATURES = ["peak_over_mean", "std_over_mean", "median_over_mean", "log_peak_over_median"]
ALL_FEATURES = MAGNITUDE_FEATURES + CONTRAST_FEATURES


def compute_features(likelihood_maps_raw, eps=EPS):
    arr = np.asarray(likelihood_maps_raw, dtype=float)
    flat = arr.reshape(arr.shape[:-2] + (-1,))
    raw_mean = flat.mean(axis=-1)
    raw_median = np.median(flat, axis=-1)
    raw_peak = flat.max(axis=-1)
    raw_std = flat.std(axis=-1)

    peak_over_mean = raw_peak / (raw_mean + eps)
    std_over_mean = raw_std / (raw_mean + eps)
    median_over_mean = raw_median / (raw_mean + eps)
    log_peak_over_median = np.log((raw_peak + eps) / (raw_median + eps))

    return {
        "raw_mean": raw_mean, "raw_median": raw_median, "raw_peak": raw_peak, "raw_std": raw_std,
        "peak_over_mean": peak_over_mean, "std_over_mean": std_over_mean,
        "median_over_mean": median_over_mean, "log_peak_over_median": log_peak_over_median,
    }


def mad(x):
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


def sep_score(a, ia):
    pooled_std = np.sqrt(0.5 * (np.var(a) + np.var(ia)))
    return float(abs(np.mean(a) - np.mean(ia)) / (pooled_std + 1e-12))


def robust_sep_score(a, ia):
    mad_a, mad_ia = mad(a), mad(ia)
    pooled_mad = np.sqrt(0.5 * (mad_a ** 2 + mad_ia ** 2))
    return float(abs(np.median(a) - np.median(ia)) / (pooled_mad + 1e-12))


def outlier_fraction(x, k=5.0):
    """Fraction of x beyond median +/- k*MAD (robust outlier rule)."""
    med = np.median(x)
    m = mad(x)
    if m < 1e-12:
        return 0.0
    return float(np.mean(np.abs(x - med) > k * m))


def winsorize(x, lo_pct=1, hi_pct=99):
    lo, hi = np.percentile(x, [lo_pct, hi_pct])
    return np.clip(x, lo, hi)


def main():
    d = np.load(DEFAULT_PATH, allow_pickle=True)
    lm_all = d["all_likelihood_maps"]
    burst_active_all = d["burst_active_per_frame"].astype(bool)
    N_flat, K = lm_all.shape[:2]

    feats_2d = compute_features(lm_all)  # each value shape (N_flat, K)
    feats = {name: arr.reshape(-1) for name, arr in feats_2d.items()}  # flatten to (N_flat*K,)
    active_flat = np.repeat(burst_active_all, K)

    print("=" * 90)
    print(f"N_flat={N_flat}, K={K} -> {N_flat*K} records; "
          f"active={active_flat.sum()} ({active_flat.mean()*100:.2f}%); eps={EPS}")
    print("=" * 90)

    # --- unclipped stats + outlier check for the 4 new features ---
    print()
    print("Unclipped stats, all 8 candidates:")
    print(f"{'feature':<22s} {'act_mean':>10s} {'act_std':>10s} {'inact_mean':>11s} {'inact_std':>10s} "
          f"{'sep':>7s} {'act_med':>9s} {'act_IQR':>9s} {'inact_med':>10s} {'inact_IQR':>9s} {'robust_sep':>10s}")
    sep_scores, robust_scores, feat_arrays = {}, {}, {}
    for name in ALL_FEATURES:
        x = feats[name]
        a, ia = x[active_flat], x[~active_flat]
        feat_arrays[name] = x
        s = sep_score(a, ia)
        rs = robust_sep_score(a, ia)
        sep_scores[name] = s
        robust_scores[name] = rs
        a_iqr = np.percentile(a, 75) - np.percentile(a, 25)
        ia_iqr = np.percentile(ia, 75) - np.percentile(ia, 25)
        print(f"{name:<22s} {np.mean(a):10.4f} {np.std(a):10.4f} {np.mean(ia):11.4f} {np.std(ia):10.4f} "
              f"{s:7.4f} {np.median(a):9.4f} {a_iqr:9.4f} {np.median(ia):10.4f} {ia_iqr:9.4f} {rs:10.4f}")

    print()
    print("Outlier check for the 4 NEW contrast/ratio features (fraction beyond median +/- 5*MAD):")
    for name in CONTRAST_FEATURES:
        x = feat_arrays[name]
        frac = outlier_fraction(x, k=5.0)
        n_out = int(round(frac * len(x)))
        print(f"  {name:<22s} outlier_frac={frac*100:.3f}%  (n={n_out} of {len(x)})  "
              f"min={x.min():.4f} max={x.max():.4f} max_abs_z_from_median_mad="
              f"{np.max(np.abs(x - np.median(x)) / (mad(x)+1e-12)):.2f}")

    # --- robust/clipped (winsorized) version, side by side ---
    print()
    print("=" * 90)
    print("Robust/clipped (winsorized at 1st/99th pctile, pooled) stats for the 4 NEW features:")
    print("=" * 90)
    print(f"{'feature':<22s} {'act_mean':>10s} {'act_std':>10s} {'inact_mean':>11s} {'inact_std':>10s} {'sep':>7s} {'robust_sep':>10s}")
    winsorized = {}
    for name in CONTRAST_FEATURES:
        x = feat_arrays[name]
        xw = winsorize(x)
        winsorized[name] = xw
        a, ia = xw[active_flat], xw[~active_flat]
        s = sep_score(a, ia)
        rs = robust_sep_score(a, ia)
        print(f"{name:<22s} {np.mean(a):10.4f} {np.std(a):10.4f} {np.mean(ia):11.4f} {np.std(ia):10.4f} {s:7.4f} {rs:10.4f}")

    print()
    print("(unclipped sep/robust_sep for the same 4, repeated for direct before/after comparison):")
    for name in CONTRAST_FEATURES:
        print(f"  {name:<22s} unclipped: sep={sep_scores[name]:.4f} robust_sep={robust_scores[name]:.4f}   "
              f"winsorized: sep={sep_score(winsorized[name][active_flat], winsorized[name][~active_flat]):.4f} "
              f"robust_sep={robust_sep_score(winsorized[name][active_flat], winsorized[name][~active_flat]):.4f}")

    # --- correlation matrix: use winsorized versions for the 4 ratio features
    # (unclipped ratios are outlier-dominated, as the original peak_over_median
    # diagnostic showed; magnitude features are not winsorized, they had no
    # outlier problem in the first diagnostic) ---
    print()
    print("=" * 90)
    print("Pairwise Pearson correlation (magnitude features raw; ratio/log features winsorized):")
    print("=" * 90)
    corr_input = np.stack(
        [feat_arrays[n] for n in MAGNITUDE_FEATURES] + [winsorized[n] for n in CONTRAST_FEATURES], axis=-1)
    corr = np.corrcoef(corr_input, rowvar=False)
    header = "                        " + "".join(f"{n:>16s}" for n in ALL_FEATURES)
    print(header)
    for i, name in enumerate(ALL_FEATURES):
        row = "".join(f"{corr[i, j]:16.3f}" for j in range(len(ALL_FEATURES)))
        print(f"{name:<24s}{row}")

    print()
    print("Ranked standardized separation (unclipped) / robust separation:")
    for name in sorted(ALL_FEATURES, key=lambda n: -sep_scores[n]):
        print(f"  {name:<22s} sep={sep_scores[name]:.4f}  robust_sep={robust_scores[name]:.4f}")

    # --- recommendation ---
    print()
    print("=" * 90)
    print("Recommendation")
    print("=" * 90)
    name_to_idx = {n: i for i, n in enumerate(ALL_FEATURES)}
    HIGH_CORR = 0.8
    # cluster magnitude features (all pairwise > HIGH_CORR per the v1 diagnostic)
    mag_block_corr = [abs(corr[name_to_idx[a], name_to_idx[b]])
                       for i, a in enumerate(MAGNITUDE_FEATURES) for b in MAGNITUDE_FEATURES[i+1:]]
    print(f"Magnitude-block pairwise |corr| range: {min(mag_block_corr):.3f} - {max(mag_block_corr):.3f}")
    contrast_vs_mag = {}
    for c in CONTRAST_FEATURES:
        contrast_vs_mag[c] = max(abs(corr[name_to_idx[c], name_to_idx[m]]) for m in MAGNITUDE_FEATURES)
    print("Max |corr(contrast, any magnitude feature)|:")
    for c, v in sorted(contrast_vs_mag.items(), key=lambda kv: kv[1]):
        print(f"  {c:<22s} {v:.3f}  robust_sep={robust_scores[c]:.4f}")


if __name__ == "__main__":
    main()
