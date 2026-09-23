"""
Offline diagnostic for Step 2 of the "improve LCP gradually" request: which 4
of 6 candidate static raw-map features best separate burst-active from
burst-inactive frames, and how redundant are they with each other.

Candidate features (per (frame, speaker), from the RAW likelihood map --
i.e. read BEFORE Code.utilities.normalize()):
    raw_mean, raw_peak, raw_median, raw_std, peak/median, peak-median

burst_active_per_frame is used ONLY here, for grouping/labeling this
diagnostic. It is never passed to LCP and never used for h-selection
(Code/two_speaker_tracking/lcp.py and eval_lcp_h_selection.py do not import
this script or read burst_active anywhere).

Does not touch Code/two_speaker_tracking/lcp.py or any LCP code -- this is a
read-only analysis of the existing burst5 npz, run separately.
"""

import argparse

import numpy as np

DATA_ROOT = "/src/data"
DEFAULT_PATH = f"{DATA_ROOT}/npz_output_tracking_burst5_test/Reverb_400_ms_SNR_15_dB/speakers_2_flat.npz"

FEATURE_NAMES = ["raw_mean", "raw_peak", "raw_median", "raw_std", "peak_over_median", "peak_minus_median"]


def compute_features(likelihood_maps_raw, eps=1e-8):
    """likelihood_maps_raw: (..., nele, nazi) raw. Returns (..., 6) in FEATURE_NAMES order."""
    arr = np.asarray(likelihood_maps_raw, dtype=float)
    flat = arr.reshape(arr.shape[:-2] + (-1,))
    raw_mean = flat.mean(axis=-1)
    raw_peak = flat.max(axis=-1)
    raw_median = np.median(flat, axis=-1)
    raw_std = flat.std(axis=-1)
    peak_over_median = raw_peak / (raw_median + eps)
    peak_minus_median = raw_peak - raw_median
    return np.stack([raw_mean, raw_peak, raw_median, raw_std, peak_over_median, peak_minus_median], axis=-1)


def summarize_group(x):
    return dict(
        mean=float(np.mean(x)), std=float(np.std(x)),
        median=float(np.median(x)),
        q25=float(np.percentile(x, 25)), q75=float(np.percentile(x, 75)),
        n=int(len(x)),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_path", default=DEFAULT_PATH)
    args = p.parse_args()

    d = np.load(args.data_path, allow_pickle=True)
    lm_all = d["all_likelihood_maps"]  # (N_flat, K, nele, nazi), raw
    burst_active_all = d["burst_active_per_frame"].astype(bool)  # (N_flat,) -- frame-level label
    N_flat, K = lm_all.shape[:2]

    # Features per (frame, speaker); the frame-level burst_active label applies
    # to both speaker slots for that frame, matching how burst_active is used
    # elsewhere in this codebase (eval_lcp_repeated_splits.py, etc.).
    feats = compute_features(lm_all)  # (N_flat, K, 6)
    feats_flat = feats.reshape(N_flat * K, 6)
    active_flat = np.repeat(burst_active_all, K)

    active = feats_flat[active_flat]
    inactive = feats_flat[~active_flat]

    print("=" * 78)
    print(f"Data: {args.data_path}")
    print(f"N_flat={N_flat} frames, K={K} speakers -> {N_flat*K} (frame,speaker) records")
    print(f"burst-active records: {active.shape[0]}  ({active.shape[0]/(N_flat*K)*100:.2f}%)")
    print(f"burst-inactive records: {inactive.shape[0]}")
    print("=" * 78)

    print()
    print(f"{'feature':<20s} {'active_mean':>12s} {'active_std':>11s} {'inactive_mean':>14s} "
          f"{'inactive_std':>13s} {'sep_score':>10s}")
    sep_scores = {}
    for j, name in enumerate(FEATURE_NAMES):
        a = active[:, j]
        ia = inactive[:, j]
        pooled_std = np.sqrt(0.5 * (np.var(a) + np.var(ia)))
        sep = float(abs(np.mean(a) - np.mean(ia)) / (pooled_std + 1e-12))
        sep_scores[name] = sep
        print(f"{name:<20s} {np.mean(a):12.4f} {np.std(a):11.4f} {np.mean(ia):14.4f} "
              f"{np.std(ia):13.4f} {sep:10.4f}")

    print()
    print("Median / IQR (robust alternative view):")
    print(f"{'feature':<20s} {'active_median':>13s} {'active_IQR':>11s} {'inactive_median':>16s} {'inactive_IQR':>13s}")
    for j, name in enumerate(FEATURE_NAMES):
        a = active[:, j]
        ia = inactive[:, j]
        a_iqr = np.percentile(a, 75) - np.percentile(a, 25)
        ia_iqr = np.percentile(ia, 75) - np.percentile(ia, 25)
        print(f"{name:<20s} {np.median(a):13.4f} {a_iqr:11.4f} {np.median(ia):16.4f} {ia_iqr:13.4f}")

    print()
    print("Standardized separation score (|mean_active - mean_inactive| / pooled_std), ranked:")
    for name, s in sorted(sep_scores.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<20s} {s:.4f}")

    print()
    print("Pairwise Pearson correlation matrix (all records, active+inactive pooled):")
    corr = np.corrcoef(feats_flat, rowvar=False)
    header = "                    " + "".join(f"{n:>18s}" for n in FEATURE_NAMES)
    print(header)
    for i, name in enumerate(FEATURE_NAMES):
        row = "".join(f"{corr[i,j]:18.3f}" for j in range(len(FEATURE_NAMES)))
        print(f"{name:<20s}{row}")

    # --- recommend 4 features balancing separation and redundancy ---
    print()
    print("=" * 78)
    print("Recommendation: greedy selection maximizing separation while penalizing")
    print("redundancy (|correlation| with already-chosen features)")
    print("=" * 78)
    ranked = [name for name, _ in sorted(sep_scores.items(), key=lambda kv: -kv[1])]
    name_to_idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    selected = [ranked[0]]
    remaining = ranked[1:]
    while len(selected) < 4 and remaining:
        best_name, best_score = None, -np.inf
        for cand in remaining:
            ci = name_to_idx[cand]
            max_corr_with_selected = max(abs(corr[ci, name_to_idx[s]]) for s in selected)
            score = sep_scores[cand] - max_corr_with_selected  # simple redundancy penalty
            if score > best_score:
                best_score, best_name = score, cand
        selected.append(best_name)
        remaining.remove(best_name)
    print(f"Selected (in order added): {selected}")
    print(f"(dropped: {[n for n in FEATURE_NAMES if n not in selected]})")


if __name__ == "__main__":
    main()
