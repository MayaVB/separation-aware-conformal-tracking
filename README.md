# Separation-aware conformal tracking

Two-speaker DOA tracking on SRP-DNN heatmaps, using conformal-prediction (CP) region
area to weight each measurement.

**Main method:** `CPWeightedFusionTracker` in `Code/two_speaker_tracking/`.
The fusion weight is area-only, `w = exp(-gamma * A)`, with `A` the normalized CP
region area. `CPWeightedFusionTracker.for_mondrian()` uses gamma = 32 and is fed
Mondrian-hatD (M = 5) CP regions. `for_global()` uses gamma = 16 with Global CP regions.
The tracker takes whichever CP regions it is given.

## Layout

| Path | Contents |
|---|---|
| `Code/two_speaker_tracking/` | Tracker, CP weight/features, association, LCP, npz adapter, metrics |
| `examples/` | Small runnable tracker demos |
| `analysis/mondrian_separation/` | Angular-separation (D) analysis and Mondrian-hatD experiments; scripts import each other, so keep them together |
| `analysis/lcp_experiments/` | LCP feature-selection experiments |
| `old/` | Superseded evaluation scripts (older tracker, baseline heatmap eval) |
| `maya_notes.yaml` | Working notes |

## Dependencies not included

This repo builds on an external CP framework (`Code.crc_ssl`, `Code.utilities`,
`Code.plots`, ...) that is not part of it. Place that framework's `Code/` files next to
`Code/two_speaker_tracking/` to run anything that imports it. Input heatmaps are the
`speakers_2_flat.npz` files exported by the SRP-DNN fork.

Scripts add the repo root to `sys.path`, so run them from anywhere, e.g.
`python analysis/mondrian_separation/eval_angular_separation.py`.
