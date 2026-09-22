"""
Smoke test for TwoSpeakerTracker with CP-weighted update.

Two synthetic speakers move in opposite directions.
CP regions are small (confident) in early frames and grow large (uncertain)
in later frames for speaker 1 — this should pull its track toward the
prediction instead of following the measurement blindly.

Run from repo root:
    python examples/run_two_speaker_tracker_skeleton.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from Code.two_speaker_tracking import TwoSpeakerTracker, smooth_two_speaker_tracks

# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------
T            = 5
nele         = 10
nazi         = 20
rng          = np.random.default_rng(seed=0)

# ------------------------------------------------------------------
# Estimated positions: two speakers drifting in opposite directions
# ------------------------------------------------------------------
# Speaker 0: moves right  (+azimuth)
# Speaker 1: moves left   (-azimuth)
base_0  = np.array([2.0, 4.0])   # [elevation_idx, azimuth_idx] (float, not int)
base_1  = np.array([6.0, 15.0])
drift_0 = np.array([0.0,  0.5])
drift_1 = np.array([0.0, -0.5])

estimated_positions = np.stack(
    [np.stack([base_0 + t * drift_0, base_1 + t * drift_1]) for t in range(T)]
)  # (T, 2, 2)

# ------------------------------------------------------------------
# CP regions:
#   Frames 0-2 → small regions (~5 % of grid cells) for both speakers.
#   Frames 3-4 → large region  (~60 % of grid cells) for speaker 1 only.
#
# measurement_uncertainty = cp_area_norm, so:
#   small region → u ≈ 0.05 → track ≈ measurement
#   large region → u ≈ 0.60 → track blends strongly toward prediction
# ------------------------------------------------------------------
cp_regions = np.zeros((T, 2, nele, nazi), dtype=np.float32)
for t in range(T):
    for k in range(2):
        fraction = 0.05 if (t < 3 or k == 0) else 0.60
        cp_regions[t, k] = (rng.random((nele, nazi)) < fraction).astype(np.float32)

# ------------------------------------------------------------------
# Likelihood maps: uniform random (content doesn't affect u for this test)
# ------------------------------------------------------------------
likelihood_maps = rng.random((T, 2, nele, nazi)).astype(np.float32)

# ------------------------------------------------------------------
# Run tracker
# ------------------------------------------------------------------
tracker = TwoSpeakerTracker(n_speakers=2)
out     = tracker.run(likelihood_maps, cp_regions, estimated_positions)

# ------------------------------------------------------------------
# Report shapes
# ------------------------------------------------------------------
print("=" * 60)
print("Output keys:", list(out.keys()))
print(f"tracks        : list of {len(out['tracks'])} arrays, "
      f"each shape {out['tracks'][0].shape}")
print(f"assignments   : shape {out['assignments'].shape}")
print(f"cp_features   : list of {len(out['cp_features'])} frames × "
      f"{len(out['cp_features'][0])} speakers")

# ------------------------------------------------------------------
# CP features: first vs. last frame
# ------------------------------------------------------------------
def _show_cp(frame_idx, speaker_idx):
    f = out["cp_features"][frame_idx][speaker_idx]
    u = f["measurement_uncertainty"]
    area = f["cp_area_norm"]
    print(f"    frame {frame_idx}, speaker {speaker_idx}: "
          f"cp_area_norm={area:.3f}  measurement_uncertainty={u:.3f}")

print("\nCP features (area_norm and uncertainty):")
for t in [0, T - 1]:
    for k in range(2):
        _show_cp(t, k)

# ------------------------------------------------------------------
# Track vs. measurement comparison
# ------------------------------------------------------------------
print("\nFrame-by-frame: measurement vs. tracked position for speaker 1")
print(f"  {'t':>2}  {'meas_az':>8}  {'track_az':>9}  {'u':>6}")
for t in range(T):
    meas_az  = estimated_positions[t, 1, 1]
    track_az = out["tracks"][1][t, 1]
    u        = out["cp_features"][t][1]["measurement_uncertainty"]
    print(f"  {t:>2}  {meas_az:>8.3f}  {track_az:>9.3f}  {u:>6.3f}")

# ------------------------------------------------------------------
# First / last positions
# ------------------------------------------------------------------
print("\nFirst frame tracked positions:")
for k in range(2):
    print(f"  speaker {k}: {out['tracks'][k][0]}")

print("\nLast frame tracked positions:")
for k in range(2):
    print(f"  speaker {k}: {out['tracks'][k][-1]}")

print("\nSpeaker 1, last frame:")
t_last = T - 1
meas  = estimated_positions[t_last, 1]
track = out["tracks"][1][t_last]
u     = out["cp_features"][t_last][1]["measurement_uncertainty"]
print(f"  measurement      : {meas}")
print(f"  tracked position : {track}")
print(f"  u (uncertainty)  : {u:.3f}")
if u > 0.1:
    diff = np.abs(track - meas)
    print(f"  |track - meas|   : {diff}  (non-zero → blend is active)")

# ------------------------------------------------------------------
# Smoother passthrough (still placeholder)
# ------------------------------------------------------------------
smoothed = smooth_two_speaker_tracks(
    tracks=out["tracks"],
    assignments=out["assignments"],
    cp_features_sequence=out["cp_features"],
)
print("\nSmoother output shapes (passthrough — identical to tracker output):")
for k, s in enumerate(smoothed):
    print(f"  speaker {k}: {s.shape}")

print("\nSmoke test passed.")
