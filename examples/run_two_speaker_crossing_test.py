"""
Crossing-trajectory association diagnostic for TwoSpeakerTracker.

NOT a unit test — a diagnostic smoke test.

Two synthetic speakers move in opposite azimuth directions and cross each
other in the middle.  At frames 3–5, the order of the measurements inside
estimated_positions is intentionally swapped to simulate a detector that
outputs speaker slots in the wrong order (permutation ambiguity).

If association is correct, the tracker should flip its assignment to [1, 0]
at the swapped frames so that each track continues to follow the correct
physical speaker.

Design choices
--------------
- Elevations are well-separated (el0=3, el1=7) so the crossing in azimuth
  does not produce a genuinely ambiguous Euclidean distance — the elevation
  difference always keeps the two speakers distinguishable.
- CP regions are tiny (~5 % of cells) → measurement_uncertainty ≈ 0.05.
  This keeps sigma near sigma_max (wide prior), so the posterior ≈ likelihood
  and the tracked position follows the likelihood argmax closely.
- Likelihood maps are 2-D Gaussians centred at estimated_positions.
  At SWAP frames the two Gaussians are at the swapped (wrong-slot) positions;
  after association re-assigns them, each track receives the correct map.

Note: if both speakers shared the same elevation AND the azimuth values
coincided exactly, both permutations would have equal cost and the
assignment would be arbitrary (tie-broken by iteration order in
itertools.permutations).  That ambiguity is expected behaviour and is
handled separately once ambiguity detection is added to association.py.

Run from repo root:
    python examples/run_two_speaker_crossing_test.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from Code.two_speaker_tracking import TwoSpeakerTracker

# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------
T    = 8
nele = 10
nazi = 20
rng  = np.random.default_rng(seed=7)

# ------------------------------------------------------------------
# True physical trajectories
#   Speaker 0: elevation 3 (constant), azimuth 5 → 12
#   Speaker 1: elevation 7 (constant), azimuth 12 → 5
# The azimuths cross between frames 3 (az0=8, az1=9) and 4 (az0=9, az1=8).
# The 4-unit elevation gap makes the crossing resolvable by Euclidean distance.
# ------------------------------------------------------------------
el0, el1 = 3.0, 7.0
az0 = np.linspace(5.0, 12.0, T)   # [5, 6, 7, 8, 9, 10, 11, 12]
az1 = np.linspace(12.0, 5.0, T)   # [12, 11, 10, 9, 8, 7, 6, 5]

true_positions = np.stack([
    np.column_stack([np.full(T, el0), az0]),   # (T, 2) for speaker 0
    np.column_stack([np.full(T, el1), az1]),   # (T, 2) for speaker 1
], axis=1)  # shape (T, 2, 2)

# ------------------------------------------------------------------
# Introduce measurement-order swap at frames 3, 4, 5.
# At these frames, estimated_positions[:, 0] = physical speaker 1's position
#                  estimated_positions[:, 1] = physical speaker 0's position.
# This simulates a detector that outputs the two measurements in the wrong slot.
# Association should compensate by returning assignment [1, 0] at these frames.
# ------------------------------------------------------------------
SWAP_FRAMES = {3, 4, 5}
estimated_positions = true_positions.copy()
for t in SWAP_FRAMES:
    estimated_positions[t, 0] = true_positions[t, 1]  # slot 0 ← physical spk 1
    estimated_positions[t, 1] = true_positions[t, 0]  # slot 1 ← physical spk 0

# ------------------------------------------------------------------
# Small CP regions → measurement_uncertainty ≈ 0.05 (wide prior, follows lmap).
# Likelihood maps: 2-D Gaussians centred at each slot's estimated_position.
# At SWAP frames the two Gaussians are at the swapped (wrong-slot) locations;
# after association re-routes them, each track receives the correct map.
# ------------------------------------------------------------------
cp_regions = (rng.random((T, 2, nele, nazi)) < 0.05).astype(np.float32)

def _gmap(el_c, az_c, sigma=1.2):
    """Gaussian likelihood map centred at (el_c, az_c) in grid-cell coordinates."""
    el_g = np.arange(nele, dtype=float)
    az_g = np.arange(nazi, dtype=float)
    EL, AZ = np.meshgrid(el_g, az_g, indexing="ij")
    return np.exp(
        -0.5 * ((EL - el_c) ** 2 + (AZ - az_c) ** 2) / sigma ** 2
    ).astype(np.float32)

likelihood_maps = np.zeros((T, 2, nele, nazi), dtype=np.float32)
for t in range(T):
    for k in range(2):
        el_c, az_c = estimated_positions[t, k]
        likelihood_maps[t, k] = _gmap(el_c, az_c)

# ------------------------------------------------------------------
# Run tracker
# ------------------------------------------------------------------
tracker = TwoSpeakerTracker(n_speakers=2)
out     = tracker.run(likelihood_maps, cp_regions, estimated_positions)

# ------------------------------------------------------------------
# Reference table: true azimuth values
# ------------------------------------------------------------------
print("=" * 75)
print("Reference: true speaker azimuth trajectories")
print(f"  {'t':>2}  " + "  ".join(f"spk{k}_az" for k in range(2)))
for t in range(T):
    vals = "  ".join(f"{true_positions[t, k, 1]:>8.3f}" for k in range(2))
    print(f"  {t:>2}  {vals}")

# ------------------------------------------------------------------
# Per-frame association & tracking table
# ------------------------------------------------------------------
print()
print("=" * 75)
print("Per-frame: measurements → assignments → tracked positions")
print()
print(f"{'t':>2}  {'swap?':5}  "
      f"{'meas0_az':>8}  {'meas1_az':>8}  "
      f"{'assign':>6}  "
      f"{'trk0_az':>8}  {'trk1_az':>8}  "
      f"{'u0':>5}  {'u1':>5}  "
      f"{'note'}")
print("-" * 75)

for t in range(T):
    swap_tag  = "SWAP" if t in SWAP_FRAMES else ""
    meas0_az  = estimated_positions[t, 0, 1]
    meas1_az  = estimated_positions[t, 1, 1]
    assign    = [int(x) for x in out["assignments"][t]]
    trk0_az   = out["tracks"][0][t, 1]
    trk1_az   = out["tracks"][1][t, 1]
    u0        = out["cp_features"][t][0]["measurement_uncertainty"]
    u1        = out["cp_features"][t][1]["measurement_uncertainty"]

    # Expected assignment: [1, 0] at swap frames, [0, 1] otherwise (frame 0 forced [0,1]).
    if t == 0:
        expected = [0, 1]   # tracks are None at t=0 → identity fallback
    elif t in SWAP_FRAMES:
        expected = [1, 0]
    else:
        expected = [0, 1]

    match = "OK" if assign == expected else f"?? (expected {expected})"

    print(f"{t:>2}  {swap_tag:5}  "
          f"{meas0_az:>8.3f}  {meas1_az:>8.3f}  "
          f"{str(assign):>6}  "
          f"{trk0_az:>8.3f}  {trk1_az:>8.3f}  "
          f"{u0:>5.3f}  {u1:>5.3f}  "
          f"{match}")

print("-" * 75)

# ------------------------------------------------------------------
# Identity diagnostic
# For each frame, decide whether each track is following its correct speaker:
#   track k is "correct" if it is closer to true_positions[t, k] than to
#   the other speaker's true position.
#
# Ambiguity threshold: if the two distances differ by less than 0.05 grid
# cells, the frame is flagged as ambiguous rather than wrong.
# Ambiguity at the exact crossing (t=3.5 would be between t=3 and t=4) is
# expected when elevation separation is small; it should NOT occur here.
# ------------------------------------------------------------------
AMBIG_THRESH = 0.05

print()
print("=" * 75)
print("Identity diagnostic (2-D Euclidean distance to each true speaker)")
print()
print(f"{'t':>2}  {'d(trk0,spk0)':>13}  {'d(trk0,spk1)':>13}  "
      f"{'d(trk1,spk0)':>13}  {'d(trk1,spk1)':>13}  verdict")
print("-" * 75)

n_correct = 0
n_ambig   = 0
n_wrong   = 0

for t in range(T):
    trk0  = out["tracks"][0][t]
    trk1  = out["tracks"][1][t]
    true0 = true_positions[t, 0]
    true1 = true_positions[t, 1]

    d00 = float(np.linalg.norm(trk0 - true0))
    d01 = float(np.linalg.norm(trk0 - true1))
    d10 = float(np.linalg.norm(trk1 - true0))
    d11 = float(np.linalg.norm(trk1 - true1))

    trk0_correct = d00 < d01
    trk1_correct = d11 < d10
    gap0 = abs(d00 - d01)
    gap1 = abs(d11 - d10)

    if gap0 < AMBIG_THRESH or gap1 < AMBIG_THRESH:
        verdict = "AMBIGUOUS (expected near crossing)"
        n_ambig += 1
    elif trk0_correct and trk1_correct:
        verdict = "correct"
        n_correct += 1
    else:
        verdict = "IDENTITY SWITCH"
        n_wrong += 1

    print(f"{t:>2}  {d00:>13.4f}  {d01:>13.4f}  "
          f"{d10:>13.4f}  {d11:>13.4f}  {verdict}")

print("-" * 75)
print(f"\nIdentity summary: {n_correct} correct  |  {n_ambig} ambiguous  |  {n_wrong} switches")

if n_wrong == 0:
    print("PASS — no unexpected identity switches.")
    print("       (Ambiguous frames, if any, are expected at the crossing point.)")
else:
    print("WARN — unexpected identity switch(es) detected.")
    print("       Investigate: tracks may have drifted past the crossing without")
    print("       re-acquiring the correct speaker identity.")

print()
