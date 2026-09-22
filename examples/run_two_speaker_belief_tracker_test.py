"""
Belief-map tracker diagnostic: noisy likelihood spike + high CP uncertainty.

Tests that the CP-modulated Gaussian prior suppresses a noise spike when
measurement_uncertainty is high.

Setup
-----
- Two speakers move smoothly across the azimuth axis.
- At NOISY_FRAME=5, a Gaussian noise spike is injected into speaker 0's
  likelihood map at a location far from the true trajectory.
  Simultaneously, the CP region for speaker 0 is made large (u ≈ 0.55),
  signalling high uncertainty.

Expected behaviour
------------------
Confident frames (u ≈ 0.05, σ ≈ 2.93):
    wide prior → posterior ≈ likelihood → track follows true peak

Noisy frame    (u ≈ 0.55, σ ≈ 1.62):
    narrower prior centred at previous position → prior suppresses the
    distant noise spike → tracked position stays near the previous
    trajectory despite the corrupted likelihood map

Run from repo root:
    python examples/run_two_speaker_belief_tracker_test.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from Code.two_speaker_tracking import TwoSpeakerTracker
from Code.two_speaker_tracking.utils import build_motion_prior

# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------
T    = 10
nele = 18
nazi = 36
rng  = np.random.default_rng(seed=5)

# ------------------------------------------------------------------
# True physical trajectories (grid-index space)
#   Speaker 0: elevation 4 (constant), azimuth  8 → 18
#   Speaker 1: elevation 12 (constant), azimuth 26 → 16
# ------------------------------------------------------------------
el0, el1 = 4.0, 12.0
az0_true = np.linspace(8.0,  18.0, T)
az1_true = np.linspace(26.0, 16.0, T)

def _gaussian_map(el_c, az_c, nele, nazi, sigma=1.5):
    """Gaussian likelihood map centred at (el_c, az_c)."""
    el_g = np.arange(nele, dtype=float)
    az_g = np.arange(nazi, dtype=float)
    EL, AZ = np.meshgrid(el_g, az_g, indexing="ij")
    return np.exp(-0.5 * ((EL - el_c) ** 2 + (AZ - az_c) ** 2) / sigma ** 2).astype(
        np.float32
    )

# ------------------------------------------------------------------
# Build clean likelihood maps and true estimated_positions
# ------------------------------------------------------------------
likelihood_maps      = np.zeros((T, 2, nele, nazi), dtype=np.float32)
estimated_positions  = np.zeros((T, 2, 2), dtype=float)

for t in range(T):
    likelihood_maps[t, 0]     = _gaussian_map(el0, az0_true[t], nele, nazi)
    likelihood_maps[t, 1]     = _gaussian_map(el1, az1_true[t], nele, nazi)
    estimated_positions[t, 0] = [el0, az0_true[t]]
    estimated_positions[t, 1] = [el1, az1_true[t]]

# ------------------------------------------------------------------
# Inject noise spike at NOISY_FRAME for speaker 0
#   True  az at frame 5: az0_true[5] ≈ 13.6
#   Noise az:            30.0  (≈ 16 cells away)
# ------------------------------------------------------------------
NOISY_FRAME = 5
NOISE_AZ    = 30.0

likelihood_maps[NOISY_FRAME, 0]     = _gaussian_map(el0, NOISE_AZ, nele, nazi)
estimated_positions[NOISY_FRAME, 0] = [el0, NOISE_AZ]   # detector is fooled

# ------------------------------------------------------------------
# CP regions:
#   Normal frames: ~5 % active cells → measurement_uncertainty ≈ 0.05
#   Noisy frame, speaker 0: ~55 % active cells → u ≈ 0.55
#     → sigma ≈ 3.0 - 0.55 × 2.5 = 1.625  (narrower prior, more stabilisation)
# ------------------------------------------------------------------
cp_regions = (rng.random((T, 2, nele, nazi)) < 0.05).astype(np.float32)
cp_regions[NOISY_FRAME, 0] = (rng.random((nele, nazi)) < 0.55).astype(np.float32)

# ------------------------------------------------------------------
# Run tracker
# ------------------------------------------------------------------
tracker = TwoSpeakerTracker(n_speakers=2)
out     = tracker.run(likelihood_maps, cp_regions, estimated_positions)

# ------------------------------------------------------------------
# Per-frame table: speaker 0
# ------------------------------------------------------------------
sigma_min, sigma_max = 0.5, 3.0   # must match TwoSpeakerTracker._SIGMA_MIN/MAX

print("=" * 80)
print("Belief-map tracker — speaker 0 (noisy frame test)")
print(f"  Noise spike injected at t={NOISY_FRAME}: az={NOISE_AZ:.0f}"
      f"  (true az ≈ {az0_true[NOISY_FRAME]:.2f})")
print()
print(f"{'t':>2}  {'true_az':>8}  {'meas_az':>8}  {'tracked_az':>10}  "
      f"{'u':>5}  {'sigma':>5}  {'|trk-true|':>10}  {'|trk-meas|':>10}  status")
print("-" * 80)

for t in range(T):
    true_az    = az0_true[t]
    meas_az    = estimated_positions[t, 0, 1]
    tracked_az = out["tracks"][0][t, 1]
    u          = out["cp_features"][t][0]["measurement_uncertainty"]
    sigma      = sigma_max - float(np.clip(u, 0, 1)) * (sigma_max - sigma_min)

    d_true = abs(tracked_az - true_az)
    d_meas = abs(tracked_az - meas_az)

    if t == NOISY_FRAME:
        if d_true < d_meas:
            status = "<-- NOISY  [STABILIZED]"
        else:
            status = "<-- NOISY  [not stabilized — check sigma]"
    else:
        status = ""

    print(f"{t:>2}  {true_az:>8.2f}  {meas_az:>8.2f}  {tracked_az:>10.2f}  "
          f"{u:>5.3f}  {sigma:>5.2f}  {d_true:>10.3f}  {d_meas:>10.3f}  {status}")

print("-" * 80)

# ------------------------------------------------------------------
# Noisy frame close-up
# ------------------------------------------------------------------
t          = NOISY_FRAME
true_az    = az0_true[t]
meas_az    = estimated_positions[t, 0, 1]
tracked_az = out["tracks"][0][t, 1]
u          = out["cp_features"][t][0]["measurement_uncertainty"]
sigma      = sigma_max - float(np.clip(u, 0, 1)) * (sigma_max - sigma_min)

print(f"\nNoisy frame detail (t={t}):")
print(f"  true speaker az       : {true_az:.3f}")
print(f"  noise spike az        : {meas_az:.3f}  ({abs(meas_az - true_az):.1f} grid cells from true)")
print(f"  measurement_uncertainty u: {u:.3f}")
print(f"  prior sigma            : {sigma:.3f}")
print(f"  tracked az (argmax posterior): {tracked_az:.3f}")
print(f"  |tracked − true|      : {abs(tracked_az - true_az):.3f}")
print(f"  |tracked − noise|     : {abs(tracked_az - meas_az):.3f}")

if abs(tracked_az - true_az) < abs(tracked_az - meas_az):
    print("  PASS — tracker resisted noise spike (closer to true trajectory).")
else:
    print("  NOTE — tracker followed noise spike (try higher u or narrower sigma).")

# ------------------------------------------------------------------
# Output shape summary
# ------------------------------------------------------------------
print(f"\nOutput shapes:")
print(f"  tracks[0]       : {out['tracks'][0].shape}")
print(f"  assignments     : {out['assignments'].shape}")
print(f"  posterior_maps  : {out['posterior_maps'].shape}")

# ------------------------------------------------------------------
# Optional debug figure: likelihood | prior | posterior at noisy frame
# ------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_dbg    = NOISY_FRAME
    k_dbg    = 0
    prev_pos = out["tracks"][0][t_dbg - 1]    # position just before noisy frame

    u_dbg    = out["cp_features"][t_dbg][k_dbg]["measurement_uncertainty"]
    sig_dbg  = sigma_max - float(np.clip(u_dbg, 0, 1)) * (sigma_max - sigma_min)
    prior    = build_motion_prior(prev_pos, (nele, nazi),
                                  sigma_el=sig_dbg, sigma_az=sig_dbg)
    lmap     = likelihood_maps[t_dbg, k_dbg].astype(float)
    post     = out["posterior_maps"][t_dbg, k_dbg]
    tracked  = out["tracks"][0][t_dbg]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"t={t_dbg}  noise spike at az={NOISE_AZ:.0f}  "
                 f"u={u_dbg:.3f}  σ={sig_dbg:.2f}", fontsize=11)

    titles = [
        f"Likelihood\n(noise peak az={NOISE_AZ:.0f})",
        f"Motion prior\n(σ={sig_dbg:.2f}, centred az={prev_pos[1]:.1f})",
        f"Posterior\n(tracked az={tracked[1]:.1f})",
    ]
    maps = [lmap, prior, post]

    for ax, m, title in zip(axes, maps, titles):
        ax.imshow(m, origin="lower", aspect="auto", cmap="hot")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("azimuth index")
        ax.set_ylabel("elevation index")

    # Mark tracked position on the posterior panel.
    axes[2].plot(tracked[1], tracked[0], "c+", ms=14, mew=2, label="argmax")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__),
                            "debug_belief_tracker_noisy_frame.png")
    plt.savefig(fig_path, dpi=100)
    print(f"\nDebug figure saved: {fig_path}")
    plt.close()

except ImportError:
    print("\n(matplotlib not available — skipping debug figure)")
except Exception as exc:
    print(f"\n(Debug figure failed: {exc})")

print("\nBelief-map tracker test complete.")
