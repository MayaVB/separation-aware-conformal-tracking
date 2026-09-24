"""
Per-frame heatmap localization evaluation — no-burst (clean) scenario.
Each frame is evaluated independently; no tracking involved.
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

from Code.two_speaker_tracking.eval_metrics import (
    wrap_azi_err_deg, argmax_to_angles, hungarian_errors, compute_metrics,
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--npz_path", default="data/npz_output_tracking_no_burst_calib/Reverb_400_ms_SNR_15_dB/speakers_2_flat.npz")
args = parser.parse_args()
NPZ_PATH = args.npz_path

data = np.load(NPZ_PATH, allow_pickle=True)

lm          = data['all_likelihood_maps']        # (N_flat, K, nele, nazi)
speaker_pos = data['speaker_pos']                # (N_flat, K, 2)  [ele, azi] rad
gt_valid    = data['gt_valid_per_frame']         # (N_flat,) bool
est_srpdnn  = data['all_estimated_positions']    # (N_flat, K, 2)  [ele, azi] rad
sample_idx  = data['sample_idx_per_frame']       # (N_flat,) int
frame_idx   = data['frame_idx_per_frame']        # (N_flat,) int
nele        = int(data['nele'])
nazi        = int(data['nazi'])
reverb      = float(data['reverb'])
snr         = float(data['snr'])

K      = lm.shape[1]
N_flat = lm.shape[0]

ele_grid = np.linspace(0,      np.pi, nele)
azi_grid = np.linspace(-np.pi, np.pi, nazi)

# ---------------------------------------------------------------------------
# Per-frame evaluation
# ---------------------------------------------------------------------------
valid_frames = np.where(gt_valid)[0]

hm_azi, hm_ele, hm_mat = [], [], []
hm_gt_u, hm_est_u      = [], []

sp_azi, sp_ele, sp_mat = [], [], []
sp_gt_u, sp_est_u      = [], []

# For per-trajectory breakdown
traj_hm_azi  = {}   # sample_idx → list of azi errors
traj_sp_azi  = {}

for f in valid_frames:
    gt_ele = speaker_pos[f, :, 0]
    gt_azi = speaker_pos[f, :, 1]

    # --- Heatmap argmax ---
    est_ele_hm, est_azi_hm = argmax_to_angles(lm[f], ele_grid, azi_grid)
    ae, ee, mat, gtu, estu = hungarian_errors(est_ele_hm, est_azi_hm, gt_ele, gt_azi)
    hm_azi.extend(ae);  hm_ele.extend(ee);  hm_mat.extend(mat)
    hm_gt_u.extend(gtu.tolist());  hm_est_u.extend(estu.tolist())

    sid = int(sample_idx[f])
    traj_hm_azi.setdefault(sid, []).extend(ae)

    # --- SRPDNN baseline ---
    est_ele_sp = est_srpdnn[f, :, 0]
    est_azi_sp = est_srpdnn[f, :, 1]
    ae, ee, mat, gtu, estu = hungarian_errors(est_ele_sp, est_azi_sp, gt_ele, gt_azi)
    sp_azi.extend(ae);  sp_ele.extend(ee);  sp_mat.extend(mat)
    sp_gt_u.extend(gtu.tolist());  sp_est_u.extend(estu.tolist())

    traj_sp_azi.setdefault(sid, []).extend(ae)

# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------
hm_metrics = compute_metrics(hm_azi, hm_ele, hm_mat, hm_gt_u, hm_est_u)
sp_metrics = compute_metrics(sp_azi, sp_ele, sp_mat, sp_gt_u, sp_est_u)

N_traj = len(traj_hm_azi)

# ---------------------------------------------------------------------------
# Print + save summary table
# ---------------------------------------------------------------------------
reverb_ms  = int(round(reverb * 1000))
snr_int    = int(round(snr))
# derive a tag from the npz directory name so no-burst and burst outputs don't collide
npz_tag    = NPZ_PATH.rstrip("/").split("/")[-3]   # e.g. "npz_output_tracking_burst_test"
txt_path   = (f"results_heatmap_eval_{npz_tag}"
              f"_Reverb_{reverb_ms}_ms_SNR_{snr_int}_dB_speakers{K}.txt")

title  = (f"Evaluation: T60={reverb:.1f}s  SNR={snr:.0f}dB  "
          f"N_frames={N_flat}  N_valid={len(valid_frames)}  N_trajectories={N_traj}")
header = (f"{'':22s}  {'MAE_azi':>8s}  {'MAE_ele':>8s}  {'RMSE_azi':>9s}  "
          f"{'RMSE_ele':>9s}  {'ACC@30%':>8s}  {'MDR':>6s}  {'FAR':>6s}")
sep    = "-" * len(header)

def fmt_row(label, m):
    return (f"{label:<22s}  {m['MAE_azi']:>7.1f}   {m['MAE_ele']:>7.1f}   "
            f"{m['RMSE_azi']:>8.1f}   {m['RMSE_ele']:>8.1f}   "
            f"{m['ACC30']:>8.3f}  {m['MDR']:>6.3f}  {m['FAR']:>6.3f}")

rows   = [fmt_row("Heatmap argmax:", hm_metrics),
          fmt_row("SRPDNN baseline:", sp_metrics)]
output = "\n".join([title, "", header, sep] + rows + [""])

print("\n" + output)

with open(txt_path, "w") as fh:
    fh.write(output + "\n")
print(f"Results saved -> {txt_path}")

# ---------------------------------------------------------------------------
# Peak-stealing analysis (burst frames only)
# ---------------------------------------------------------------------------
if 'burst_active_per_frame' in data:
    burst_active   = data['burst_active_per_frame']    # (N_flat,) bool
    burst_grid_idx = data['burst_grid_idx_per_frame']  # (N_flat, 8, 2)

    burst_valid = np.where(gt_valid & burst_active.astype(bool))[0]
    n_bv = len(burst_valid)

    if n_bv == 0:
        print("\n[Peak-stealing] No burst-active valid frames in this dataset.")
    else:
        stolen = np.zeros(K, dtype=int)
        total  = np.zeros(K, dtype=int)

        for f in burst_valid:
            burst_cells = [(int(ei), int(ai))
                           for ei, ai in burst_grid_idx[f] if ei >= 0 and ai >= 0]
            if not burst_cells:
                continue

            # Hungarian-match estimated speakers to GT sources (same as localization eval)
            gt_ele_f = speaker_pos[f, :, 0]
            gt_azi_f = speaker_pos[f, :, 1]
            est_ele_f, est_azi_f = argmax_to_angles(lm[f], ele_grid, azi_grid)
            cost = np.array([[wrap_azi_err_deg(est_azi_f[e], gt_azi_f[g])
                              for g in range(K)] for e in range(K)])
            est_idx, gt_idx = linear_sum_assignment(cost)

            for e, g in zip(est_idx, gt_idx):
                true_ei = np.argmin(np.abs(speaker_pos[f, g, 0] - ele_grid))
                true_ai = np.argmin(np.abs(speaker_pos[f, g, 1] - azi_grid))
                lm_true  = lm[f, e, true_ei, true_ai]
                lm_burst = max(lm[f, e, bei, bai] for bei, bai in burst_cells)
                total[e] += 1
                if lm_burst > lm_true:
                    stolen[e] += 1

        hdr = f"\nPeak-stealing analysis  ({n_bv} burst-active valid frames / {N_flat} total)"
        print(hdr)
        print("-" * len(hdr.strip()))
        print(f"  {'Speaker':<12s}  {'Stolen':>8s}  {'Total':>8s}  {'Rate':>8s}")
        for k in range(K):
            rate = stolen[k] / total[k] if total[k] else float('nan')
            print(f"  Speaker {k+1:<4d}   {stolen[k]:>8d}  {total[k]:>8d}  {rate:>7.1%}")
        ovr_rate = stolen.sum() / total.sum() if total.sum() else float('nan')
        print(f"  {'Overall':<12s}  {stolen.sum():>8d}  {total.sum():>8d}  {ovr_rate:>7.1%}")

# ---------------------------------------------------------------------------
# Per-trajectory MAE_azi distribution
# ---------------------------------------------------------------------------
traj_ids   = sorted(traj_hm_azi.keys())
hm_traj_mae  = np.array([np.mean(traj_hm_azi[t])  for t in traj_ids])
sp_traj_mae  = np.array([np.mean(traj_sp_azi[t])   for t in traj_ids])

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

axes[0].hist(hm_traj_mae, bins=20, edgecolor='black', alpha=0.75, color='steelblue')
axes[0].axvline(np.mean(hm_traj_mae), color='red', linestyle='--',
                label=f'mean={np.mean(hm_traj_mae):.1f}°')
axes[0].set_title('Heatmap argmax — per-trajectory MAE_azi')
axes[0].set_xlabel('MAE azimuth (°)')
axes[0].set_ylabel('Trajectory count')
axes[0].legend()

axes[1].hist(sp_traj_mae, bins=20, edgecolor='black', alpha=0.75, color='darkorange')
axes[1].axvline(np.mean(sp_traj_mae), color='red', linestyle='--',
                label=f'mean={np.mean(sp_traj_mae):.1f}°')
axes[1].set_title('SRPDNN baseline — per-trajectory MAE_azi')
axes[1].set_xlabel('MAE azimuth (°)')
axes[1].set_ylabel('Trajectory count')
axes[1].legend()

plt.suptitle(f'Per-trajectory MAE_azi distribution  '
             f'(T60={reverb:.1f}s, SNR={snr:.0f}dB, {N_traj} trajectories)',
             fontsize=11)
plt.tight_layout()

out_fig = "heatmap_eval_noburst_traj_distribution.png"
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
print(f"Trajectory distribution plot saved → {out_fig}")
plt.show()
