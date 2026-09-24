"""
Small deterministic tests for CPWeightedFusionTracker association / periodic
azimuth / track-ordered diagnostics / persistent-identity evaluation.

Synthetic data on the real SRP-DNN grid (37 x 73; ele = polar angle in
[0, pi], azi in [-pi, pi] with duplicated +-pi column). Each IDL slot's
"measurement package" is: estimated DOA (grid idx), a single-peak likelihood
map, and a CP region (disk of a slot-specific radius around the estimate).

Run from repo root (pytest not required):
    python3 tests/test_cp_weighted_tracker_association.py
Also collectable by pytest if installed.

Test E (real crossing) is a REPORT, not a pass/fail requirement -- the
tracker must not be modified just to make it pass.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Code.two_speaker_tracking.cp_weighted_tracker import CPWeightedFusionTracker
from Code.two_speaker_tracking.association import associate_two_speakers
from Code.two_speaker_tracking.utils import (
    grid_index_to_angles, great_circle_deg, great_circle_distance_grid,
    fold_azimuth_endpoint, unfold_azimuth_endpoint, propagate_belief_periodic_azimuth,
)
from Code.two_speaker_tracking.eval_metrics import persistent_identity_eval

NELE, NAZI = 37, 73
GRID = (NELE, NAZI)
EL_IDX, AZ_IDX = np.meshgrid(np.arange(NELE), np.arange(NAZI), indexing="ij")
GRID_ELE, GRID_AZI = grid_index_to_angles(np.stack([EL_IDX, AZ_IDX], -1), GRID)


# ---------------------------------------------------------------------------
# Synthetic measurement packages
# ---------------------------------------------------------------------------

def doa_to_grid(ele_deg, azi_deg):
    """Nearest grid cell of a DOA; azimuth reported in [0, NAZI-2] (+-180 -> col 0)."""
    ele_i = int(round(ele_deg / 180.0 * (NELE - 1)))
    azi_wrapped = (azi_deg + 180.0) % 360.0 - 180.0
    azi_i = int(round((azi_wrapped + 180.0) / 360.0 * (NAZI - 1))) % (NAZI - 1)
    return np.array([ele_i, azi_i], dtype=float)


def gc_map_deg(ele_deg, azi_deg):
    """(NELE, NAZI) great-circle distance of every grid cell to a DOA."""
    return great_circle_deg(GRID_ELE, GRID_AZI, np.radians(ele_deg), np.radians(azi_deg))


def make_package(ele_deg, azi_deg, peak_width_deg=8.0, region_radius_deg=10.0):
    """One IDL slot package: (grid DOA, likelihood map, CP region)."""
    d = gc_map_deg(ele_deg, azi_deg)
    lmap = np.exp(-0.5 * (d / peak_width_deg) ** 2) + 1e-3
    region = (d <= region_radius_deg).astype(float)
    return doa_to_grid(ele_deg, azi_deg), lmap, region


def build_sequence(src0, src1, slot_order, radius0=10.0, radius1=30.0, noise_deg=0.0, seed=0):
    """src0/src1: (T, 2) true [ele_deg, azi_deg]. slot_order[t] = 0 -> slot0 holds
    source 0; 1 -> slot0 holds source 1 (the WHOLE package moves). Source 0 always
    gets CP radius radius0 and source 1 radius1, so A tells the sources apart."""
    rng = np.random.default_rng(seed)
    T = len(src0)
    lm = np.zeros((T, 2, NELE, NAZI))
    cp = np.zeros((T, 2, NELE, NAZI))
    est = np.zeros((T, 2, 2))
    for t in range(T):
        pk = []
        for (e, a), r in ((src0[t], radius0), (src1[t], radius1)):
            n = rng.normal(0.0, noise_deg, 2) if noise_deg > 0 else np.zeros(2)
            pk.append(make_package(e + n[0], a + n[1], region_radius_deg=r))
        if slot_order[t] == 1:
            pk = pk[::-1]
        for m in range(2):
            est[t, m], lm[t, m], cp[t, m] = pk[m]
    return lm, cp, est


def tracks_rad(result):
    tr = np.stack(result["tracks"], axis=1)  # (T, 2, 2) grid idx
    ele, azi = grid_index_to_angles(tr, GRID)
    return np.stack([ele, azi], axis=-1)


def gt_rad(src0, src1):
    return np.radians(np.stack([src0, src1], axis=1))  # (T, 2, 2)


# ---------------------------------------------------------------------------
# A. IDL slot swap -- complete package follows association
# ---------------------------------------------------------------------------

def test_A_slot_swap_package_follows():
    T = 20
    src0 = np.stack([np.full(T, 60.0), np.linspace(-60, -40, T)], 1)
    src1 = np.stack([np.full(T, 110.0), np.linspace(60, 40, T)], 1)
    slot_order = np.array([0, 1] * (T // 2))  # swap EVERY frame
    lm, cp, est = build_sequence(src0, src1, slot_order)
    res = CPWeightedFusionTracker().run(lm, cp, est)

    # Frame 0 identity: track k <- slot k, i.e. track 0 = source 0.
    # Afterwards, track 0 must always receive whichever slot holds source 0.
    expected_slot_track0 = slot_order  # slot index holding source 0
    assert np.array_equal(res["track_assigned_slot"][:, 0], expected_slot_track0), res["track_assigned_slot"][:, 0]
    assert np.array_equal(res["track_assigned_slot"][:, 1], 1 - expected_slot_track0)

    # Package-level: track 0's A must be source 0's (small radius) at EVERY frame,
    # even though slot 0 alternates between the small and the large region.
    A0, A1 = res["track_A"][:, 0], res["track_A"][:, 1]
    assert np.all(A0 == A0[0]) or np.ptp(A0) < 0.3 * A0.mean(), A0  # constant up to pole geometry
    assert np.all(A1 > 3 * A0), (A0, A1)
    # slot-ordered features DO alternate (proves the naming distinction matters)
    slotA0 = np.array([f[0]["cp_area_norm"] for f in res["slot_cp_features"]])
    assert slotA0[0] < slotA0[1] and slotA0[2] < slotA0[1]
    # w follows the package: track 0 (compact region) trusts its measurement more
    assert np.all(res["track_w"][1:, 0] > res["track_w"][1:, 1])

    ev = persistent_identity_eval(tracks_rad(res), gt_rad(src0, src1))
    assert ev["initial_mapping"] == (0, 1) and ev["id_switches"] == 0
    assert np.nanmax(ev["persistent_err"]) < 6.0, np.nanmax(ev["persistent_err"])
    return dict(max_persistent_err=float(np.nanmax(ev["persistent_err"])),
                id_switches=ev["id_switches"], A_track0=float(A0.mean()), A_track1=float(A1.mean()),
                w_track0=float(np.nanmean(res["track_w"][:, 0])), w_track1=float(np.nanmean(res["track_w"][:, 1])))


# ---------------------------------------------------------------------------
# B. azimuth boundary in association
# ---------------------------------------------------------------------------

def test_B_association_across_pm180():
    # Grid step is 5 deg, so "+179 / -179" is realised as +175 (col 71) / -175 (col 1).
    track0_pred = doa_to_grid(90, 175)
    track1_pred = doa_to_grid(90, 0)
    meas = np.stack([doa_to_grid(90, 5), doa_to_grid(90, -175)])  # slot0 near track1, slot1 near track0
    gc = great_circle_distance_grid(track0_pred, meas[1], GRID)
    assert abs(gc - 10.0) < 1e-9, gc

    tracks = [{"position": track0_pred}, {"position": track1_pred}]
    a, costs = associate_two_speakers(tracks, meas, grid_shape=GRID, return_costs=True)
    assert a == [1, 0], a
    assert costs["cost_swapped"] < costs["cost_identity"]
    legacy = associate_two_speakers(tracks, meas)  # Euclidean grid-index, as before the fix
    return dict(gc_deg_175_to_m175=gc, assignment=a, cost_identity=costs["cost_identity"],
                cost_swapped=costs["cost_swapped"], legacy_euclidean_assignment=legacy)


def test_B2_tracker_moves_through_pm180():
    T = 21
    src0 = np.stack([np.full(T, 80.0), np.linspace(160, 200, T)], 1)  # crosses +-180
    src1 = np.stack([np.full(T, 100.0), np.full(T, 0.0)], 1)
    slot_order = np.array([0, 1] * 10 + [0])
    lm, cp, est = build_sequence(src0, src1, slot_order)
    res = CPWeightedFusionTracker().run(lm, cp, est)
    ev = persistent_identity_eval(tracks_rad(res), gt_rad(src0, src1))
    assert ev["id_switches"] == 0 and np.nanmax(ev["persistent_err"]) < 6.0, ev
    return dict(max_persistent_err=float(np.nanmax(ev["persistent_err"])), id_switches=ev["id_switches"])


# ---------------------------------------------------------------------------
# C. propagation across the +-180 edge
# ---------------------------------------------------------------------------

def test_C_propagation_across_pm180():
    b = np.zeros(GRID)
    b[18, 71] = 1.0  # ele 90, azi +175
    bf = fold_azimuth_endpoint(b, kind="mass")
    assert np.allclose(unfold_azimuth_endpoint(bf).sum(), 1.0)
    assert np.allclose(fold_azimuth_endpoint(unfold_azimuth_endpoint(bf), "mass"), bf)  # exact round trip

    pf = propagate_belief_periodic_azimuth(bf, sigma_el=2.0, sigma_az=2.0)
    p = unfold_azimuth_endpoint(pf)
    row = p[18]
    # +175 is 1 cell from 170 (col 70) and 1 cell from +-180 (folded col 0);
    # 2 cells from 165 (col 69) and from -175 (col 1).
    assert row[1] > 0 and row[2] > 0, "mass did not cross +180 -> -180"
    assert np.isclose(pf[18, 69], pf[18, 1]), (pf[18, 69], pf[18, 1])
    assert np.isclose(pf[18, 70], pf[18, 0])
    assert np.isclose(p.sum(), 1.0)
    # elevation does NOT wrap: a mass at the north pole row loses mass (then renormalised),
    # it never appears at the south pole row.
    bp = np.zeros(GRID); bp[0, 36] = 1.0
    pp = propagate_belief_periodic_azimuth(fold_azimuth_endpoint(bp, "mass"), 2.0, 2.0)
    assert pp[-1].sum() == 0.0

    from scipy.ndimage import gaussian_filter
    old = gaussian_filter(b, sigma=(2.0, 2.0), mode="constant")
    old /= old.sum()
    return dict(mass_at_m175_new=float(row[1]), mass_at_m170_new=float(row[2]),
                mass_at_m175_old_constant_mode=float(old[18, 1]),
                mass_at_165_new=float(row[69]), total=float(p.sum()))


# ---------------------------------------------------------------------------
# D. ordinary non-crossing trajectory with noisy, randomly slot-ordered measurements
# ---------------------------------------------------------------------------

def test_D_ordinary_noncrossing():
    T = 60
    rng = np.random.default_rng(1)
    src0 = np.stack([np.linspace(70, 80, T), np.linspace(-90, -40, T)], 1)
    src1 = np.stack([np.linspace(110, 95, T), np.linspace(40, 100, T)], 1)
    slot_order = rng.integers(0, 2, T)  # ~50% random slot order, like real IDL output
    slot_order[0] = 0
    lm, cp, est = build_sequence(src0, src1, slot_order, noise_deg=3.0, seed=2)
    res = CPWeightedFusionTracker().run(lm, cp, est)
    ev = persistent_identity_eval(tracks_rad(res), gt_rad(src0, src1))

    raw_ele, raw_azi = grid_index_to_angles(est, GRID)
    ev_raw = persistent_identity_eval(np.stack([raw_ele, raw_azi], -1), gt_rad(src0, src1))
    assert ev["id_switches"] == 0, ev["id_switches"]
    assert np.nanmean(ev["persistent_err"]) < 6.0
    return dict(tracked_persistent_MAE=float(np.nanmean(ev["persistent_err"])),
                tracked_hungarian_MAE=float(np.nanmean(ev["hungarian_err"])),
                tracked_id_switches=ev["id_switches"],
                raw_slot_order_persistent_MAE=float(np.nanmean(ev_raw["persistent_err"])),
                raw_slot_order_hungarian_MAE=float(np.nanmean(ev_raw["hungarian_err"])),
                raw_slot_order_id_switches=ev_raw["id_switches"],
                slot_flips_in_input=int(np.sum(slot_order[1:] != slot_order[:-1])))


# ---------------------------------------------------------------------------
# E. real crossing -- REPORT ONLY
# ---------------------------------------------------------------------------

def _crossing_case(ele_offset_deg, noise_deg, seed):
    T = 41
    rng = np.random.default_rng(seed)
    src0 = np.stack([np.full(T, 90.0 - ele_offset_deg / 2), np.linspace(-40, 40, T)], 1)
    src1 = np.stack([np.full(T, 90.0 + ele_offset_deg / 2), np.linspace(40, -40, T)], 1)
    slot_order = rng.integers(0, 2, T); slot_order[0] = 0
    lm, cp, est = build_sequence(src0, src1, slot_order, noise_deg=noise_deg, seed=seed + 100)
    res = CPWeightedFusionTracker().run(lm, cp, est)
    ev = persistent_identity_eval(tracks_rad(res), gt_rad(src0, src1))
    pe = ev["persistent_err"]
    return dict(ele_offset_deg=ele_offset_deg, noise_deg=noise_deg,
                id_switches=ev["id_switches"],
                persistent_MAE_before=float(np.nanmean(pe[:18])),
                persistent_MAE_after=float(np.nanmean(pe[23:])),
                hungarian_MAE_after=float(np.nanmean(ev["hungarian_err"][23:])),
                min_assoc_gap_deg=float(np.nanmin(res["assoc_cost_gap_deg"])),
                track0_end_azi_deg=float(np.degrees(tracks_rad(res)[-1, 0, 1])),
                source0_end_azi_deg=40.0)


def test_E_real_crossing_report():
    cases = [_crossing_case(0.0, 0.0, 0), _crossing_case(0.0, 3.0, 1), _crossing_case(10.0, 0.0, 2)]
    for c in cases:  # only sanity: it runs and produces finite numbers
        assert np.isfinite(c["persistent_MAE_after"])
    return cases


# ---------------------------------------------------------------------------
# F. azimuth-periodic S / V descriptors
# ---------------------------------------------------------------------------

def _region(rows, cols):
    r = np.zeros(GRID)
    for i in rows:
        for j in cols:
            r[i, j] = 1.0
    return r


def _SV(region):
    from Code.two_speaker_tracking.cp_features import extract_cp_features
    from Code.two_speaker_tracking.cp_weight import compute_cp_weight
    f = extract_cp_features(region)
    _, d = compute_cp_weight(f, GRID)
    return f, d


def test_F_periodic_SV():
    rows = range(10, 15)
    away = _region(rows, range(34, 39))              # 5 azimuths centred on 0 deg
    cross = _region(rows, [70, 71, 72, 1, 2])         # 5 distinct azimuths centred on +-180 (col 72 == col 0)
    cross_dup = _region(rows, [70, 71, 72, 0, 1, 2])  # same directions, +-180 present in BOTH duplicate columns
    fa, da = _SV(away)
    fc, dc = _SV(cross)
    fd, dd = _SV(cross_dup)
    # 1+2: away-from-seam vs crossing: same periodic span/dispersion, same S/V
    assert fa["cp_width_az_periodic"] == fc["cp_width_az_periodic"] == fd["cp_width_az_periodic"] == 5
    assert np.isclose(fa["cp_var_periodic"], fc["cp_var_periodic"])
    assert np.isclose(fa["cp_var_periodic"], fd["cp_var_periodic"])
    assert np.isclose(da["span_norm"], dc["span_norm"]) and np.isclose(da["var_norm"], dc["var_norm"])
    assert np.isclose(da["span_norm"], dd["span_norm"]) and np.isclose(da["var_norm"], dd["var_norm"])
    # A is unchanged by design: equal for equal cell counts, one extra column for the duplicate
    assert da["size_norm"] == dc["size_norm"] and dd["size_norm"] > dc["size_norm"]
    # away from the seam, periodic == old definitions exactly
    assert fa["cp_width_az"] == fa["cp_width_az_periodic"] and np.isclose(fa["cp_var"], fa["cp_var_periodic"])
    # the old non-periodic definitions blow up for the crossing region
    assert fc["cp_width_az"] == 72 and fd["cp_width_az"] == 73  # raw index span 1..72 / 0..72
    return dict(S_away=da["span_norm"], S_cross=dc["span_norm"], S_cross_dup=dd["span_norm"],
                V_away=da["var_norm"], V_cross=dc["var_norm"], V_cross_dup=dd["var_norm"],
                old_width_az_cross=fc["cp_width_az"], old_var_cross=fc["cp_var"], old_var_away=fa["cp_var"])


def test_F2_periodic_SV_full_ring():
    """A band covering every azimuth keeps full width (72 distinct columns)."""
    f, _ = _SV(_region(range(0, 3), range(NAZI)))
    assert f["cp_width_az_periodic"] == NAZI - 1
    return dict(width_az_periodic=f["cp_width_az_periodic"], width_az_old=f["cp_width_az"])


if __name__ == "__main__":
    tests = [test_A_slot_swap_package_follows, test_B_association_across_pm180,
             test_B2_tracker_moves_through_pm180, test_C_propagation_across_pm180,
             test_D_ordinary_noncrossing, test_E_real_crossing_report,
             test_F_periodic_SV, test_F2_periodic_SV_full_ring]
    failed = 0
    for fn in tests:
        try:
            out = fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            out = None
            print(f"FAIL  {fn.__name__}: {e!r}")
        if out is not None:
            for row in (out if isinstance(out, list) else [out]):
                print("      " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                          for k, v in row.items()))
    sys.exit(1 if failed else 0)
