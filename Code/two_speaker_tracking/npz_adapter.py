"""
Bridge between real SRP-DNN npz heatmap exports and TwoSpeakerTracker.

Nothing in this package previously converted a real (radians) npz export
into calibrated conformal-prediction regions and grid-cell-index positions
that TwoSpeakerTracker actually expects — every example script in
`examples/` uses synthetic data instead. This module is that bridge.

Coordinate convention (see the "fix the mismatch" note in tracker.py /
cp_features.py / association.py / utils.py): TwoSpeakerTracker and its
motion-prior math operate entirely in grid-cell indices [el_idx, az_idx],
NOT radians. Real npz data (`all_estimated_positions`, `speaker_pos`) is in
radians [elevation, azimuth]. The functions below are the single place this
conversion happens; use them at every boundary between the two spaces.
"""

import numpy as np

from Code.crc_ssl import CoverageSet
from Code.plots import plot_roi_neighbours
from Code.utilities import normalize


# ---------------------------------------------------------------------------
# Radians <-> grid-cell-index conversion
# ---------------------------------------------------------------------------

def _grids(nele, nazi):
    """Same grid CRC_SSL_N.py's rir_obj encodes and CoverageSet._project_onto_grid
    reads off room.xl[:,0]/room.yl[0] after its xl/yl swap — reproduced here
    standalone so this conversion doesn't need a CoverageSet/room instance."""
    ele_grid = np.linspace(0.0, np.pi, nele)
    azi_grid = np.linspace(-np.pi, np.pi, nazi)
    return ele_grid, azi_grid


def radians_to_grid_index(position_rad, nele, nazi):
    """[elevation, azimuth] radians -> [el_idx, az_idx] grid-cell indices.

    Parameters
    ----------
    position_rad : array-like, shape (2,)
    nele, nazi : int

    Returns
    -------
    np.ndarray, shape (2,), dtype float — [el_idx, az_idx]
    """
    position_rad = np.asarray(position_rad, dtype=float)
    ele_grid, azi_grid = _grids(nele, nazi)
    el_idx = int(np.argmin(np.abs(position_rad[0] - ele_grid)))
    az_idx = int(np.argmin(np.abs(position_rad[1] - azi_grid)))
    return np.array([el_idx, az_idx], dtype=float)


def grid_index_to_radians(grid_idx, nele, nazi):
    """Inverse of radians_to_grid_index. Non-integer indices are linearly
    interpolated along each grid axis (useful since tracker outputs/priors
    can peak at sub-cell resolution in principle, though the current
    tracker's argmax output is always integer-valued).

    Parameters
    ----------
    grid_idx : array-like, shape (2,) — [el_idx, az_idx]
    nele, nazi : int

    Returns
    -------
    np.ndarray, shape (2,), dtype float — [elevation, azimuth] radians
    """
    grid_idx = np.asarray(grid_idx, dtype=float)
    ele_grid, azi_grid = _grids(nele, nazi)
    ele = np.interp(grid_idx[0], np.arange(nele), ele_grid)
    azi = np.interp(grid_idx[1], np.arange(nazi), azi_grid)
    return np.array([ele, azi], dtype=float)


def positions_to_grid_indices(positions_rad, nele, nazi):
    """Vectorized radians_to_grid_index over an arbitrary leading shape.

    Parameters
    ----------
    positions_rad : array-like, shape (..., 2)
    nele, nazi : int

    Returns
    -------
    np.ndarray, shape (..., 2), dtype float
    """
    positions_rad = np.asarray(positions_rad, dtype=float)
    ele_grid, azi_grid = _grids(nele, nazi)
    flat = positions_rad.reshape(-1, 2)
    el_idx = np.argmin(np.abs(flat[:, 0:1] - ele_grid[None, :]), axis=1)
    az_idx = np.argmin(np.abs(flat[:, 1:2] - azi_grid[None, :]), axis=1)
    out = np.stack([el_idx, az_idx], axis=1).astype(float)
    return out.reshape(positions_rad.shape)


# ---------------------------------------------------------------------------
# Conformal calibration + CP region construction
# ---------------------------------------------------------------------------

def _build_room(npz_file):
    """Replicates CRC_SSL_N.py's _load_npz room construction exactly
    (including the xl/yl swap), since CoverageSet.calibrate()/.test()
    internally call self._project_onto_grid(), which reads room.xl[:,0]
    (expected to be the elevation grid) / room.yl[0] (azimuth grid)."""
    r_obj = npz_file['rir_obj'].item()
    room = type('Room', (object,), r_obj)()
    room.xl, room.yl = room.yl, room.xl
    return room


def calibrate_lambda_thresholds(calib_npz_path, significance_levels, lambda_steps=500,
                                 n_calib_frames=None, seed=0):
    """Calibrate per-speaker conformal lambda thresholds from a real npz.

    Parameters
    ----------
    calib_npz_path : str
    significance_levels : array-like of float
        Target miscoverage rates (alpha), e.g. [0.1, 0.05].
    lambda_steps : int
    n_calib_frames : int or None
        None = use every frame in the file. Otherwise randomly subsample
        (without replacement, seeded) this many frames — frames are
        calibrated independently of each other (CoverageSet has no notion
        of trajectory continuity), so subsampling is a valid speed/accuracy
        tradeoff for dev iteration.
    seed : int

    Returns
    -------
    dict {alpha: np.ndarray shape (n_speakers,)} — lambda threshold per speaker.
    """
    d = np.load(calib_npz_path, allow_pickle=True)
    speaker_pos = d['speaker_pos']
    est_pos = d['all_estimated_positions']
    likelihood_maps = d['all_likelihood_maps']
    room = _build_room(d)

    n_total = speaker_pos.shape[0]
    if n_calib_frames is not None and n_calib_frames < n_total:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n_total, size=n_calib_frames, replace=False))
    else:
        idx = np.arange(n_total)

    lambda_list = np.linspace(0.0, 1.0, lambda_steps)
    cov_set_obj = CoverageSet(
        true_position=speaker_pos[idx],
        estimated_positions=est_pos[idx],
        likelihood_maps=likelihood_maps[idx],
        lambda_list=lambda_list,
        room=room,
        path_=None,
        plot_function=plot_roi_neighbours,
    )
    cov_set_obj.calibrate(plot=False, plot_coverage_set=False)

    return {
        float(alpha): cov_set_obj._calc_conformal_risk_control(float(alpha))
        for alpha in significance_levels
    }


def build_cp_regions_for_frames(likelihood_maps, estimated_positions_rad, lambdas, nele, nazi):
    """Build calibrated CP region masks for a sequence of frames.

    Parameters
    ----------
    likelihood_maps : array-like, shape (T, K, nele, nazi), raw (not pre-normalized)
    estimated_positions_rad : array-like, shape (T, K, 2), radians [ele, azi]
    lambdas : array-like, shape (K,) — per-speaker threshold from calibrate_lambda_thresholds
    nele, nazi : int

    Returns
    -------
    np.ndarray, shape (T, K, nele, nazi), dtype bool
    """
    likelihood_maps = np.asarray(likelihood_maps, dtype=float)
    estimated_positions_rad = np.asarray(estimated_positions_rad, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    T, K = likelihood_maps.shape[:2]

    cp_regions = np.zeros((T, K, nele, nazi), dtype=bool)
    for t in range(T):
        for k in range(K):
            # TODO: as of 2026-09-13, `all_likelihood_maps` in the npz is raw
            # (the SRP-DNN-CP export no longer normalizes it -- see that
            # repo's CLAUDE.md). We normalize here because neighbours_coverage_set's
            # threshold (lambda) is calibrated in normalized [0,1] units, matching
            # Code.crc_ssl.CoverageSet.calibrate()/.test()'s own normalize() calls.
            # But raw per-frame magnitude carries real signal (e.g. it's what
            # separates a directional-burst frame from a clean one -- see the
            # trajectory/burst diagnostic in eval_tracker_conditions.py's history).
            # This should become an explicit, controllable flag (e.g.
            # normalize_before_cp: bool) once we decide whether/how to build a
            # magnitude-aware CP/uncertainty signal instead of always normalizing
            # away the one feature that's actually informative for bursts.
            norm_map = normalize(likelihood_maps[t, k])
            seed = radians_to_grid_index(estimated_positions_rad[t, k], nele, nazi)
            cp_regions[t, k] = CoverageSet.neighbours_coverage_set(
                norm_map, lambdas[k], estimated_position=tuple(seed.astype(int))
            )
    return cp_regions
