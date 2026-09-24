"""
Shared utility helpers for the two-speaker tracker.

These are small, stateless functions that other modules import.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# SRP-DNN grid geometry
# ---------------------------------------------------------------------------
# The exported maps live on
#     ele_grid = linspace(0, pi, nele)     polar angle from +z (Dataset.cart2sph)
#     azi_grid = linspace(-pi, pi, nazi)   so column 0 (-pi) and column nazi-1
#                                          (+pi) are the SAME direction.
# The model evaluates that direction twice (verified: col 0 == col 72 exactly
# in the exported maps), so there are only nazi-1 distinct azimuths, spaced
# 2*pi/(nazi-1) apart, and the azimuth axis is periodic with period nazi-1.

def grid_index_to_angles(grid_idx, grid_shape):
    """[el_idx, az_idx] (any leading shape, float ok) -> (ele, azi) radians."""
    grid_idx = np.asarray(grid_idx, dtype=float)
    nele, nazi = int(grid_shape[0]), int(grid_shape[1])
    ele = grid_idx[..., 0] * (np.pi / (nele - 1))
    azi = -np.pi + grid_idx[..., 1] * (2.0 * np.pi / (nazi - 1))
    return ele, azi


def great_circle_deg(ele_a, azi_a, ele_b, azi_b):
    """Great-circle angle (degrees) between DOAs given as polar elevation
    (from +z) and azimuth, radians. Vectorized. Periodic in azimuth by
    construction and correct near the poles."""
    cos_d = (np.cos(ele_a) * np.cos(ele_b)
             + np.sin(ele_a) * np.sin(ele_b) * np.cos(azi_a - azi_b))
    return np.degrees(np.arccos(np.clip(cos_d, -1.0, 1.0)))


def great_circle_distance_grid(a, b, grid_shape):
    """Great-circle angle (degrees) between two [el_idx, az_idx] grid positions."""
    ele_a, azi_a = grid_index_to_angles(a, grid_shape)
    ele_b, azi_b = grid_index_to_angles(b, grid_shape)
    return float(great_circle_deg(ele_a, azi_a, ele_b, azi_b))


def fold_azimuth_endpoint(arr, kind):
    """(..., nele, nazi) map on the duplicated-endpoint grid -> (..., nele, nazi-1)
    map on the nazi-1 distinct azimuths (column 0 = the +-pi direction).

    kind="mass"       : a probability mass split over the two duplicate columns
                        -> SUM them (total mass preserved).
    kind="likelihood" : two evaluations of the same direction
                        -> AVERAGE them (a direction is not counted twice).
    """
    arr = np.asarray(arr, dtype=float)
    out = arr[..., :-1].copy()
    if kind == "mass":
        out[..., 0] += arr[..., -1]
    elif kind == "likelihood":
        out[..., 0] = 0.5 * (arr[..., 0] + arr[..., -1])
    else:
        raise ValueError(f"kind must be 'mass' or 'likelihood', got {kind!r}")
    return out


def unfold_azimuth_endpoint(folded):
    """Inverse of fold_azimuth_endpoint(kind="mass"): the +-pi mass is split
    equally over columns 0 and nazi-1. fold(unfold(x), "mass") == x exactly."""
    folded = np.asarray(folded, dtype=float)
    out = np.concatenate([folded, folded[..., :1]], axis=-1)
    out[..., 0] *= 0.5
    out[..., -1] *= 0.5
    return out


def propagate_belief_periodic_azimuth(belief_folded, sigma_el, sigma_az):
    """Fixed Gaussian motion propagation T^T b on the FOLDED grid (nele, nazi-1).

    Azimuth: periodic ("wrap") over the nazi-1 distinct columns, so mass near
    +pi crosses to -pi. Elevation: NOT periodic ("constant", unchanged from
    the previous behaviour -- mass blurred past a pole is dropped and the
    result renormalised). Returns a normalised folded belief.
    """
    b = gaussian_filter(np.asarray(belief_folded, dtype=float),
                        sigma=(sigma_el, sigma_az), mode=("constant", "wrap"))
    s = b.sum()
    if s > 1e-300:
        return b / s
    return np.full(b.shape, 1.0 / b.size)


def euclidean_distance(a, b):
    """Return the Euclidean distance between two 1-D coordinate vectors.

    Parameters
    ----------
    a, b : array-like, shape (D,)
        Coordinate vectors in grid-cell indices [el_idx, az_idx] — the same
        convention TwoSpeakerTracker uses internally throughout this package,
        NOT radians. Convert real-valued [elevation, azimuth] radians to this
        convention first via npz_adapter.radians_to_grid_index.

    Returns
    -------
    float
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b))


def validate_two_speaker_inputs(likelihood_maps, cp_regions, estimated_positions):
    """Validate the shapes of per-frame two-speaker inputs.

    Expected conventions (all arrays are numpy arrays):
        likelihood_maps    : (T, 2, nele, nazi)  -- one likelihood map per speaker per frame
        cp_regions         : (T, 2, nele, nazi)  -- boolean CP region mask per speaker per frame
        estimated_positions: (T, 2, 2)           -- [el_idx, az_idx] grid-cell indices per
                                                     speaker per frame, NOT radians. Real-valued
                                                     [elevation, azimuth] radians (as stored in
                                                     SRP-DNN npz exports) must be converted first
                                                     via npz_adapter.radians_to_grid_index.

    Parameters
    ----------
    likelihood_maps : np.ndarray
    cp_regions      : np.ndarray
    estimated_positions : np.ndarray

    Raises
    ------
    ValueError
        If shapes are inconsistent or K != 2.
    """
    lm = np.asarray(likelihood_maps)
    cp = np.asarray(cp_regions)
    ep = np.asarray(estimated_positions)

    if lm.ndim != 4:
        raise ValueError(f"likelihood_maps must be 4-D (T,2,nele,nazi), got shape {lm.shape}")
    if cp.ndim != 4:
        raise ValueError(f"cp_regions must be 4-D (T,2,nele,nazi), got shape {cp.shape}")
    if ep.ndim != 3:
        raise ValueError(f"estimated_positions must be 3-D (T,2,2), got shape {ep.shape}")

    T_lm, K_lm = lm.shape[:2]
    T_cp, K_cp = cp.shape[:2]
    T_ep, K_ep = ep.shape[:2]

    if K_lm != 2 or K_cp != 2 or K_ep != 2:
        raise ValueError(
            f"All inputs must have K=2 speakers. Got K={K_lm},{K_cp},{K_ep}."
        )
    if not (T_lm == T_cp == T_ep):
        raise ValueError(
            f"Time dimensions must match. Got T={T_lm},{T_cp},{T_ep}."
        )
    if lm.shape[2:] != cp.shape[2:]:
        raise ValueError(
            f"Grid size mismatch: likelihood_maps grid {lm.shape[2:]}, "
            f"cp_regions grid {cp.shape[2:]}."
        )
    if ep.shape[2] != 2:
        raise ValueError(
            f"estimated_positions last dim must be 2 ([elevation, azimuth]). Got {ep.shape[2]}."
        )


def build_motion_prior(prev_position, grid_shape, sigma_el=1.0, sigma_az=1.0):
    """Build a normalised 2-D Gaussian motion prior on the elevation-azimuth grid.

    The prior expresses where the speaker is expected to be this frame given
    its previous tracked position.  A narrow Gaussian (small sigma) concentrates
    probability near that position; a wide Gaussian spreads it across the grid.

    Relationship to measurement trust
    ----------------------------------
    Wide prior  (large sigma) → prior barely constrains the posterior
                               → posterior ≈ likelihood map
                               → use when measurement is CONFIDENT (u ≈ 0)

    Narrow prior (small sigma) → prior strongly pulls posterior toward prev_position
                               → posterior resists noise spikes far from the track
                               → use when measurement is UNCERTAIN (u ≈ 1)

    Parameters
    ----------
    prev_position : array-like, shape (2,)
        [elevation_index, azimuth_index] of the previous tracked position,
        in grid-cell coordinates (may be non-integer / sub-grid).
    grid_shape : tuple of int
        (nele, nazi) — shape of the likelihood / CP grid.
    sigma_el : float
        Standard deviation along the elevation axis, in grid cells.
    sigma_az : float
        Standard deviation along the azimuth axis, in grid cells.

    Returns
    -------
    prior_map : np.ndarray, shape (nele, nazi), dtype float64
        2-D Gaussian normalised to sum to 1.
        Truncation at grid borders is handled automatically by
        renormalisation — no special border padding is needed.
    """
    nele, nazi = int(grid_shape[0]), int(grid_shape[1])
    el0 = float(prev_position[0])
    az0 = float(prev_position[1])

    el_coords = np.arange(nele, dtype=float)
    az_coords = np.arange(nazi, dtype=float)
    EL, AZ = np.meshgrid(el_coords, az_coords, indexing="ij")  # (nele, nazi)

    prior = np.exp(
        -0.5 * ((EL - el0) / sigma_el) ** 2
        - 0.5 * ((AZ - az0) / sigma_az) ** 2
    )

    total = prior.sum()
    if total > 1e-300:
        prior /= total
    else:
        # Safety fallback: uniform prior (only reachable if sigma is near-zero
        # and prev_position falls between grid cells — should not happen in practice).
        prior = np.ones((nele, nazi), dtype=float) / (nele * nazi)

    return prior
