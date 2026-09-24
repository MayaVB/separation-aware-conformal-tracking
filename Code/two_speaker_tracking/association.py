"""
Data association for the two-speaker tracker.

Assigns the two current IDL measurement packages (one per IDL slot) to the
two existing tracks by choosing the permutation that minimises the total
distance between track reference positions and measurements.

For exactly K=2 speakers there are only 2 possible permutations:
    option A: track 0 -> measurement 0, track 1 -> measurement 1
    option B: track 0 -> measurement 1, track 1 -> measurement 0

The function returns the assignment that has the lower total distance
(exact, not greedy; an exact tie keeps option A). If reference positions
are unavailable (e.g. at frame 0) the identity assignment [0, 1] is returned.

Distance
--------
- grid_shape given  : great-circle angle (degrees) between the two DOAs,
                      converting grid indices with utils.grid_index_to_angles.
                      Periodic in azimuth, correct near the poles. Used by
                      CPWeightedFusionTracker.
- grid_shape None   : legacy Euclidean grid-index distance (not periodic in
                      azimuth). Kept only so TwoSpeakerTracker (the frozen
                      baseline) is unchanged.

TODO
----
- Add ambiguity detection: if both permutations have similar cost, flag
  the frame as ambiguous and propagate uncertainty accordingly (the costs
  are already returned via return_costs=True for diagnostics).
- Incorporate CP features (cp_features_list) into the cost matrix so that
  high-uncertainty measurements are penalised less for large distances.
"""

import numpy as np
from itertools import permutations

from Code.two_speaker_tracking.utils import euclidean_distance, great_circle_distance_grid


def associate_two_speakers(prev_tracks, current_measurements, current_cp_features=None,
                           grid_shape=None, return_costs=False):
    """Assign each current measurement to one of the two existing tracks.

    Parameters
    ----------
    prev_tracks : list of dict, length 2
        Each dict represents one track and must contain at least:
            "position" : array-like, shape (2,) or None
                The track's reference position [el_idx, az_idx] for
                association, in grid-cell indices (NOT radians). For
                CPWeightedFusionTracker this is the PREDICTED position
                (argmax of T^T b_t-1). If None, identity assignment is used.
    current_measurements : array-like, shape (2, 2)
        Two candidate measurements, one per IDL slot, [el_idx, az_idx]
        grid-cell indices (NOT radians — convert first via
        npz_adapter.radians_to_grid_index).
    current_cp_features : list of dict or None, optional
        Currently unused — reserved for future cost-matrix augmentation.
    grid_shape : tuple (nele, nazi) or None
        If given, the cost is great-circle distance in degrees (see module
        docstring); if None, legacy Euclidean grid-index distance.
    return_costs : bool
        If True, also return {"cost_identity", "cost_swapped"}: total cost
        of option A ([0, 1]) and option B ([1, 0]); both NaN when the
        identity fallback is used.

    Returns
    -------
    assignment : list of int, length 2
        assignment[k] = m  means track k is assigned to measurement (IDL slot) m.
    costs : dict (only if return_costs)
    """
    measurements = np.asarray(current_measurements, dtype=float)  # (2, 2)

    pred_positions = [t.get("position") for t in prev_tracks]
    if any(p is None for p in pred_positions):
        assignment = [0, 1]
        if return_costs:
            return assignment, {"cost_identity": np.nan, "cost_swapped": np.nan}
        return assignment

    pred_positions = [np.asarray(p, dtype=float) for p in pred_positions]

    if grid_shape is None:
        dist = euclidean_distance
    else:
        dist = lambda a, b: great_circle_distance_grid(a, b, grid_shape)

    # Evaluate both permutations of 2 measurements. permutations() yields
    # (0, 1) first, so a strict "<" keeps the identity on an exact tie.
    best_assignment = [0, 1]
    best_cost = np.inf
    perm_costs = {}
    for perm in permutations(range(len(measurements))):
        cost = sum(dist(pred_positions[k], measurements[perm[k]]) for k in range(2))
        perm_costs[perm] = cost
        if cost < best_cost:
            best_cost = cost
            best_assignment = list(perm)

    if return_costs:
        return best_assignment, {"cost_identity": float(perm_costs[(0, 1)]),
                                 "cost_swapped": float(perm_costs[(1, 0)])}
    return best_assignment
