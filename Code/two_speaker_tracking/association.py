"""
Data association for the two-speaker tracker.

Assigns current measurements (one per speaker candidate) to the two
existing tracks by choosing the permutation that minimises total
Euclidean distance between predicted track positions and measurements.

For exactly K=2 speakers there are only 2 possible permutations:
    option A: track 0 -> measurement 0, track 1 -> measurement 1
    option B: track 0 -> measurement 1, track 1 -> measurement 0

The function returns the assignment that has the lower total distance.
If predicted positions are unavailable (e.g. at frame 0) the identity
assignment [0, 1] is returned.

TODO
----
- Replace the brute-force 2-permutation search with the Hungarian algorithm
  (scipy.optimize.linear_sum_assignment) for generalisation to N speakers.
- Add ambiguity detection: if both permutations have similar cost, flag
  the frame as ambiguous and propagate uncertainty accordingly.
- Incorporate CP features (cp_features_list) into the cost matrix so that
  high-uncertainty measurements are penalised less for large distances.
"""

import numpy as np
from itertools import permutations

from Code.two_speaker_tracking.utils import euclidean_distance


def associate_two_speakers(prev_tracks, current_measurements, current_cp_features=None):
    """Assign each current measurement to one of the two existing tracks.

    Parameters
    ----------
    prev_tracks : list of dict, length 2
        Each dict represents one track and must contain at least:
            "position" : array-like, shape (2,) or None
                The last known / predicted position [el_idx, az_idx] of the
                track, in grid-cell indices (the same convention
                TwoSpeakerTracker uses internally throughout this package,
                NOT radians). If None (e.g. track not yet initialised),
                identity assignment is used.
    current_measurements : array-like, shape (2, 2)
        Two candidate measurements, one per speaker.
        Each row is [el_idx, az_idx] of the measurement, grid-cell indices
        (NOT radians — convert first via npz_adapter.radians_to_grid_index).
    current_cp_features : list of dict or None, optional
        CP feature dicts (one per measurement) returned by extract_cp_features.
        Currently unused — reserved for future cost-matrix augmentation.

    Returns
    -------
    assignment : list of int, length 2
        assignment[k] = m  means track k is assigned to measurement m.
        E.g. [0, 1] → identity, [1, 0] → swapped.

    TODO
    ----
    - Use current_cp_features to weight assignment cost.
    - Extend to N speakers via Hungarian algorithm.
    - Return a confidence score for the assignment.
    """
    measurements = np.asarray(current_measurements, dtype=float)  # (2, 2)

    # Fall back to identity assignment when tracks have no position yet.
    pred_positions = [t.get("position") for t in prev_tracks]
    if any(p is None for p in pred_positions):
        return [0, 1]

    pred_positions = [np.asarray(p, dtype=float) for p in pred_positions]

    # Evaluate both permutations of 2 measurements.
    best_assignment = [0, 1]
    best_cost = np.inf

    for perm in permutations(range(len(measurements))):
        cost = sum(
            euclidean_distance(pred_positions[k], measurements[perm[k]])
            for k in range(2)
        )
        if cost < best_cost:
            best_cost = cost
            best_assignment = list(perm)

    return best_assignment
