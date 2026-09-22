"""
Backward smoother for two-speaker tracks.

After the forward tracker has produced a sequence of track states,
the smoother refines them by using future observations to correct
past estimates (classic RTS / Rauch-Tung-Striebel idea).

The CP features can modulate the smoother gain in the same way they
modulate the tracker update: high uncertainty → trust the smoother
correction less (or: widen the backward pass variance).

TODO
----
- Implement the RTS backward pass for each track independently.
- Incorporate cp_features_sequence to set per-frame smoother gains:
      gain = f(cp_features)  where gain ∈ [0, 1]
      smoothed[t] = forward[t] + gain * (smoothed[t+1] - predicted[t+1])
- Handle the two-speaker assignment sequence so that label switching
  found by the forward pass is propagated consistently through the
  backward pass.
- Consider a joint smoothing step that re-evaluates association given
  the smoothed trajectories.
"""

import numpy as np


def smooth_two_speaker_tracks(tracks, assignments=None, cp_features_sequence=None):
    """Apply backward smoothing to two-speaker forward tracks.

    Parameters
    ----------
    tracks : list of np.ndarray, length 2
        Each element has shape (T, 2) and contains the [elevation, azimuth]
        track positions produced by the forward tracker (one row per frame).
    assignments : list of list or None, optional
        Per-frame assignment lists (each of length 2) as returned by
        associate_two_speakers. Shape: (T, 2). Used to propagate consistent
        labels through the backward pass.
    cp_features_sequence : list of list of dict or None, optional
        Per-frame, per-speaker CP feature dicts as returned by extract_cp_features.
        Shape: (T, 2, feature_dict). Used to modulate smoother gain.

    Returns
    -------
    smoothed_tracks : list of np.ndarray, length 2
        Each element has shape (T, 2). Currently identical to the input
        tracks (placeholder passthrough).

    TODO
    ----
    - Replace passthrough with actual RTS backward pass.
    - Use cp_features_sequence["measurement_uncertainty"] as smoother gain
      modulator: larger uncertainty → smaller correction applied.
    """
    # TODO: implement backward smoother.
    # Placeholder — return tracks unchanged.
    smoothed_tracks = [np.copy(t) for t in tracks]
    return smoothed_tracks
