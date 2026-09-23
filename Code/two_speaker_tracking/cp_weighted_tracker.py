"""
Two-speaker forward tracker — CP-aware full-belief fusion.

Restores the fusion principle of the earlier "Conformal Prediction-Aware DOA
Tracking" paper (1-D DOA classes), extended to the current 2-D
elevation x azimuth grid. This is a sibling to TwoSpeakerTracker
(tracker.py, the multiplicative Gaussian-product tracker), not a
replacement -- that class is left untouched as the baseline.

Fusion principle
-----------------
Track state is the FULL belief map over the grid, not a single point.
Each frame:

    b_tilde_t,k = T^T b_t-1,k                                  (1)
    b_t,k       = (1 - w_t,k) * b_tilde_t,k + w_t,k * p_tilde_t,k  (2)
    y_hat_t,k   = argmax_y b_t,k(y)                             (3)

(1) Motion propagation: T is a FIXED (not CP-derived) 2-D Gaussian
    transition kernel (sigma_el, sigma_az), applied to the *entire* previous
    belief map via a Gaussian blur (scipy.ndimage.gaussian_filter) --
    mathematically the transition-matrix action T^T b on a translation
    -invariant Gaussian kernel. This is a real recursive Bayes-filter
    propagation: the previous belief's shape/spread carries forward, not
    just its argmax.

(2) Fusion: w_t,k in (0, 1] is a scalar computed purely from speaker k's own
    CP/LCP region at frame t (see cp_weight.compute_cp_weight) -- CP
    uncertainty controls ONLY this interpolation weight, never sigma_el/
    sigma_az. p_tilde_t,k is the full normalized likelihood map, never
    reduced to a peak before fusion.

(3) Each speaker k has its own belief, CP region, descriptors, and weight,
    computed completely independently of the other speaker.

Association (which measurement belongs to which track) is unchanged --
this module calls the existing associate_two_speakers exactly as
tracker.py does.

Array shape conventions
-----------------------
  likelihood_maps_t     : (2, nele, nazi)  raw likelihood map per speaker at t
  cp_regions_t          : (2, nele, nazi)  CP region boolean mask per speaker at t
  estimated_positions_t : (2, 2)           [[el_idx0, az_idx0], [el_idx1, az_idx1]] at t,
                          grid-cell indices (NOT radians), same convention as
                          tracker.py / cp_features.py / association.py / utils.py
  tracks output         : list of (T, 2) arrays, one per speaker, grid-cell indices
  posterior_maps output : (T, 2, nele, nazi) per-frame belief maps b_t,k
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from Code.two_speaker_tracking.cp_features import extract_cp_features
from Code.two_speaker_tracking.association import associate_two_speakers
from Code.two_speaker_tracking.utils import validate_two_speaker_inputs
from Code.two_speaker_tracking.cp_weight import compute_cp_weight

_EPS = 1e-300


class CPWeightedFusionTracker:
    """Forward tracker for exactly two simultaneous speakers, CP-weighted
    full-belief fusion (see module docstring).

    Parameters
    ----------
    n_speakers : int
        Must be 2. Reserved for future N-speaker generalisation.
    sigma_el, sigma_az : float
        FIXED standard deviations (grid cells) of the motion transition
        kernel T, used every frame regardless of CP uncertainty. Free/
        tunable hyperparameters not specified by the fusion equations
        themselves -- defaults are in the same magnitude as
        TwoSpeakerTracker's sigma_min (tracker.py), a reasonable starting
        point given that documented real inter-frame GT motion in these
        exports is well under 1 grid cell/frame. Not tuned here.
    lambda_var, lambda_size, lambda_span : float
        Coefficients of the CP-aware weight formula (see
        cp_weight.compute_cp_weight). Starting defaults only -- not tuned
        here.
    """

    def __init__(self, n_speakers=2, sigma_el=2.0, sigma_az=2.0,
                 lambda_var=1.0, lambda_size=2.0, lambda_span=1.0):
        if n_speakers != 2:
            raise ValueError("CPWeightedFusionTracker currently supports exactly 2 speakers.")
        self.n_speakers = n_speakers
        self.sigma_el = float(sigma_el)
        self.sigma_az = float(sigma_az)
        self.lambda_var = float(lambda_var)
        self.lambda_size = float(lambda_size)
        self.lambda_span = float(lambda_span)
        self.reset()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self):
        """Reset internal state so the tracker can be reused on a new sequence."""
        # Each track carries both a point "position" (for association, which
        # only needs a point -- see association.py) and the full "belief"
        # map (for the recursive T^T b propagation).
        self._tracks = [
            {"position": None, "belief": None, "history": []}
            for _ in range(self.n_speakers)
        ]
        self._assignments_history = []
        self._cp_features_history = []
        self._posterior_history = []   # list of (2, nele, nazi) belief arrays, one per frame
        self._debug_history = []       # list of (2,) list of {"w","size_norm","span_norm","var_norm"}

    def step(self, likelihood_maps_t, cp_regions_t, estimated_positions_t):
        """Process a single frame.

        Parameters
        ----------
        likelihood_maps_t : array-like, shape (2, nele, nazi)
        cp_regions_t      : array-like, shape (2, nele, nazi)
        estimated_positions_t : array-like, shape (2, 2)

        Returns
        -------
        frame_result : dict
            "positions"   : np.ndarray, shape (2, 2)  – updated track positions
            "assignment"  : list of int, length 2
            "cp_features" : list of dict, length 2
            "posteriors"  : np.ndarray, shape (2, nele, nazi)  – belief maps b_t,k
            "debug"       : list of dict, length 2  – {"w","size_norm","span_norm","var_norm"}
        """
        likelihood_maps_t = np.asarray(likelihood_maps_t, dtype=float)
        cp_regions_t = np.asarray(cp_regions_t, dtype=float)
        measurements = np.asarray(estimated_positions_t, dtype=float)  # (2, 2)

        # Step (a): extract CP uncertainty features per speaker candidate.
        cp_features_t = [
            extract_cp_features(
                cp_region=cp_regions_t[k],
                likelihood_map=likelihood_maps_t[k],
                estimated_position=measurements[k],
            )
            for k in range(self.n_speakers)
        ]

        # Step (b): associate measurements to existing tracks. Unchanged --
        # identical call to tracker.py's.
        assignment = associate_two_speakers(
            prev_tracks=self._tracks,
            current_measurements=measurements,
            current_cp_features=cp_features_t,
        )

        # Step (c): update each track's full belief via CP-weighted fusion.
        nele, nazi = likelihood_maps_t.shape[1:]
        updated_positions = np.empty((self.n_speakers, 2), dtype=float)
        frame_posteriors = np.empty((self.n_speakers, nele, nazi), dtype=float)
        frame_debug = []

        for k in range(self.n_speakers):
            m_idx = assignment[k]
            updated_pos, belief, w, descriptors = self._update_single_track(
                prev_belief=self._tracks[k]["belief"],
                measurement=measurements[m_idx],
                cp_features=cp_features_t[m_idx],
                likelihood_map=likelihood_maps_t[m_idx],
            )
            self._tracks[k]["position"] = updated_pos
            self._tracks[k]["belief"] = belief
            self._tracks[k]["history"].append(updated_pos.copy())
            updated_positions[k] = updated_pos
            frame_posteriors[k] = belief
            frame_debug.append({"w": w, **descriptors})

        self._assignments_history.append(assignment)
        self._cp_features_history.append(cp_features_t)
        self._posterior_history.append(frame_posteriors)
        self._debug_history.append(frame_debug)

        return {
            "positions": updated_positions,
            "assignment": assignment,
            "cp_features": cp_features_t,
            "posteriors": frame_posteriors,
            "debug": frame_debug,
        }

    def run(self, likelihood_maps, cp_regions, estimated_positions):
        """Run the tracker over an entire sequence.

        Parameters
        ----------
        likelihood_maps     : array-like, shape (T, 2, nele, nazi)
        cp_regions          : array-like, shape (T, 2, nele, nazi)
        estimated_positions : array-like, shape (T, 2, 2)

        Returns
        -------
        result : dict, output-compatible with TwoSpeakerTracker.run()
            "tracks"         : list of np.ndarray, shape (T, 2) — one per speaker
            "assignments"    : np.ndarray, shape (T, 2)
            "cp_features"    : list of list of dict, shape (T, 2)
            "posterior_maps" : np.ndarray, shape (T, 2, nele, nazi) — belief maps b_t,k
            "debug"          : dict with extra diagnostic information, including
                                per-frame/per-speaker "w", "size_norm", "span_norm",
                                "var_norm" (each np.ndarray, shape (T, 2))
        """
        likelihood_maps = np.asarray(likelihood_maps, dtype=float)
        cp_regions = np.asarray(cp_regions, dtype=float)
        estimated_positions = np.asarray(estimated_positions, dtype=float)

        validate_two_speaker_inputs(likelihood_maps, cp_regions, estimated_positions)

        T = likelihood_maps.shape[0]
        self.reset()

        frame_positions = []  # (T, 2, 2)
        for t in range(T):
            frame_result = self.step(
                likelihood_maps_t=likelihood_maps[t],
                cp_regions_t=cp_regions[t],
                estimated_positions_t=estimated_positions[t],
            )
            frame_positions.append(frame_result["positions"])

        frame_positions = np.stack(frame_positions, axis=0)  # (T, 2, 2)

        # Reformat as one (T, 2) trajectory per speaker.
        tracks = [frame_positions[:, k, :] for k in range(self.n_speakers)]

        # Reformat per-frame debug scalars into (T, 2) arrays per key.
        debug_keys = ("w", "size_norm", "span_norm", "var_norm")
        debug_arrays = {
            key: np.array([[frame[k][key] for k in range(self.n_speakers)]
                            for frame in self._debug_history], dtype=float)
            for key in debug_keys
        }

        return {
            "tracks": tracks,
            "assignments": np.array(self._assignments_history),          # (T, 2)
            "cp_features": self._cp_features_history,                    # (T, 2, dict)
            "posterior_maps": np.stack(self._posterior_history, axis=0),  # (T, 2, nele, nazi)
            "debug": {
                "n_frames": T,
                "grid_shape": likelihood_maps.shape[2:],
                **debug_arrays,
            },
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_single_track(self, prev_belief, measurement, cp_features, likelihood_map):
        """Update one track's belief via CP-weighted fusion (see module docstring).

        Parameters
        ----------
        prev_belief   : np.ndarray, shape (nele, nazi), or None
            b_t-1,k. None on frame 0 -- no prior belief available yet.
        measurement   : np.ndarray, shape (2,)
            Assigned detector measurement [el_idx, az_idx]. Used only for
            frame-0 initialisation.
        cp_features   : dict
            Output of extract_cp_features for this speaker's region.
        likelihood_map : np.ndarray, shape (nele, nazi)
            Raw (unnormalised) likelihood map from the upstream model.

        Returns
        -------
        updated_position : np.ndarray, shape (2,)
            argmax(belief), grid-cell indices (float).
        belief : np.ndarray, shape (nele, nazi)
            b_t,k, normalised to sum to 1.
        w : float
            CP-aware fusion weight used this frame (np.nan on frame 0, where
            there is no prediction to fuse against).
        descriptors : dict
            {"size_norm", "span_norm", "var_norm"} (np.nan-filled on frame 0).
        """
        measurement = np.asarray(measurement, dtype=float)
        likelihood_map = np.asarray(likelihood_map, dtype=float)
        grid_shape = likelihood_map.shape  # (nele, nazi)
        nele, nazi = grid_shape

        # p_tilde: full normalized likelihood map (never reduced to a peak).
        p_tilde = np.clip(likelihood_map, 0.0, None)
        psum = p_tilde.sum()
        if psum > _EPS:
            p_tilde = p_tilde / psum
        else:
            # Degenerate all-zero/negative likelihood map: fall back to
            # uniform rather than propagating NaN.
            p_tilde = np.full(grid_shape, 1.0 / (nele * nazi))

        # ---- Frame 0: no prior belief available --------------------------
        if prev_belief is None:
            belief = p_tilde
            position = measurement.copy()  # mirrors TwoSpeakerTracker's frame-0 convention
            descriptors = {"size_norm": np.nan, "span_norm": np.nan, "var_norm": np.nan}
            return position, belief, np.nan, descriptors

        # ---- Frames 1+: T^T b_t-1,k, then CP-weighted fusion --------------
        # Motion propagation over the ENTIRE previous belief map (fixed
        # sigma_el/sigma_az, independent of CP -- CP affects only w below).
        b_tilde = gaussian_filter(prev_belief, sigma=(self.sigma_el, self.sigma_az),
                                   mode="constant")
        bsum = b_tilde.sum()
        if bsum > _EPS:
            b_tilde = b_tilde / bsum
        else:
            b_tilde = np.full(grid_shape, 1.0 / (nele * nazi))

        w, descriptors = compute_cp_weight(
            cp_features, grid_shape,
            lambda_var=self.lambda_var, lambda_size=self.lambda_size,
            lambda_span=self.lambda_span,
        )

        belief = (1.0 - w) * b_tilde + w * p_tilde
        bsum2 = belief.sum()
        if bsum2 > _EPS:
            belief = belief / bsum2
        else:
            belief = np.full(grid_shape, 1.0 / (nele * nazi))

        peak_flat = np.argmax(belief)
        peak_idx = np.unravel_index(peak_flat, grid_shape)
        position = np.array([float(peak_idx[0]), float(peak_idx[1])])

        return position, belief, w, descriptors
