"""
Two-speaker forward tracker — belief-map Bayesian update.

Philosophy
----------
Mirrors the CRC framework: CP regions are NOT hard gates.
At each frame the tracker:
  1. Extracts CP uncertainty features for each speaker's CP region.
  2. Associates the two current measurements to the two existing tracks.
  3. Updates each track with a Bayesian fusion:

         posterior ∝ likelihood_map × motion_prior

     where the motion_prior is a Gaussian centred at the previous tracked
     position and CP uncertainty controls prior sharpness:
       - confident measurement (small CP region, u ≈ 0) → wide prior
         → posterior ≈ likelihood → track follows measurement peak
       - uncertain measurement (large CP region, u ≈ 1) → narrow prior
         → posterior pulled toward previous position → track is stabilised

Array shape conventions
-----------------------
  likelihood_maps_t     : (2, nele, nazi)  raw likelihood map per speaker at t
  cp_regions_t          : (2, nele, nazi)  CP region boolean mask per speaker at t
  estimated_positions_t : (2, 2)           [[el_idx0, az_idx0], [el_idx1, az_idx1]] at t,
                          in grid-cell indices (NOT radians) — enforced consistently
                          across cp_features.py/association.py/utils.py. Real-valued
                          [elevation, azimuth] radians (e.g. from SRP-DNN npz exports)
                          must be converted first via npz_adapter.radians_to_grid_index.
                          used for association and frame-0 initialisation
  tracks output         : list of (T, 2) arrays, one per speaker, grid-cell indices
  posterior_maps output : (T, 2, nele, nazi) per-frame posterior belief maps

TODO
----
- Add a motion model (constant velocity, random walk, …) to produce a
  predicted position that is more informative than the last accepted position.
- Extend to N speakers by generalising association and track management.
"""

import numpy as np

from Code.two_speaker_tracking.cp_features import extract_cp_features
from Code.two_speaker_tracking.association import associate_two_speakers
from Code.two_speaker_tracking.utils import validate_two_speaker_inputs, build_motion_prior


class TwoSpeakerTracker:
    """Forward tracker for exactly two simultaneous speakers.

    Parameters
    ----------
    n_speakers : int
        Must be 2. Reserved for future N-speaker generalisation.
    sigma_min, sigma_max : float
        Motion-prior standard deviation bounds, in grid cells (see
        _update_single_track). Defaults were empirically re-tuned against
        real SRP-DNN npz exports (eval_tracker_conditions.py): the original
        defaults (0.5, 3.0) measurably hurt accuracy even on clean baseline
        data (MAE_azi ~3.1deg raw vs ~6.1deg tracked) because sigma_max=3.0
        grid cells is still narrow enough, relative to the 37x73 grid, to
        meaningfully reshape the posterior away from the likelihood argmax
        even in the "confident" regime. Real inter-frame GT motion in these
        exports is tiny (median ~0.28deg/frame, p90 ~1.5deg/frame, i.e. well
        under 1 grid cell at 5deg/cell), so widening sigma_max does not
        sacrifice genuine motion-tracking ability on this data.
    """

    _SIGMA_MIN = 2.0    # narrow  -> posterior pulled toward prediction (uncertain frame)
    _SIGMA_MAX = 12.0   # wide    -> posterior ~= likelihood map        (confident frame)

    def __init__(self, n_speakers=2, sigma_min=None, sigma_max=None):
        if n_speakers != 2:
            raise ValueError("TwoSpeakerTracker currently supports exactly 2 speakers.")
        self.n_speakers = n_speakers
        self.sigma_min = self._SIGMA_MIN if sigma_min is None else float(sigma_min)
        self.sigma_max = self._SIGMA_MAX if sigma_max is None else float(sigma_max)
        self.reset()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self):
        """Reset internal state so the tracker can be reused on a new sequence."""
        # Each track is a dict with the last known position and history list.
        self._tracks = [
            {"position": None, "history": []}
            for _ in range(self.n_speakers)
        ]
        self._assignments_history  = []
        self._cp_features_history  = []
        self._posterior_history    = []   # list of (2, nele, nazi) arrays, one per frame

    def step(self, likelihood_maps_t, cp_regions_t, estimated_positions_t):
        """Process a single frame.

        Parameters
        ----------
        likelihood_maps_t : array-like, shape (2, nele, nazi)
        cp_regions_t      : array-like, shape (2, nele, nazi)
        estimated_positions_t : array-like, shape (2, 2)
            [[elevation_speaker0, azimuth_speaker0],
             [elevation_speaker1, azimuth_speaker1]]

        Returns
        -------
        frame_result : dict
            "positions"   : np.ndarray, shape (2, 2)  – updated track positions
            "assignment"  : list of int, length 2
            "cp_features" : list of dict, length 2
            "posteriors"  : np.ndarray, shape (2, nele, nazi)
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

        # Step (b): associate measurements to existing tracks.
        assignment = associate_two_speakers(
            prev_tracks=self._tracks,
            current_measurements=measurements,
            current_cp_features=cp_features_t,
        )

        # Step (c): update each track with Bayesian belief-map fusion.
        nele, nazi = likelihood_maps_t.shape[1:]
        updated_positions  = np.empty((self.n_speakers, 2), dtype=float)
        frame_posteriors   = np.empty((self.n_speakers, nele, nazi), dtype=float)

        for k in range(self.n_speakers):
            m_idx = assignment[k]
            updated_pos, posterior = self._update_single_track(
                prediction=self._tracks[k]["position"],
                measurement=measurements[m_idx],
                cp_features=cp_features_t[m_idx],
                likelihood_map=likelihood_maps_t[m_idx],
            )
            self._tracks[k]["position"] = updated_pos
            self._tracks[k]["history"].append(updated_pos.copy())
            updated_positions[k]  = updated_pos
            frame_posteriors[k]   = posterior

        self._assignments_history.append(assignment)
        self._cp_features_history.append(cp_features_t)
        self._posterior_history.append(frame_posteriors)

        return {
            "positions":  updated_positions,
            "assignment": assignment,
            "cp_features": cp_features_t,
            "posteriors": frame_posteriors,
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
        result : dict
            "tracks"         : list of np.ndarray, shape (T, 2) — one per speaker
            "assignments"    : np.ndarray, shape (T, 2)
            "cp_features"    : list of list of dict, shape (T, 2)
            "posterior_maps" : np.ndarray, shape (T, 2, nele, nazi)
                               per-frame posterior belief maps for each speaker
            "debug"          : dict with extra diagnostic information
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

        return {
            "tracks":         tracks,
            "assignments":    np.array(self._assignments_history),         # (T, 2)
            "cp_features":    self._cp_features_history,                   # (T, 2, dict)
            "posterior_maps": np.stack(self._posterior_history, axis=0),   # (T, 2, nele, nazi)
            "debug": {
                "n_frames":   T,
                "grid_shape": likelihood_maps.shape[2:],
            },
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_single_track(self, prediction, measurement, cp_features, likelihood_map):
        """Update one track with a Bayesian belief-map fusion.

        Posterior formula
        -----------------
            posterior(x) ∝ likelihood(x) × prior(x)

        where
            prior(x) = N(x | prev_position, sigma·I)   (2-D isotropic Gaussian)
            likelihood(x) = normalised likelihood_map

        CP uncertainty modulates prior sharpness
        -----------------------------------------
        u = cp_features["measurement_uncertainty"]  ∈ [0, 1]

            sigma = sigma_max - u × (sigma_max - sigma_min)

        Direction (important):
            u = 0  (small CP region → confident measurement)
                   → sigma = self.sigma_max (wide prior, barely constrains)
                   → posterior ≈ likelihood_map
                   → tracked position follows the measurement peak

            u = 1  (large CP region → uncertain measurement)
                   → sigma = self.sigma_min (narrow prior, strong memory)
                   → posterior concentrated near prev_position
                   → tracked position is stabilised against noise spikes

        A wider prior exerts LESS influence on the posterior, so the
        measurement (likelihood) dominates — this is the correct direction
        for trusting a confident measurement.

        Parameters
        ----------
        prediction    : np.ndarray, shape (2,) or None
            Last accepted tracked position [el_idx, az_idx].
            None on frame 0 — measurement is returned directly.
        measurement   : np.ndarray, shape (2,)
            Assigned detector measurement [el_idx, az_idx].
            Used only for frame-0 initialisation; posterior argmax is used
            for all subsequent frames.
        cp_features   : dict
            Output of extract_cp_features for this speaker's region.
        likelihood_map : np.ndarray, shape (nele, nazi)
            Raw (unnormalised) likelihood map from the upstream model.

        Returns
        -------
        updated_position : np.ndarray, shape (2,)
            Argmax of the posterior (grid-cell indices, float).
        posterior_map    : np.ndarray, shape (nele, nazi)
            Normalised posterior belief map (sums to 1).
            On frame 0 the normalised likelihood is returned as a proxy.
        """
        _EPS = 1e-300

        measurement    = np.asarray(measurement,    dtype=float)
        likelihood_map = np.asarray(likelihood_map, dtype=float)
        grid_shape     = likelihood_map.shape         # (nele, nazi)

        # Normalise likelihood to a probability-like map.
        lhood = np.maximum(likelihood_map, 0.0)
        lsum  = lhood.sum()
        lhood = lhood / (lsum + _EPS)

        # Add a tiny uniform floor (1 part in 10^6 of total mass, spread uniformly).
        # Without this, when the noise spike and the prior are far apart, both
        # distributions are ~0 at every grid cell and np.argmax finds the
        # product's maximum at a spurious midpoint instead of near the prior center.
        # With the floor, cells near the prior center have lhood ≈ floor > 0,
        # so the prior peak always dominates the product and the track stays stable.
        lhood += 1e-6 / lhood.size
        lhood /= lhood.sum()

        # ---- Frame 0: no prior available --------------------------------
        if prediction is None:
            # Initialise the track at the detector's measurement output.
            # Return the normalised likelihood as a proxy posterior.
            return measurement.copy(), lhood.copy()

        # ---- Frames 1+ : Bayesian fusion --------------------------------
        prediction = np.asarray(prediction, dtype=float)

        # Sanitise the overall uncertainty scalar (fallback for either axis
        # below if its own per-axis width is unavailable).
        u = cp_features.get("measurement_uncertainty", 0.0)
        u = float(np.nan_to_num(u, nan=0.0, posinf=1.0, neginf=0.0))
        u = float(np.clip(u, 0.0, 1.0))

        # Per-axis uncertainty: elevation and azimuth are tracked with
        # independent prior widths, each driven by the CP region's own
        # bounding-box span along that axis (cp_width_el/cp_width_az from
        # extract_cp_features), normalised by that axis's grid size. This
        # matters because the array resolves elevation and azimuth with
        # different reliability/dynamics, so a single isotropic sigma
        # (the previous behaviour) over- or under-constrains one axis
        # whenever the CP region isn't roughly square.
        u_el = cp_features.get("cp_width_el", np.nan) / grid_shape[0]
        u_az = cp_features.get("cp_width_az", np.nan) / grid_shape[1]
        u_el = u if not np.isfinite(u_el) else float(np.clip(u_el, 0.0, 1.0))
        u_az = u if not np.isfinite(u_az) else float(np.clip(u_az, 0.0, 1.0))

        # Prior sigma per axis: decreases as uncertainty increases (more
        # uncertainty -> narrower prior on that axis specifically).
        sigma_el = self.sigma_max - u_el * (self.sigma_max - self.sigma_min)
        sigma_az = self.sigma_max - u_az * (self.sigma_max - self.sigma_min)

        # Build Gaussian motion prior centred at previous tracked position.
        prior_map = build_motion_prior(
            prev_position=prediction,
            grid_shape=grid_shape,
            sigma_el=sigma_el,
            sigma_az=sigma_az,
        )

        # Fuse: posterior ∝ likelihood × prior.
        posterior = lhood * prior_map
        psum      = posterior.sum()
        if psum > _EPS:
            posterior /= psum
        else:
            posterior = prior_map.copy()   # degenerate fallback

        # Extract tracked position as the posterior argmax.
        peak_flat = np.argmax(posterior)
        peak_idx  = np.unravel_index(peak_flat, grid_shape)
        updated_position = np.array([float(peak_idx[0]), float(peak_idx[1])])

        return updated_position, posterior
