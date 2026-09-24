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
Each frame t >= 1, for tracks k = 0, 1 and IDL measurement slots m = 0, 1:

    b_pred_t,k  = T^T b_t-1,k                                       (1)
    x_pred_t,k  = argmax_y b_pred_t,k(y)                              (2)
    a_t         = argmin over the 2 permutations of
                    sum_k GC(x_pred_t,k, z_t,a_t(k))                  (3)
    b_t,k       = (1 - w_t,a_t(k)) * b_pred_t,k + w_t,a_t(k) * p_t,a_t(k)   (4)
    y_hat_t,k   = argmax_y b_t,k(y)                                   (5)

(1) Motion propagation: T is a FIXED (not CP-derived) 2-D Gaussian
    transition kernel (sigma_el, sigma_az), applied to the *entire* previous
    belief map via a Gaussian blur -- the transition-matrix action T^T b of a
    translation-invariant kernel. Azimuth is periodic, elevation is not (see
    "Periodic azimuth" below).

(2)-(3) Association happens AFTER propagation and BEFORE fusion, against
    the PREDICTED position of each track. GC = great-circle angle between
    DOAs (association.py). z_t,m is IDL slot m's estimated DOA. Exact
    2-permutation search.

(4) Package-level fusion: track k receives the COMPLETE measurement package
    of its assigned slot a_t(k) -- estimated DOA, likelihood map p (full
    normalized map, never reduced to a peak), CP-region descriptors A/S/V,
    and the CP weight w computed from that slot's region
    (cp_weight.compute_cp_weight). CP uncertainty controls ONLY w, never
    sigma_el/sigma_az. Then renormalise.

Frame 0: no prior; b_0,k = p_0,k of the identity assignment (slot k ->
track k) and the output position is slot k's estimated DOA (same convention
as TwoSpeakerTracker). Track labels are therefore arbitrary but persistent.

Periodic azimuth
----------------
The grid's azimuth axis is linspace(-pi, pi, nazi): columns 0 and nazi-1
are the same direction. All belief arithmetic is done on the FOLDED grid of
the nazi-1 distinct azimuths (utils.fold_azimuth_endpoint): a belief (mass)
folds by summing the two duplicate columns, a likelihood map folds by
averaging them (two evaluations of one direction). The blur uses
mode=("constant", "wrap") on the folded grid. Stored/returned beliefs are
unfolded back to (nele, nazi) by splitting the +-pi mass equally over the two
duplicate columns, so fold(unfold(b)) == b exactly across frames.

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

from Code.two_speaker_tracking.cp_features import extract_cp_features
from Code.two_speaker_tracking.association import associate_two_speakers
from Code.two_speaker_tracking.utils import (
    validate_two_speaker_inputs, fold_azimuth_endpoint, unfold_azimuth_endpoint,
    propagate_belief_periodic_azimuth,
)
from Code.two_speaker_tracking.cp_weight import compute_cp_weight

_EPS = 1e-300

# Main method: area-only weight w = exp(-gamma * A), A = normalized CP region
# area. gamma_M = 32 was selected for Mondrian-hatD (M=5) regions on the
# close-separation criterion; the Global CP counterpart is frozen at 16.
GAMMA_MONDRIAN = 32.0
GAMMA_GLOBAL = 16.0


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
    fixed_w : float or None
        Non-CP baseline: constant fusion weight for every frame/track
        (fixed_w=1.0 = association only, no temporal fusion). None (default)
        = CP-derived w.
    """

    def __init__(self, n_speakers=2, sigma_el=2.0, sigma_az=2.0,
                 lambda_var=1.0, lambda_size=2.0, lambda_span=1.0, fixed_w=None):
        if n_speakers != 2:
            raise ValueError("CPWeightedFusionTracker currently supports exactly 2 speakers.")
        self.n_speakers = n_speakers
        self.sigma_el = float(sigma_el)
        self.sigma_az = float(sigma_az)
        self.lambda_var = float(lambda_var)
        self.lambda_size = float(lambda_size)
        self.lambda_span = float(lambda_span)
        # Baseline switch: if not None, every frame/track uses this constant w
        # instead of the CP-derived one (same association, same motion model;
        # A/S/V are still computed and reported). None = CP-weighted (default).
        self.fixed_w = None if fixed_w is None else float(fixed_w)
        self.reset()

    @classmethod
    def a_only(cls, gamma, **kwargs):
        """Area-only weight w = exp(-gamma * A); span and dispersion terms off."""
        return cls(lambda_size=gamma, lambda_span=0.0, lambda_var=0.0, **kwargs)

    @classmethod
    def for_mondrian(cls, **kwargs):
        """Main method: A-only weight with gamma_M = 32. Feed it the Mondrian-hatD
        (M=5) CP regions as `cp_regions` (build them with mondrian.calibrate_mondrian +
        mondrian.build_mondrian_regions); the tracker itself is region-agnostic."""
        return cls.a_only(GAMMA_MONDRIAN, **kwargs)

    @classmethod
    def for_global(cls, **kwargs):
        """Global-CP baseline: A-only weight with gamma_G = 16."""
        return cls.a_only(GAMMA_GLOBAL, **kwargs)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self):
        """Reset internal state so the tracker can be reused on a new sequence."""
        # Each track carries its last fused point "position" (output only),
        # and the full "belief" map (unfolded (nele, nazi), for the recursive
        # T^T b propagation). Association uses the PREDICTED position, which
        # is recomputed from the belief every frame in step().
        self._tracks = [
            {"position": None, "belief": None, "history": []}
            for _ in range(self.n_speakers)
        ]
        self._assignments_history = []
        self._cp_features_history = []
        self._posterior_history = []   # list of (2, nele, nazi) belief arrays, one per frame
        self._debug_history = []       # list of (2,) list of {"w","size_norm","span_norm","var_norm"}
        self._track_diag_history = []  # list of per-frame track-ordered diagnostics (see step())

    def step(self, likelihood_maps_t, cp_regions_t, estimated_positions_t):
        """Process a single frame.

        Parameters
        ----------
        likelihood_maps_t : array-like, shape (2, nele, nazi)   -- IDL-slot ordered
        cp_regions_t      : array-like, shape (2, nele, nazi)   -- IDL-slot ordered
        estimated_positions_t : array-like, shape (2, 2)        -- IDL-slot ordered

        Returns
        -------
        frame_result : dict
            "positions"   : np.ndarray, shape (2, 2)  – updated track positions (TRACK order)
            "assignment"  : list of int, length 2 – assignment[k] = IDL slot given to track k
            "cp_features" : list of dict, length 2 – SLOT order (NOT reordered)
            "posteriors"  : np.ndarray, shape (2, nele, nazi)  – belief maps b_t,k (TRACK order)
            "debug"       : list of dict, length 2 (TRACK order) – {"w","size_norm","span_norm","var_norm"}
            "track_diag"  : dict of TRACK-ordered diagnostics, see run()
        """
        likelihood_maps_t = np.asarray(likelihood_maps_t, dtype=float)
        cp_regions_t = np.asarray(cp_regions_t, dtype=float)
        measurements = np.asarray(estimated_positions_t, dtype=float)  # (2, 2)
        grid_shape = likelihood_maps_t.shape[1:]
        nele, nazi = grid_shape

        # Step (a): measurement packages, SLOT order. Each slot's CP features
        # come from that slot's own region/map/DOA; w is derived from them.
        cp_features_t = [
            extract_cp_features(
                cp_region=cp_regions_t[m],
                likelihood_map=likelihood_maps_t[m],
                estimated_position=measurements[m],
            )
            for m in range(self.n_speakers)
        ]
        slot_w, slot_desc = zip(*[
            compute_cp_weight(cp_features_t[m], grid_shape,
                              lambda_var=self.lambda_var, lambda_size=self.lambda_size,
                              lambda_span=self.lambda_span)
            for m in range(self.n_speakers)
        ])
        if self.fixed_w is not None:
            slot_w = (self.fixed_w,) * self.n_speakers

        # Step (b): propagate each track with the FIXED motion model and read
        # off its predicted position (frames 1+ only).
        initialized = self._tracks[0]["belief"] is not None
        pred_folded = [None] * self.n_speakers
        pred_positions = [None] * self.n_speakers
        if initialized:
            for k in range(self.n_speakers):
                prev_folded = fold_azimuth_endpoint(self._tracks[k]["belief"], kind="mass")
                pred_folded[k] = propagate_belief_periodic_azimuth(
                    prev_folded, self.sigma_el, self.sigma_az)
                pred_positions[k] = self._argmax_position(pred_folded[k])

        # Step (c): associate the two slot packages to the PREDICTED tracks,
        # great-circle cost, exact 2-permutation search.
        assignment, costs = associate_two_speakers(
            prev_tracks=[{"position": p} for p in pred_positions],
            current_measurements=measurements,
            current_cp_features=cp_features_t,
            grid_shape=grid_shape,
            return_costs=True,
        )

        # Step (d): each track fuses its assigned slot's COMPLETE package.
        updated_positions = np.empty((self.n_speakers, 2), dtype=float)
        frame_posteriors = np.empty((self.n_speakers, nele, nazi), dtype=float)
        frame_debug = []
        for k in range(self.n_speakers):
            m_idx = assignment[k]
            p_folded = self._normalized_likelihood_folded(likelihood_maps_t[m_idx])
            if not initialized:
                belief_folded = p_folded
                updated_pos = measurements[m_idx].copy()  # mirrors TwoSpeakerTracker's frame-0 convention
                w_used = np.nan
                desc_used = {"size_norm": np.nan, "span_norm": np.nan, "var_norm": np.nan}
            else:
                w_used = slot_w[m_idx]
                desc_used = slot_desc[m_idx]
                belief_folded = (1.0 - w_used) * pred_folded[k] + w_used * p_folded
                bsum = belief_folded.sum()
                belief_folded = (belief_folded / bsum if bsum > _EPS
                                 else np.full(belief_folded.shape, 1.0 / belief_folded.size))
                updated_pos = self._argmax_position(belief_folded)
            belief = unfold_azimuth_endpoint(belief_folded)
            self._tracks[k]["position"] = updated_pos
            self._tracks[k]["belief"] = belief
            self._tracks[k]["history"].append(updated_pos.copy())
            updated_positions[k] = updated_pos
            frame_posteriors[k] = belief
            frame_debug.append({"w": w_used, **desc_used})

        # TRACK-ordered diagnostics. A/S/V are the assigned slot's descriptors
        # at EVERY frame (incl. frame 0); w is the weight actually used in
        # fusion (NaN on frame 0, where there is no fusion).
        track_diag = {
            "track_assigned_slot": np.array(assignment, dtype=int),
            "track_A": np.array([slot_desc[assignment[k]]["size_norm"] for k in range(2)]),
            "track_S": np.array([slot_desc[assignment[k]]["span_norm"] for k in range(2)]),
            "track_V": np.array([slot_desc[assignment[k]]["var_norm"] for k in range(2)]),
            "track_w": np.array([d["w"] for d in frame_debug], dtype=float),
            "track_pred_position": (np.stack(pred_positions) if initialized
                                    else np.full((2, 2), np.nan)),
            "assoc_cost_identity_deg": costs["cost_identity"],
            "assoc_cost_swapped_deg": costs["cost_swapped"],
        }

        self._assignments_history.append(assignment)
        self._cp_features_history.append(cp_features_t)
        self._posterior_history.append(frame_posteriors)
        self._debug_history.append(frame_debug)
        self._track_diag_history.append(track_diag)

        return {
            "positions": updated_positions,
            "assignment": assignment,
            "cp_features": cp_features_t,
            "posteriors": frame_posteriors,
            "debug": frame_debug,
            "track_diag": track_diag,
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
            "tracks"         : list of np.ndarray, shape (T, 2) — one per TRACK
            "assignments"    : np.ndarray, shape (T, 2) — [t, k] = IDL slot given to track k
            "cp_features"    : list of list of dict, shape (T, 2) — IDL-SLOT order
                               (NOT reordered by association; same object as
                               "slot_cp_features")
            "posterior_maps" : np.ndarray, shape (T, 2, nele, nazi) — belief maps b_t,k (TRACK order)
            "debug"          : dict with extra diagnostic information, including
                                per-frame/per-TRACK "w", "size_norm", "span_norm",
                                "var_norm" (each np.ndarray, shape (T, 2); NaN at frame 0)

            Explicitly named additions (TRACK order = [t, k] refers to persistent track k):
            "slot_cp_features"        : alias of "cp_features" (IDL-slot order)
            "track_assigned_slot"     : (T, 2) int  — IDL slot whose package track k received
            "track_A"                 : (T, 2) — normalized CP area of that slot (= size_norm)
            "track_S"                 : (T, 2) — normalized CP span of that slot (= span_norm)
            "track_V"                 : (T, 2) — normalized CP spatial variance (= var_norm)
            "track_w"                 : (T, 2) — fusion weight used (NaN at frame 0)
            "track_pred_position"     : (T, 2, 2) — predicted [el_idx, az_idx] used for
                                        association (NaN at frame 0)
            "assoc_cost_identity_deg" : (T,) — total great-circle cost of slot0->track0, slot1->track1
            "assoc_cost_swapped_deg"  : (T,) — total great-circle cost of slot1->track0, slot0->track1
            "assoc_cost_gap_deg"      : (T,) — |identity - swapped| (NaN at frame 0)
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

        diag = {key: np.stack([np.asarray(fr[key]) for fr in self._track_diag_history], axis=0)
                for key in self._track_diag_history[0]}
        diag["assoc_cost_gap_deg"] = np.abs(diag["assoc_cost_identity_deg"]
                                            - diag["assoc_cost_swapped_deg"])

        return {
            "tracks": tracks,
            "assignments": np.array(self._assignments_history),          # (T, 2)
            "cp_features": self._cp_features_history,                    # (T, 2, dict) SLOT order
            "slot_cp_features": self._cp_features_history,               # explicit alias
            "posterior_maps": np.stack(self._posterior_history, axis=0),  # (T, 2, nele, nazi)
            "debug": {
                "n_frames": T,
                "grid_shape": likelihood_maps.shape[2:],
                **debug_arrays,
            },
            **diag,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalized_likelihood_folded(likelihood_map):
        """p_t,m: full likelihood map of one slot, clipped at 0, folded onto the
        nazi-1 distinct azimuths (duplicate +-pi columns AVERAGED -- two
        evaluations of one direction), normalised to sum to 1. Never reduced
        to a peak."""
        p = fold_azimuth_endpoint(np.clip(np.asarray(likelihood_map, dtype=float), 0.0, None),
                                  kind="likelihood")
        psum = p.sum()
        if psum > _EPS:
            return p / psum
        # Degenerate all-zero/negative map: uniform rather than NaN.
        return np.full(p.shape, 1.0 / p.size)

    @staticmethod
    def _argmax_position(belief_folded):
        """argmax of a folded belief -> [el_idx, az_idx] (float). Folded column j
        is original column j, so the index is valid on the (nele, nazi) grid;
        the +-pi direction is reported as column 0."""
        peak_idx = np.unravel_index(np.argmax(belief_folded), belief_folded.shape)
        return np.array([float(peak_idx[0]), float(peak_idx[1])])
