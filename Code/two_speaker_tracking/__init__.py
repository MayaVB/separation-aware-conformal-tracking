"""
Two-speaker tracker-smoother for the noiseless scenario.

Philosophy (mirrors the single-speaker CRC framework):
- CP regions are NOT hard masks; they quantify uncertainty.
- CP region features control how much each track trusts measurement vs. prediction.
- Conformal prediction sets produced by CoverageSet feed into this module as cp_regions.

Public API
----------
TwoSpeakerTracker   -- baseline tracker class, multiplicative Gaussian-product fusion (tracker.py)
CPWeightedFusionTracker -- CP-weighted full-belief fusion tracker (cp_weighted_tracker.py)
smooth_two_speaker_tracks -- backward smoother (smoother.py)
extract_cp_features  -- uncertainty feature extractor (cp_features.py)
compute_cp_weight    -- CP-aware fusion weight for CPWeightedFusionTracker (cp_weight.py)
associate_two_speakers -- data association (association.py)
"""

from Code.two_speaker_tracking.tracker import TwoSpeakerTracker
from Code.two_speaker_tracking.cp_weighted_tracker import CPWeightedFusionTracker
from Code.two_speaker_tracking.smoother import smooth_two_speaker_tracks
from Code.two_speaker_tracking.cp_features import extract_cp_features
from Code.two_speaker_tracking.cp_weight import compute_cp_weight
from Code.two_speaker_tracking.association import associate_two_speakers

__all__ = [
    "TwoSpeakerTracker",
    "CPWeightedFusionTracker",
    "smooth_two_speaker_tracks",
    "extract_cp_features",
    "compute_cp_weight",
    "associate_two_speakers",
]
