"""
Extract uncertainty features from a speaker's CP (conformal prediction) region.

Philosophy
----------
The CP region produced by CoverageSet is NOT used as a hard mask that either
includes or excludes a measurement. Instead, scalar features of the region
(area, width, entropy, ...) are used to scale measurement trust vs. motion
prediction trust inside the tracker update step.

A small region  → high confidence in the CP set → trust the measurement more.
A large region  → high uncertainty              → lean more on the prediction.
"""

import numpy as np


# -----------------------------------------------------------------------
# Private helpers
# -----------------------------------------------------------------------

def _safe_entropy(probabilities):
    """Shannon entropy of a probability vector, in nats.

    Parameters
    ----------
    probabilities : np.ndarray, 1-D
        Non-negative values that already sum (approximately) to 1.
        Zeros are handled safely via epsilon clipping.

    Returns
    -------
    float
        H = -sum(p * log(p)).  Returns 0.0 for a deterministic distribution
        and log(N) for a uniform distribution over N elements.
    """
    p = np.asarray(probabilities, dtype=float)
    p = np.clip(p, 1e-12, None)   # avoid log(0)
    p = p / p.sum()                # re-normalise after clipping
    return float(-np.sum(p * np.log(p)))


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def _periodic_azimuth_cells(rows, cols, nazi):
    """Map active cells onto the nazi-1 distinct azimuths (col nazi-1 == col 0),
    de-duplicate, and unwrap columns along the shortest circular arc that covers
    all occupied columns.

    Returns (rows, unwrapped_cols, arc_width_in_cells). The arc starts right
    after the largest circular run of empty columns, so a compact region
    straddling +-pi gets the same span/variance as the same region elsewhere.
    """
    n_az = nazi - 1
    cells = np.unique(np.stack([rows, np.asarray(cols) % n_az], axis=1), axis=0)
    r, c = cells[:, 0], cells[:, 1]
    occ = np.unique(c)
    if len(occ) == n_az:
        start, width = 0, n_az
    else:
        gaps = np.diff(np.concatenate([occ, [occ[0] + n_az]])) - 1  # empty cols after each occupied col
        i = int(np.argmax(gaps))
        start, width = int(occ[(i + 1) % len(occ)]), int(n_az - gaps[i])
    return r, (c - start) % n_az, float(width)


def extract_cp_features(cp_region, likelihood_map=None, estimated_position=None):
    """Extract scalar uncertainty features from one speaker's CP region.

    Parameters
    ----------
    cp_region : np.ndarray, shape (nele, nazi)
        Boolean or numeric mask of the conformal prediction region.
        Any cell with value > 0 is treated as active.
    likelihood_map : np.ndarray, shape (nele, nazi), optional
        Raw likelihood map for the same speaker.  Used for entropy, peak, and
        mass features.  NaNs are replaced with 0 before any computation.
    estimated_position : array-like, shape (2,), optional
        [el_idx, az_idx] of the current measurement in grid-cell indices,
        consistent with TwoSpeakerTracker's internal convention (NOT radians
        — see npz_adapter.radians_to_grid_index for the radians->grid-index
        boundary conversion). Currently unused by the feature computations
        themselves; reserved for future geometry features (e.g. distance from
        estimate to CP centroid), which must NOT assume radians here.

    Returns
    -------
    dict with keys
        cp_area : int
            Number of active cells in the CP region.

        cp_area_norm : float
            cp_area divided by the total number of grid cells (nele * nazi).
            In [0, 1].  Resolution-invariant proxy for region size.

        cp_width_el : float
            Bounding-box span of active cells along the elevation axis
            (row dimension), measured in grid cells.
            Computed as  max(row) - min(row) + 1  over all active cells.
            np.nan if the CP region is empty.

        cp_width_az : float
            Same as cp_width_el but along the azimuth axis (column dimension).
            np.nan if the CP region is empty.

        cp_width_total : float
            Diagonal of the elevation-azimuth bounding box:
                sqrt(cp_width_el**2 + cp_width_az**2).
            np.nan if the CP region is empty.

        cp_var : float
            Spatial-dispersion descriptor: population variance (ddof=0) of
            the active cells' row indices plus that of their column indices,
            i.e. Var(rows) + Var(cols). Equivalently the mean squared
            Euclidean distance of active cells from their 2-D centroid.
            Units: grid cells squared. 0.0 for a single active cell.
            np.nan if the CP region is empty.
            NOTE: this is a new 2-D spatial-dispersion feature for this
            tracker, not a generalization of any 1-D "gap variance" feature.

        cp_width_az_periodic, cp_width_total_periodic, cp_var_periodic : float
            Azimuth-periodic versions of cp_width_az / cp_width_total / cp_var
            (see _periodic_azimuth_cells): cells on the nazi-1 distinct
            azimuths, span = shortest covering circular arc, variance after
            unwrapping along it. Elevation unchanged. Identical to the old
            values for regions that don't touch the +-pi seam. These are what
            cp_weight.compute_cp_weight uses for S and V. (The old keys are kept
            unchanged because TwoSpeakerTracker reads cp_width_az.)

        cp_entropy : float
            Shannon entropy (nats) of the likelihood distribution restricted
            to the active CP cells.  Likelihood values are normalised to sum
            to 1 before the entropy is computed.
            A peaked distribution inside the CP → low entropy → confident.
            A flat  distribution inside the CP → high entropy → uncertain.
            np.nan if likelihood_map is None or the CP region is empty.

        peak_likelihood : float
            Maximum likelihood value among active CP cells.
            np.nan if likelihood_map is None or the CP region is empty.

        mass_inside_cp : float
            Fraction of the total likelihood mass contained in the CP region:
                sum(likelihood_map[cp_region]) / sum(likelihood_map)
            In [0, 1].  High mass → the CP region captures the main mode.
            np.nan if likelihood_map is None or total likelihood is zero.

        measurement_uncertainty : float
            A single scalar in [0, 1] summarising how uncertain the
            measurement is at this frame.  Currently defined as cp_area_norm.
            A value near 0 → trust the measurement (small CP region).
            A value near 1 → lean on the prediction (large CP region).

            TODO: replace with a learned or principled combination, e.g.
                u = w1 * cp_area_norm + w2 * cp_entropy_norm + ...
    """
    cp = np.asarray(cp_region, dtype=float)
    nele, nazi = cp.shape
    n_total = nele * nazi

    # Boolean mask of active cells.
    active = cp > 0

    # ------------------------------------------------------------------
    # Area features (always computable)
    # ------------------------------------------------------------------
    cp_area = int(active.sum())
    cp_area_norm = cp_area / n_total

    # ------------------------------------------------------------------
    # Bounding-box width features
    # ------------------------------------------------------------------
    if cp_area == 0:
        cp_width_el = np.nan
        cp_width_az = np.nan
        cp_width_total = np.nan
        cp_var = np.nan
    else:
        rows, cols = np.where(active)
        cp_width_el = float(rows.max() - rows.min() + 1)
        cp_width_az = float(cols.max() - cols.min() + 1)
        cp_width_total = float(np.sqrt(cp_width_el ** 2 + cp_width_az ** 2))
        # Spatial-dispersion descriptor: population variance of the active
        # cells' row/col indices, summed over axes (= mean squared Euclidean
        # distance of active cells from their 2-D centroid). This is a NEW
        # 2-D descriptor for this tracker -- not a generalization of the
        # 1-D "variance of consecutive sorted-gap" feature from the earlier
        # 1-D DOA-class tracker paper. 0.0 for a single active cell.
        cp_var = float(np.var(rows) + np.var(cols))

    # ------------------------------------------------------------------
    # Azimuth-periodic span / dispersion (used by cp_weight.compute_cp_weight)
    # ------------------------------------------------------------------
    # The grid's azimuth axis is linspace(-pi, pi, nazi): columns 0 and nazi-1 are
    # the same direction, so there are nazi-1 distinct azimuths on a circle.
    # Cells are mapped to that circle (col nazi-1 -> 0) and de-duplicated; the
    # azimuth span is the shortest circular arc covering all occupied columns,
    # and the azimuth variance is taken after unwrapping the columns along that
    # arc. Elevation is unchanged (not periodic). Units stay grid cells, so for
    # any region that does not touch the +-pi seam these equal the old values.
    if cp_area == 0:
        cp_width_az_periodic = np.nan
        cp_width_total_periodic = np.nan
        cp_var_periodic = np.nan
    else:
        p_rows, p_cols, cp_width_az_periodic = _periodic_azimuth_cells(rows, cols, nazi)
        cp_width_total_periodic = float(np.sqrt(cp_width_el ** 2 + cp_width_az_periodic ** 2))
        cp_var_periodic = float(np.var(p_rows) + np.var(p_cols))

    # ------------------------------------------------------------------
    # Likelihood-based features
    # ------------------------------------------------------------------
    if likelihood_map is None or cp_area == 0:
        cp_entropy = np.nan
        peak_likelihood = np.nan
        mass_inside_cp = np.nan
    else:
        lm = np.nan_to_num(np.asarray(likelihood_map, dtype=float), nan=0.0)

        vals_inside = lm[active]                # 1-D array of likelihood values inside CP
        total_mass = lm.sum()

        peak_likelihood = float(vals_inside.max())

        # Entropy: normalise inside-CP likelihoods to a probability vector.
        inside_sum = vals_inside.sum()
        if inside_sum <= 0:
            cp_entropy = np.nan
        else:
            cp_entropy = _safe_entropy(vals_inside / inside_sum)

        # Mass fraction.
        if total_mass <= 0:
            mass_inside_cp = np.nan
        else:
            mass_inside_cp = float(inside_sum / total_mass)

    # ------------------------------------------------------------------
    # Composite uncertainty scalar
    # ------------------------------------------------------------------
    # TODO: replace with a principled combination once features are validated.
    measurement_uncertainty = cp_area_norm

    return {
        "cp_area": cp_area,
        "cp_area_norm": cp_area_norm,
        "cp_width_el": cp_width_el,
        "cp_width_az": cp_width_az,
        "cp_width_total": cp_width_total,
        "cp_var": cp_var,
        "cp_width_az_periodic": cp_width_az_periodic,
        "cp_width_total_periodic": cp_width_total_periodic,
        "cp_var_periodic": cp_var_periodic,
        "cp_entropy": cp_entropy,
        "peak_likelihood": peak_likelihood,
        "mass_inside_cp": mass_inside_cp,
        "measurement_uncertainty": measurement_uncertainty,
    }
