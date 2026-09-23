"""
Minimal Localized Conformal Prediction (LCP) for the 2-speaker case, speaker-wise.

Scope (first experiment only -- see /home/mayavb/.claude/plans/ok-we-need-to-lazy-pumpkin.md
for the full design discussion that led here):
  - 2-D raw context per (frame, speaker): X = [raw_mean, raw_peak_to_background],
    computed from the RAW likelihood map (before Code.utilities.normalize()).
  - Speakers are calibrated/weighted completely separately -- never concatenated.
  - Feature standardization is fit on calibration data only, then applied to test.
  - RBF kernel on standardized context: H(Xj,Xi) = exp(-||Xj-Xi||^2 / (2*h^2)).
  - This implements Algorithm 7.8's localized weighting + recalibration exactly
    (not a weighted-CRC approximation of the existing global procedure).

Key simplification, confirmed before implementing (see plan file): because the
SRP-DNN score/likelihood-map function is fixed (never refit on a hypothesized
augmented dataset), the calibration scores S_i and the calibration/test context
features X_i do NOT depend on the candidate spatial label y. Only the test
point's own score S_{n+1}(y) is candidate-dependent. So:
  - calibration needs one scalar score S_i = -lambda_star_i per (frame, speaker)
    (lambda_star_i = the tightest existing-global-CP threshold that still covers
    that calibration frame's true DOA -- computed independently here, reusing
    CoverageSet.neighbours_coverage_set() read-only; crc_ssl.py is not modified).
  - the test point needs one FULL score map S_{n+1}(y) = -Lambda(y) over every
    grid cell y, computed via a widest-path (maximum-bottleneck-path) sweep from
    the estimated-position seed.
  - kernel weights are computed ONCE per test point (against the fixed calib
    pool + itself), not recomputed per candidate y.

binary_fill_holes caveat: Code.crc_ssl.CoverageSet.neighbours_coverage_set()
calls scipy's binary_fill_holes() after the BFS region-growing, which can pull
in a fully-enclosed low-value interior cell regardless of whether it's reachable
via a >=threshold path from the seed. widest_path_score_map() below does NOT
reproduce this -- it is the pure widest-path/maximum-bottleneck-path value, with
no post-hoc hole filling. A diagnostic (8 real frames x 2 speakers x 5 lambda
values = 80 checks, on npz_output_tracking_burst_test) found this causes 1168
differing cells out of 216,080 checked (0.54% overall, growing to ~12% at very
permissive lambda=0.1 on one map), and in EVERY one of the 80 checks the
differences were exclusively hole-filled interior cells (never a boundary/
reachability discrepancy). This LCP module deliberately uses the pure
widest-path score for this first experiment and does not attempt to reproduce
hole-filling in the score map yet.
"""

import numpy as np

from Code.crc_ssl import CoverageSet
from Code.plots import plot_roi_neighbours
from Code.utilities import normalize
from Code.two_speaker_tracking.npz_adapter import radians_to_grid_index


# ---------------------------------------------------------------------------
# Context features (raw, pre-normalization)
# ---------------------------------------------------------------------------

def compute_raw_context_features(likelihood_maps_raw, eps=1e-8):
    """X = [raw_mean, raw_peak_to_background] from RAW (not normalize()'d) maps.

    Parameters
    ----------
    likelihood_maps_raw : array-like, shape (..., nele, nazi)
        Any leading shape (e.g. (T,K) or (n_calib,)); operates on the last two
        axes only. Must be the raw likelihood map, i.e. read BEFORE any call to
        Code.utilities.normalize().
    eps : float
        Numerical safety floor for the peak/background ratio's denominator.

    Returns
    -------
    np.ndarray, shape (..., 2) -- [raw_mean, raw_peak_to_background]
    """
    arr = np.asarray(likelihood_maps_raw, dtype=float)
    flat = arr.reshape(arr.shape[:-2] + (-1,))
    raw_mean = flat.mean(axis=-1)
    raw_max = flat.max(axis=-1)
    raw_median = np.median(flat, axis=-1)
    raw_ptb = raw_max / (raw_median + eps)
    return np.stack([raw_mean, raw_ptb], axis=-1)


# ---------------------------------------------------------------------------
# Configurable named context features (static-feature ablation)
#
# Generalizes compute_raw_context_features to an arbitrary named subset of
# raw-map statistics, so alternative context vectors (e.g. for a feature
# ablation) can be selected without duplicating the LCP pipeline. The rest of
# the module (fit_standardizer, standardize_features, rbf_kernel_weight_matrix,
# localized_cp_decision, ...) already operates on X of any dimension, so no
# other function needs to change to support a different feature set.
#
# Numerical definitions match analyze_lcp_feature_diagnostic_v2.py exactly
# (including its eps=0.02 default for ratio/log denominators, chosen there to
# stay positive against the ~0.15% of records with a near/below-zero raw
# mean/median -- see that module's docstring). compute_raw_context_features
# above is untouched and keeps its own eps=1e-8 default for backward
# compatibility with existing experiment scripts.
# ---------------------------------------------------------------------------

FEATURE_REGISTRY = {
    "raw_mean": lambda flat, eps: flat.mean(axis=-1),
    "raw_median": lambda flat, eps: np.median(flat, axis=-1),
    "raw_peak": lambda flat, eps: flat.max(axis=-1),
    "raw_std": lambda flat, eps: flat.std(axis=-1),
    "raw_peak_to_background": lambda flat, eps: flat.max(axis=-1) / (np.median(flat, axis=-1) + eps),
    "median_over_mean": lambda flat, eps: np.median(flat, axis=-1) / (flat.mean(axis=-1) + eps),
    "log_peak_over_median": lambda flat, eps: np.log(
        (flat.max(axis=-1) + eps) / (np.median(flat, axis=-1) + eps)),
}

# Named feature-set configurations used by the static-feature ablation
# (eval_lcp_feature_ablation.py). "original" reproduces
# compute_raw_context_features's own [raw_mean, raw_peak_to_background] via
# the registry (same formula, same eps=1e-8) purely so all four configs can
# share one code path; compute_raw_context_features itself remains the
# canonical/unchanged implementation used elsewhere.
FEATURE_SETS = {
    "original": (["raw_mean", "raw_peak_to_background"], 1e-8),
    "F2": (["raw_median", "raw_std"], 0.02),
    "F3": (["raw_median", "raw_std", "median_over_mean"], 0.02),
    "F4": (["raw_median", "raw_std", "median_over_mean", "log_peak_over_median"], 0.02),
}


def compute_context_features(likelihood_maps_raw, feature_names, eps):
    """X = [feature_names...] from RAW (not normalize()'d) maps, computed via
    FEATURE_REGISTRY. General, named-feature counterpart of
    compute_raw_context_features.

    Parameters
    ----------
    likelihood_maps_raw : array-like, shape (..., nele, nazi)
    feature_names : list of str, keys into FEATURE_REGISTRY, in output order
    eps : float, denominator floor shared by every ratio/log feature in this call

    Returns
    -------
    np.ndarray, shape (..., len(feature_names))
    """
    arr = np.asarray(likelihood_maps_raw, dtype=float)
    flat = arr.reshape(arr.shape[:-2] + (-1,))
    cols = [FEATURE_REGISTRY[name](flat, eps) for name in feature_names]
    return np.stack(cols, axis=-1)


# ---------------------------------------------------------------------------
# Calibration-only feature standardization
# ---------------------------------------------------------------------------

def fit_standardizer(X_calib, eps=1e-12):
    """X_calib: (n,2) for ONE speaker. Returns (mean, std), each shape (2,)."""
    X_calib = np.asarray(X_calib, dtype=float)
    mean = X_calib.mean(axis=0)
    std = X_calib.std(axis=0)
    std = np.where(std < eps, 1.0, std)  # avoid div-by-zero on a degenerate feature
    return mean, std


def standardize_features(X, mean, std):
    return (np.asarray(X, dtype=float) - mean) / std


# ---------------------------------------------------------------------------
# RBF kernel weights (Algorithm 7.8's w_{i,j})
# ---------------------------------------------------------------------------

def rbf_kernel_weight_matrix(X_pool_std, bandwidth):
    """Pairwise localized weights over a pool of m examples (already standardized).

    w_{i,j} = H(X_j, X_i) / sum_j' H(X_j', X_i),  H(Xj,Xi) = exp(-||Xj-Xi||^2 / (2 h^2))

    Parameters
    ----------
    X_pool_std : array-like, shape (m, 2)
    bandwidth : float, h

    Returns
    -------
    np.ndarray, shape (m, m) -- W[i, j] = w_{i,j}. Each row sums to 1.
    """
    X_pool_std = np.asarray(X_pool_std, dtype=float)
    diff = X_pool_std[:, None, :] - X_pool_std[None, :, :]
    d2 = np.sum(diff ** 2, axis=-1)
    H = np.exp(-d2 / (2.0 * bandwidth ** 2))
    W = H / H.sum(axis=1, keepdims=True)
    return W


# ---------------------------------------------------------------------------
# Calibration scores: S_i = -lambda_star_i  (scalar per calibration frame/speaker)
# ---------------------------------------------------------------------------

def compute_lambda_star(norm_map, seed, true_grid_idx, lambda_list_desc):
    """The tightest (largest) lambda in lambda_list_desc (sorted descending) at
    which neighbours_coverage_set(norm_map, lambda, seed) still covers true_grid_idx.
    Falls back to the most permissive tested lambda if never covered (rare --
    only possible if lambda_list's minimum isn't small enough to cover the whole
    grid; normalize()'d maps have min exactly 0, so a lambda_list including 0
    always covers everything)."""
    for lam in lambda_list_desc:
        region = CoverageSet.neighbours_coverage_set(norm_map, float(lam), estimated_position=seed)
        if region[true_grid_idx]:
            return float(lam)
    return float(lambda_list_desc[-1])


def compute_calibration_scores(likelihood_maps_calib_raw, est_pos_calib_rad, true_pos_calib_rad,
                                lambda_list, nele, nazi):
    """S_i = -lambda_star_i for every calibration frame, per estimated-speaker slot.

    Mirrors CoverageSet.calibrate()'s own true<->estimated speaker matching
    (CoverageSet._match_estimated_to_source, reused read-only) so scores line
    up with the same speaker-slot convention the existing global CP uses.

    Parameters
    ----------
    likelihood_maps_calib_raw : (n_calib, K, nele, nazi), raw
    est_pos_calib_rad, true_pos_calib_rad : (n_calib, K, 2), radians
    lambda_list : array-like, the same lambda grid used for global calibration
    nele, nazi : int

    Returns
    -------
    np.ndarray, shape (n_calib, K) -- S_i per calibration frame, indexed by
    estimated-speaker slot (NaN should not occur; every frame has K matched pairs).
    """
    lambda_list_desc = np.sort(np.asarray(lambda_list, dtype=float))[::-1]
    n_calib, K = likelihood_maps_calib_raw.shape[:2]
    S = np.full((n_calib, K), np.nan, dtype=float)
    for i in range(n_calib):
        true_order, est_order = CoverageSet._match_estimated_to_source(
            true_pos_calib_rad[i], est_pos_calib_rad[i])
        for true_s, est_s in zip(true_order, est_order):
            norm_map = normalize(likelihood_maps_calib_raw[i, est_s])
            seed = tuple(radians_to_grid_index(est_pos_calib_rad[i, est_s], nele, nazi).astype(int))
            true_idx = tuple(radians_to_grid_index(true_pos_calib_rad[i, true_s], nele, nazi).astype(int))
            lam_star = compute_lambda_star(norm_map, seed, true_idx, lambda_list_desc)
            S[i, est_s] = -lam_star
    return S


def calibrate_global_lambda_from_arrays(likelihood_maps_calib_raw, est_pos_calib_rad, true_pos_calib_rad,
                                         room, lambda_list, err):
    """Same computation as npz_adapter.calibrate_lambda_thresholds, but from
    already-sliced arrays instead of a whole-file path. Needed for a mixed
    (clean+burst) calibration pool that is a TRAJECTORY-LEVEL SUBSET of a file
    that also contains the test pool -- the path-based helper always samples
    from an entire file and can't express that split. Does not modify
    Code.crc_ssl.CoverageSet or npz_adapter.calibrate_lambda_thresholds; reuses
    them read-only, identically to how the path-based helper itself works.

    Parameters
    ----------
    likelihood_maps_calib_raw : (n_calib, K, nele, nazi), raw
    est_pos_calib_rad, true_pos_calib_rad : (n_calib, K, 2), radians
    room : object with .xl/.yl grids (see npz_adapter._build_room)
    lambda_list : array-like
    err : float, target significance level (alpha)

    Returns
    -------
    np.ndarray, shape (K,) -- global lambda threshold per speaker.
    """
    lambda_list = np.asarray(lambda_list, dtype=float)
    cov = CoverageSet(
        true_position=true_pos_calib_rad, estimated_positions=est_pos_calib_rad,
        likelihood_maps=likelihood_maps_calib_raw, lambda_list=lambda_list,
        room=room, path_=None, plot_function=plot_roi_neighbours,
    )
    cov.calibrate(plot=False, plot_coverage_set=False)
    return cov._calc_conformal_risk_control(float(err))


# ---------------------------------------------------------------------------
# Test score MAP: S_{n+1}(y) = -Lambda(y), via widest-path sweep
# ---------------------------------------------------------------------------

def widest_path_score_map(norm_map, seed):
    """Lambda(y) for every grid cell y: the maximum-bottleneck-path value from
    seed to y, matching neighbours_coverage_set's BFS ">=threshold, 4-connected"
    semantics EXCEPT for the post-hoc binary_fill_holes step (see module
    docstring -- 0.54% of cells differ overall in a real-data diagnostic,
    exclusively hole-filled interior cells; not reproduced here).

    The seed itself is unconditionally included in neighbours_coverage_set at
    any threshold (coverage_set[estimated_position] = True before any
    threshold check), so Lambda(seed) = +inf here.

    Parameters
    ----------
    norm_map : (nele, nazi), already normalize()'d (matches how the existing
        global CP and this module's calibration scores use it)
    seed : tuple (i, j)

    Returns
    -------
    np.ndarray, shape (nele, nazi), dtype float -- Lambda(y). S_{n+1}(y) = -Lambda(y).
    """
    import heapq

    nele, nazi = norm_map.shape
    Lambda = np.full((nele, nazi), -np.inf, dtype=float)
    Lambda[seed] = np.inf
    finalized = np.zeros((nele, nazi), dtype=bool)
    finalized[seed] = True
    heap = []  # max-heap via negation: (-bottleneck_value, i, j)
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni, nj = seed[0] + di, seed[1] + dj
        if 0 <= ni < nele and 0 <= nj < nazi:
            val = float(norm_map[ni, nj])
            if val > Lambda[ni, nj]:
                Lambda[ni, nj] = val
                heapq.heappush(heap, (-val, ni, nj))
    while heap:
        neg_val, i, j = heapq.heappop(heap)
        if finalized[i, j]:
            continue
        finalized[i, j] = True
        cur_lambda = -neg_val
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < nele and 0 <= nj < nazi and not finalized[ni, nj]:
                candidate = min(cur_lambda, float(norm_map[ni, nj]))
                if candidate > Lambda[ni, nj]:
                    Lambda[ni, nj] = candidate
                    heapq.heappush(heap, (-candidate, ni, nj))
    return Lambda


# ---------------------------------------------------------------------------
# Algorithm 7.8: localized statistic + recalibration -> localized CP region
# ---------------------------------------------------------------------------

def localized_cp_decision(S_calib_speaker, X_calib_std_speaker, S_test_map, X_test_std,
                           bandwidth, alpha):
    """Build the localized CP region for one test (frame, speaker) via Algorithm 7.8.

    For i in {1..n+1} (calibration pool + this test point), weights w_{i,j} are
    computed once (candidate-independent, since context features don't depend
    on the hypothesized label y). Only the test point's own score S_{n+1}(y)
    varies with the candidate y, so per candidate y:
        S^y = [S_1, ..., S_n, S_{n+1}(y)]
        tilde_S_i^y = sum_j w_{i,j} * 1[S_j^y < S_i^y]   for i = 1..n+1
        tilde_q^y   = Quantile(tilde_S_1^y, ..., tilde_S_{n+1}^y; 1-alpha)
        include y   iff  tilde_S_{n+1}^y <= tilde_q^y
    The outer quantile is a plain (unweighted) empirical quantile over the n+1
    already-localized-rank statistics -- the localization weighting is already
    absorbed into each tilde_S_i^y itself, matching Algorithm 7.8's own
    two-step structure (do not further reweight this quantile).

    Parameters
    ----------
    S_calib_speaker : (n,) fixed calibration scores S_i for this speaker
    X_calib_std_speaker : (n,2) standardized calibration contexts for this speaker
    S_test_map : (nele,nazi) test score map S_{n+1}(y) (from widest_path_score_map,
        negated)
    X_test_std : (2,) standardized test context (candidate-independent)
    bandwidth : float, h
    alpha : float, target miscoverage rate

    Returns
    -------
    region : (nele,nazi) bool -- the localized CP region
    tildeS_test_map : (nele,nazi) float -- tilde_S_{n+1}^y
    q_map : (nele,nazi) float -- tilde_q^y
    weights_test_row : (n+1,) float -- w_{n+1,j} for j=1..n (calib) then j=n+1 (self, always 0-effect)
    """
    S_calib_speaker = np.asarray(S_calib_speaker, dtype=float)
    n = S_calib_speaker.shape[0]
    nele, nazi = S_test_map.shape
    S_map_flat = S_test_map.reshape(-1)

    X_pool = np.vstack([X_calib_std_speaker, np.asarray(X_test_std, dtype=float)[None, :]])  # (n+1,2)
    W = rbf_kernel_weight_matrix(X_pool, bandwidth)  # (n+1,n+1); test point is row/col index n

    # tilde_S_i^y for calibration i (i < n): fixed base term (both scores fixed)
    # plus one candidate-dependent term (comparison against the test point's S^y).
    lt_calib_calib = (S_calib_speaker[None, :] < S_calib_speaker[:, None]).astype(float)  # (n,n) [i,j]
    base = (W[:n, :n] * lt_calib_calib).sum(axis=1)  # (n,)
    w_i_test = W[:n, n]  # (n,) weight each calib i gives to the test point
    lt_test_lt_calib_i = (S_map_flat[None, :] < S_calib_speaker[:, None]).astype(float)  # (n, m)
    tildeS_calib = base[:, None] + w_i_test[:, None] * lt_test_lt_calib_i  # (n, m)

    # tilde_S_{n+1}^y for the test point itself: entirely candidate-dependent,
    # since S_{n+1}^y varies with y (the self-comparison term j=n+1 contributes 0).
    w_test_j = W[n, :n]  # (n,) weights the test point gives to each calib point
    lt_calib_lt_test = (S_calib_speaker[:, None] < S_map_flat[None, :]).astype(float)  # (n, m)
    tildeS_test = (w_test_j[:, None] * lt_calib_lt_test).sum(axis=0)  # (m,)

    all_tildeS = np.vstack([tildeS_calib, tildeS_test[None, :]])  # (n+1, m)
    q = np.quantile(all_tildeS, 1.0 - alpha, axis=0)  # (m,)

    region_flat = tildeS_test <= q
    region = region_flat.reshape(nele, nazi)
    tildeS_test_map = tildeS_test.reshape(nele, nazi)
    q_map = q.reshape(nele, nazi)
    return region, tildeS_test_map, q_map, W[n, :]


# ---------------------------------------------------------------------------
# Calibration-only RBF bandwidth selection via leave-one-out
# ---------------------------------------------------------------------------

def select_bandwidth_via_loo(likelihood_maps_calib_raw, est_pos_calib_rad, true_pos_calib_rad,
                              S_calib, X_calib_std, speaker_k, h_grid, alpha, nele, nazi):
    """Select the RBF bandwidth h for speaker slot `speaker_k` using ONLY the
    calibration pool (no test-half data, no burst_active label -- this
    function never reads burst_active).

    For every calibration frame i matched to speaker_k (same true<->estimated
    matching convention as compute_calibration_scores), i is temporarily
    treated as a held-out "test" point: its own widest-path score map
    S_i(y) = -Lambda_i(y) is computed once from its own raw map + estimated
    seed (this step does not depend on h, so it is done once and reused
    across every h in h_grid). For each h, localized_cp_decision is then run
    with the OTHER n-1 calibration points as the pool, giving a leave-one-out
    coverage decision + region area for point i at that h. Averaging over all
    i gives, per h, a LOO calibration coverage and mean area.

    Selection rule (exactly as specified): among h with LOO coverage >=
    1-alpha, pick the one with the smallest mean area; if none reach the
    target coverage, pick the highest-coverage h, tie-broken by smaller area.

    Parameters
    ----------
    likelihood_maps_calib_raw : (n_calib, K, nele, nazi), raw
    est_pos_calib_rad, true_pos_calib_rad : (n_calib, K, 2), radians
    S_calib : (n_calib, K) fixed calibration scores (compute_calibration_scores output)
    X_calib_std : (n_calib, 2) standardized calibration contexts for speaker_k
        (already standardized with that speaker's own calibration mean/std)
    speaker_k : int, which estimated-speaker slot (0 or 1)
    h_grid : iterable of float, candidate bandwidths
    alpha : float, target miscoverage rate
    nele, nazi : int

    Returns
    -------
    selected_h : float
    per_h_stats : dict h -> {"coverage": float, "mean_area": float, "n": int}
    """
    n_calib = likelihood_maps_calib_raw.shape[0]
    S_calib = np.asarray(S_calib, dtype=float)
    X_calib_std = np.asarray(X_calib_std, dtype=float)

    lambda_maps = []
    true_idxs = []
    valid_i = []
    for i in range(n_calib):
        true_order, est_order = CoverageSet._match_estimated_to_source(
            true_pos_calib_rad[i], est_pos_calib_rad[i])
        match_true_s = None
        for true_s, est_s in zip(true_order, est_order):
            if int(est_s) == speaker_k:
                match_true_s = int(true_s)
                break
        if match_true_s is None:
            continue
        raw_map = likelihood_maps_calib_raw[i, speaker_k]
        norm_map = normalize(raw_map)
        seed = tuple(radians_to_grid_index(est_pos_calib_rad[i, speaker_k], nele, nazi).astype(int))
        true_idx = tuple(radians_to_grid_index(true_pos_calib_rad[i, match_true_s], nele, nazi).astype(int))
        lambda_maps.append(widest_path_score_map(norm_map, seed))
        true_idxs.append(true_idx)
        valid_i.append(i)

    valid_i = np.asarray(valid_i)
    S_valid = S_calib[valid_i, speaker_k]
    X_valid = X_calib_std[valid_i]
    n = len(valid_i)

    per_h_stats = {}
    for h in h_grid:
        covered = np.zeros(n, dtype=bool)
        areas = np.zeros(n, dtype=float)
        for idx in range(n):
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            S_test_map = -lambda_maps[idx]
            region, _, _, _ = localized_cp_decision(
                S_valid[mask], X_valid[mask], S_test_map, X_valid[idx], h, alpha)
            covered[idx] = bool(region[true_idxs[idx]])
            areas[idx] = float(region.sum())
        per_h_stats[float(h)] = dict(coverage=float(covered.mean()), mean_area=float(areas.mean()), n=n)

    target = 1.0 - alpha
    reaching = [h for h in per_h_stats if per_h_stats[h]["coverage"] >= target]
    if reaching:
        selected_h = min(reaching, key=lambda h: per_h_stats[h]["mean_area"])
    else:
        selected_h = min(per_h_stats, key=lambda h: (-per_h_stats[h]["coverage"], per_h_stats[h]["mean_area"]))
    return selected_h, per_h_stats


# ---------------------------------------------------------------------------
# Tiny synthetic/unit checks (no npz data needed)
# ---------------------------------------------------------------------------

def run_unit_checks(verbose=True):
    """Sanity checks requested before running the real experiment:
      1. RBF weights sum to 1.
      2. Identical context points get the largest weight.
      3. S = -Lambda has the expected ordering (closer/higher-likelihood cells
         get larger Lambda, hence smaller/more-negative S).
      4. With a very large bandwidth, weights approach uniform 1/(n+1).
    Raises AssertionError on failure. Returns a dict of the numeric evidence
    for each check (for printing/inspection).
    """
    rng = np.random.default_rng(0)
    evidence = {}

    # --- 1 & 2: RBF weight matrix properties ---
    X_pool = rng.normal(size=(6, 2))
    X_pool[3] = X_pool[0]  # make index 3 an exact duplicate of index 0
    W = rbf_kernel_weight_matrix(X_pool, bandwidth=1.0)
    row_sums = W.sum(axis=1)
    assert np.allclose(row_sums, 1.0), f"RBF weight rows must sum to 1, got {row_sums}"
    evidence["rbf_row_sums"] = row_sums.tolist()

    row0 = W[0].copy()
    row0_self_excluded = row0.copy()
    row0_self_excluded[0] = -1  # exclude self-weight from the "largest" check
    largest_other = np.argmax(row0_self_excluded)
    assert largest_other == 3, (
        f"Identical context point (index 3, a copy of index 0) should get the "
        f"largest non-self weight in row 0, got argmax={largest_other}, row={row0}")
    evidence["duplicate_gets_largest_weight"] = {"row0": row0.tolist(), "argmax_excl_self": int(largest_other)}

    # --- 3: S = -Lambda ordering, on a synthetic single-peak map ---
    nele, nazi = 15, 21
    ele_axis = np.arange(nele)[:, None]
    azi_axis = np.arange(nazi)[None, :]
    seed = (7, 10)
    dist = np.sqrt((ele_axis - seed[0]) ** 2 + (azi_axis - seed[1]) ** 2)
    synth_map = np.exp(-0.5 * (dist / 3.0) ** 2)  # smooth single peak at seed, decaying outward
    Lambda = widest_path_score_map(synth_map, seed)
    S = -Lambda
    near_cell = (7, 11)   # 1 step from seed, high value
    far_cell = (0, 0)     # far corner, low value
    assert Lambda[near_cell] > Lambda[far_cell], (
        f"Expected Lambda(near) > Lambda(far), got {Lambda[near_cell]} vs {Lambda[far_cell]}")
    assert S[near_cell] < S[far_cell], (
        f"Expected S(near) < S(far) since S=-Lambda, got {S[near_cell]} vs {S[far_cell]}")
    evidence["ordering_check"] = {
        "Lambda_near": float(Lambda[near_cell]), "Lambda_far": float(Lambda[far_cell]),
        "S_near": float(S[near_cell]), "S_far": float(S[far_cell]),
    }

    # --- 4: very large bandwidth -> uniform weights ---
    X_pool2 = rng.normal(size=(8, 2)) * 5.0  # spread the points out
    W_huge_h = rbf_kernel_weight_matrix(X_pool2, bandwidth=1e6)
    uniform = np.full(X_pool2.shape[0], 1.0 / X_pool2.shape[0])
    max_dev = np.max(np.abs(W_huge_h - uniform[None, :]))
    assert max_dev < 1e-6, f"Expected near-uniform weights at huge bandwidth, max deviation {max_dev}"
    evidence["large_bandwidth_max_deviation_from_uniform"] = float(max_dev)

    if verbose:
        print("[LCP unit checks] ALL PASSED")
        for k, v in evidence.items():
            print(f"    {k}: {v}")
    return evidence
