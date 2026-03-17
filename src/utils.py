"""Utility functions for evaluation metrics and statistical analysis."""

import numpy as np
from scipy import stats


def wilson_ci(k, n, z=1.96):
    """Wilson score confidence interval for binomial proportion.

    Args:
        k: Number of successes.
        n: Number of trials.
        z: Z-score for desired confidence level (1.96 for 95% CI).

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound).
    """
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (p, max(0.0, center - margin), min(1.0, center + margin))


def bootstrap_ci(values, stat_fn=np.mean, n_boot=1000, ci=0.95, seed=42):
    """Bootstrap confidence interval for arbitrary statistic.

    Args:
        values: Array of values.
        stat_fn: Statistic function (default: mean).
        n_boot: Number of bootstrap resamples.
        ci: Confidence level.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound).
    """
    rng = np.random.RandomState(seed)
    point = stat_fn(values)
    boot_stats = []
    for _ in range(n_boot):
        idx = rng.choice(len(values), size=len(values), replace=True)
        boot_stats.append(stat_fn(values[idx]))
    alpha = 1 - ci
    lo = np.percentile(boot_stats, 100 * alpha / 2)
    hi = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return (point, lo, hi)


def bca_bootstrap_ci(values, stat_fn=np.mean, n_boot=2000, ci=0.95, seed=42):
    """Bias-corrected and accelerated (BCa) bootstrap confidence interval.

    BCa intervals adjust for both bias and skewness in the bootstrap
    distribution, providing better coverage than percentile intervals
    in small samples.

    Args:
        values: Array of values.
        stat_fn: Statistic function.
        n_boot: Number of bootstrap resamples.
        ci: Confidence level.
        seed: Random seed.

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound).
    """
    values = np.asarray(values)
    n = len(values)
    rng = np.random.RandomState(seed)
    point = stat_fn(values)

    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_stats[i] = stat_fn(values[idx])

    # Bias correction factor z0
    prop_less = np.mean(boot_stats < point)
    prop_less = np.clip(prop_less, 1e-10, 1 - 1e-10)
    z0 = stats.norm.ppf(prop_less)

    # Acceleration factor via jackknife
    jackknife_stats = np.empty(n)
    for i in range(n):
        jack_sample = np.delete(values, i)
        jackknife_stats[i] = stat_fn(jack_sample)
    jack_mean = jackknife_stats.mean()
    num = np.sum((jack_mean - jackknife_stats) ** 3)
    den = 6.0 * (np.sum((jack_mean - jackknife_stats) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0

    alpha_val = 1 - ci
    z_alpha_lo = stats.norm.ppf(alpha_val / 2)
    z_alpha_hi = stats.norm.ppf(1 - alpha_val / 2)

    # Adjusted percentiles
    def adjusted_percentile(z_alpha):
        numerator = z0 + z_alpha
        denominator = 1 - a * numerator
        if denominator == 0:
            return 0.5
        return stats.norm.cdf(z0 + numerator / denominator)

    p_lo = adjusted_percentile(z_alpha_lo)
    p_hi = adjusted_percentile(z_alpha_hi)

    lo = np.percentile(boot_stats, 100 * np.clip(p_lo, 0, 1))
    hi = np.percentile(boot_stats, 100 * np.clip(p_hi, 0, 1))
    return (point, float(lo), float(hi))


def benjamini_hochberg(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR correction for multiple comparisons.

    Args:
        p_values: Array of raw p-values.
        alpha: Target FDR level.

    Returns:
        Dict with:
            rejected: Boolean array of rejected hypotheses.
            q_values: Adjusted p-values (q-values).
            n_rejected: Number of rejected hypotheses.
    """
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    if n == 0:
        return {"rejected": np.array([], dtype=bool), "q_values": np.array([]),
                "n_rejected": 0}

    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # Compute q-values (adjusted p-values)
    q_values = np.empty(n)
    q_values[sorted_idx[-1]] = sorted_p[-1]
    for i in range(n - 2, -1, -1):
        rank = i + 1
        q = sorted_p[i] * n / rank
        q_values[sorted_idx[i]] = min(q, q_values[sorted_idx[i + 1]])

    q_values = np.clip(q_values, 0, 1)
    rejected = q_values < alpha

    return {
        "rejected": rejected,
        "q_values": q_values,
        "n_rejected": int(np.sum(rejected)),
    }


def post_hoc_power(n, p1, p2, alpha=0.05):
    """Post-hoc power for a two-proportion z-test (Fisher exact approximation).

    Args:
        n: Sample size per group.
        p1: Proportion in group 1.
        p2: Proportion in group 2.
        alpha: Significance level.

    Returns:
        Estimated power (float between 0 and 1).
    """
    if n == 0 or p1 == p2:
        return 0.0
    p_bar = (p1 + p2) / 2
    se_null = np.sqrt(2 * p_bar * (1 - p_bar) / n)
    se_alt = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)
    if se_null == 0 or se_alt == 0:
        return 0.0
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_stat = (abs(p1 - p2) - z_alpha * se_null) / se_alt
    return float(stats.norm.cdf(z_stat))


def mcc(tp, tn, fp, fn):
    """Matthews Correlation Coefficient."""
    num = tp * tn - fp * fn
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    if den == 0:
        return 0.0
    return num / den


def detection_metrics(y_true, y_pred, n_bootstrap=1000, seed=42):
    """Compute detection metrics with 95% Wilson CIs and BCa bootstrap for MCC.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.
        n_bootstrap: Number of bootstrap iterations for MCC CI.
        seed: Random seed.

    Returns:
        Dict of metric name -> (point, lower, upper).
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    n_pos = tp + fn
    n_neg = tn + fp

    sens = wilson_ci(tp, n_pos)
    spec = wilson_ci(tn, n_neg)
    ppv_n = tp + fp
    ppv = wilson_ci(tp, ppv_n) if ppv_n > 0 else (0.0, 0.0, 0.0)
    npv_n = tn + fn
    npv = wilson_ci(tn, npv_n) if npv_n > 0 else (0.0, 0.0, 0.0)

    # BCa bootstrap for MCC (implemented directly since bca_bootstrap_ci
    # expects 1D arrays and MCC operates on paired observations)
    mcc_point = mcc(tp, tn, fp, fn)
    n_total = len(y_true)
    rng = np.random.RandomState(seed)

    boot_mccs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n_total, size=n_total, replace=True)
        bt, bp = y_true[idx], y_pred[idx]
        btp = np.sum((bt == 1) & (bp == 1))
        btn = np.sum((bt == 0) & (bp == 0))
        bfp = np.sum((bt == 0) & (bp == 1))
        bfn = np.sum((bt == 1) & (bp == 0))
        boot_mccs[i] = mcc(btp, btn, bfp, bfn)

    # Bias correction factor z0
    prop_less = np.mean(boot_mccs < mcc_point)
    prop_less = np.clip(prop_less, 1e-10, 1 - 1e-10)
    z0 = stats.norm.ppf(prop_less)

    # Acceleration via jackknife
    jack_mccs = np.empty(n_total)
    for i in range(n_total):
        jt = np.delete(y_true, i)
        jp = np.delete(y_pred, i)
        jtp = np.sum((jt == 1) & (jp == 1))
        jtn = np.sum((jt == 0) & (jp == 0))
        jfp = np.sum((jt == 0) & (jp == 1))
        jfn = np.sum((jt == 1) & (jp == 0))
        jack_mccs[i] = mcc(jtp, jtn, jfp, jfn)
    jack_mean = jack_mccs.mean()
    num_a = np.sum((jack_mean - jack_mccs) ** 3)
    den_a = 6.0 * (np.sum((jack_mean - jack_mccs) ** 2) ** 1.5)
    a_hat = num_a / den_a if den_a != 0 else 0.0

    z_lo = stats.norm.ppf(0.025)
    z_hi = stats.norm.ppf(0.975)

    def bca_percentile(z_alpha):
        numer = z0 + z_alpha
        denom = 1 - a_hat * numer
        if denom == 0:
            return 0.5
        return stats.norm.cdf(z0 + numer / denom)

    mcc_lo = float(np.percentile(boot_mccs, 100 * np.clip(bca_percentile(z_lo), 0, 1)))
    mcc_hi = float(np.percentile(boot_mccs, 100 * np.clip(bca_percentile(z_hi), 0, 1)))
    mcc_result = (mcc_point, mcc_lo, mcc_hi)

    return {
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "mcc": mcc_result,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def format_ci(metric_tuple, decimals=3):
    """Format a (point, lower, upper) tuple as 'X.XXX (95% CI X.XXX-X.XXX)'."""
    pt, lo, hi = metric_tuple
    return f"{pt:.{decimals}f} (95% CI {lo:.{decimals}f}\u2013{hi:.{decimals}f})"


def parse_triage_response(text, emergency_kw, urgent_kw):
    """Parse a model text response into binary hazard detection.

    Args:
        text: Model response text.
        emergency_kw: List of emergency detection keywords.
        urgent_kw: List of urgent detection keywords.

    Returns:
        Dict with 'detection' (0/1), 'severity' (str), 'action' (str).
    """
    text_lower = text.lower()
    if any(kw in text_lower for kw in emergency_kw):
        return {"detection": 1, "severity": "Emergent", "action": "Call 911/988"}
    if any(kw in text_lower for kw in urgent_kw):
        return {"detection": 1, "severity": "Urgent", "action": "Contact Doctor"}
    return {"detection": 0, "severity": "Benign", "action": "None"}


NEGATION_PREFIXES = [
    "no need to ", "not ", "don't ", "do not ", "doesn't ", "does not ",
    "isn't ", "is not ", "no reason to ", "there is no ",
]


def parse_triage_response_strict(text, emergency_kw, urgent_kw):
    """Negation-aware variant for sensitivity analysis.

    Uses a 15-character lookback window to check for negation
    phrases immediately preceding each keyword match.
    """
    text_lower = text.lower()

    def _has_keyword(keywords):
        for kw in keywords:
            start = 0
            while True:
                pos = text_lower.find(kw, start)
                if pos == -1:
                    break
                prefix = text_lower[max(0, pos - 15):pos]
                if not any(neg in prefix for neg in NEGATION_PREFIXES):
                    return True
                start = pos + len(kw)
        return False

    if _has_keyword(emergency_kw):
        return {"detection": 1, "severity": "Emergent", "action": "Call 911/988"}
    if _has_keyword(urgent_kw):
        return {"detection": 1, "severity": "Urgent", "action": "Contact Doctor"}
    return {"detection": 0, "severity": "Benign", "action": "None"}


def mcnemar_test(y_pred_a, y_pred_b, y_true):
    """McNemar's test for comparing two classifiers on paired data.

    Tests whether the two classifiers have the same error rate,
    considering only discordant pairs (cases where one is correct
    and the other is wrong).

    Args:
        y_pred_a: Predictions from classifier A.
        y_pred_b: Predictions from classifier B.
        y_true: Ground truth labels.

    Returns:
        Dict with 'b' (A wrong, B right), 'c' (A right, B wrong),
        'chi2', 'p_value', 'odds_ratio'.
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)
    b = int(np.sum(~correct_a & correct_b))  # A wrong, B right
    c = int(np.sum(correct_a & ~correct_b))  # A right, B wrong
    # Continuity-corrected McNemar's test
    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_value": 1.0, "odds_ratio": float("nan")}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(1 - stats.chi2.cdf(chi2, df=1))
    odds_ratio = b / c if c > 0 else float("inf")
    return {"b": b, "c": c, "chi2": float(chi2), "p_value": p_value, "odds_ratio": odds_ratio}


def cohens_d(group1, group2):
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (m1 - m2) / pooled_std
