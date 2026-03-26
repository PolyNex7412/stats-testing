"""Statistical test implementations."""
import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class TestResult:
    test_name: str
    statistic: float
    p_value: float
    alpha: float
    reject_h0: bool
    h0: str
    h1: str
    dist_type: str          # "t" | "normal" | "chi2"
    tail: str               # "two" | "left" | "right"
    df: Optional[float] = None
    ci: Optional[Tuple[float, float]] = None
    effect_size: Optional[float] = None
    n: Optional[int] = None
    n2: Optional[int] = None
    mean1: Optional[float] = None
    mean2: Optional[float] = None
    std1: Optional[float] = None
    std2: Optional[float] = None
    extra: dict = field(default_factory=dict)


# ── 1標本 t 検定 ──────────────────────────────────────────────────────────────
def one_sample_t(data: np.ndarray, mu0: float, alpha: float = 0.05, tail: str = "two") -> TestResult:
    n = len(data)
    stat, p_two = stats.ttest_1samp(data, mu0)
    p = _adjust_p(stat, p_two, tail)
    df = n - 1
    ci = stats.t.interval(1 - alpha, df, loc=np.mean(data), scale=stats.sem(data))
    d = float((np.mean(data) - mu0) / np.std(data, ddof=1))

    return TestResult(
        test_name="1標本 t 検定",
        statistic=float(stat), p_value=float(p), alpha=alpha,
        reject_h0=p < alpha, df=float(df), ci=ci,
        h0=f"μ = {mu0}",
        h1=f"μ {'≠' if tail=='two' else ('<' if tail=='left' else '>')} {mu0}",
        dist_type="t", tail=tail,
        n=n, mean1=float(np.mean(data)), std1=float(np.std(data, ddof=1)),
        effect_size=d,
    )


# ── 2標本 t 検定 ──────────────────────────────────────────────────────────────
def two_sample_t(data1: np.ndarray, data2: np.ndarray,
                 alpha: float = 0.05, tail: str = "two",
                 equal_var: bool = True) -> TestResult:
    stat, p_two = stats.ttest_ind(data1, data2, equal_var=equal_var)
    p = _adjust_p(stat, p_two, tail)
    n1, n2 = len(data1), len(data2)

    s1, s2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
    if equal_var:
        df = n1 + n2 - 2
    else:
        df = (s1/n1 + s2/n2)**2 / ((s1/n1)**2/(n1-1) + (s2/n2)**2/(n2-1))

    diff = float(np.mean(data1) - np.mean(data2))
    se = float(np.sqrt(s1/n1 + s2/n2))
    t_crit = stats.t.ppf(1 - alpha/2, df)
    ci = (diff - t_crit*se, diff + t_crit*se)

    pooled = float(np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2)))
    d = diff / pooled

    return TestResult(
        test_name=f"2標本 t 検定 ({'等分散' if equal_var else 'Welch'})",
        statistic=float(stat), p_value=float(p), alpha=alpha,
        reject_h0=p < alpha, df=float(df), ci=ci,
        h0="μ₁ = μ₂",
        h1=f"μ₁ {'≠' if tail=='two' else ('<' if tail=='left' else '>')} μ₂",
        dist_type="t", tail=tail,
        n=n1, n2=n2,
        mean1=float(np.mean(data1)), mean2=float(np.mean(data2)),
        std1=float(np.std(data1, ddof=1)), std2=float(np.std(data2, ddof=1)),
        effect_size=float(d),
    )


# ── 対応のある t 検定 ──────────────────────────────────────────────────────────
def paired_t(data1: np.ndarray, data2: np.ndarray,
             alpha: float = 0.05, tail: str = "two") -> TestResult:
    stat, p_two = stats.ttest_rel(data1, data2)
    p = _adjust_p(stat, p_two, tail)
    diff = data1 - data2
    n = len(diff)
    df = n - 1
    ci = stats.t.interval(1 - alpha, df, loc=np.mean(diff), scale=stats.sem(diff))
    d = float(np.mean(diff) / np.std(diff, ddof=1))

    return TestResult(
        test_name="対応のある t 検定",
        statistic=float(stat), p_value=float(p), alpha=alpha,
        reject_h0=p < alpha, df=float(df), ci=ci,
        h0="μ_d = 0",
        h1=f"μ_d {'≠' if tail=='two' else ('<' if tail=='left' else '>')} 0",
        dist_type="t", tail=tail,
        n=n, mean1=float(np.mean(data1)), mean2=float(np.mean(data2)),
        std1=float(np.std(diff, ddof=1)),
        effect_size=d,
        extra={"diff_mean": float(np.mean(diff)), "diff_std": float(np.std(diff, ddof=1))},
    )


# ── カイ二乗検定（独立性）────────────────────────────────────────────────────
def chi2_independence(table: np.ndarray, alpha: float = 0.05):
    stat, p, dof, expected = stats.chi2_contingency(table)
    result = TestResult(
        test_name="カイ二乗検定（独立性の検定）",
        statistic=float(stat), p_value=float(p), alpha=alpha,
        reject_h0=p < alpha, df=float(dof),
        h0="2変数は独立（関連なし）",
        h1="2変数は独立でない（関連あり）",
        dist_type="chi2", tail="right",
    )
    return result, expected


# ── z 検定（母分散既知）──────────────────────────────────────────────────────
def z_test(data: np.ndarray, mu0: float, sigma: float,
           alpha: float = 0.05, tail: str = "two") -> TestResult:
    n = len(data)
    z = float((np.mean(data) - mu0) / (sigma / np.sqrt(n)))
    if tail == "two":
        p = float(2 * (1 - stats.norm.cdf(abs(z))))
    elif tail == "left":
        p = float(stats.norm.cdf(z))
    else:
        p = float(1 - stats.norm.cdf(z))
    z_crit = stats.norm.ppf(1 - alpha/2)
    se = sigma / np.sqrt(n)
    ci = (float(np.mean(data)) - z_crit*se, float(np.mean(data)) + z_crit*se)

    return TestResult(
        test_name="z 検定（母分散既知）",
        statistic=z, p_value=p, alpha=alpha,
        reject_h0=p < alpha, df=None, ci=ci,
        h0=f"μ = {mu0}",
        h1=f"μ {'≠' if tail=='two' else ('<' if tail=='left' else '>')} {mu0}",
        dist_type="normal", tail=tail,
        n=n, mean1=float(np.mean(data)),
        effect_size=float((np.mean(data) - mu0) / sigma),
    )


# ── Power / sample size ───────────────────────────────────────────────────────
def compute_power(effect_size: float, n: int, alpha: float,
                  tail: str = "two", dist: str = "t") -> float:
    """Compute power (1 - β) for a t-test or z-test given effect size and n."""
    if dist == "t":
        df = n - 1
        if tail == "two":
            crit = stats.t.ppf(1 - alpha/2, df)
            nc = effect_size * np.sqrt(n)
            power = 1 - stats.t.cdf(crit, df, nc) + stats.t.cdf(-crit, df, nc)
        else:
            crit = stats.t.ppf(1 - alpha, df)
            nc = effect_size * np.sqrt(n)
            power = 1 - stats.t.cdf(crit, df, nc)
    else:
        if tail == "two":
            crit = stats.norm.ppf(1 - alpha/2)
            nc = effect_size * np.sqrt(n)
            power = 1 - stats.norm.cdf(crit - nc) + stats.norm.cdf(-crit - nc)
        else:
            crit = stats.norm.ppf(1 - alpha)
            nc = effect_size * np.sqrt(n)
            power = 1 - stats.norm.cdf(crit - nc)
    return float(np.clip(power, 0, 1))


# ── helpers ───────────────────────────────────────────────────────────────────
def _adjust_p(stat: float, p_two: float, tail: str) -> float:
    if tail == "two":
        return float(p_two)
    elif tail == "left":
        return float(p_two / 2 if stat < 0 else 1 - p_two / 2)
    else:
        return float(p_two / 2 if stat > 0 else 1 - p_two / 2)
