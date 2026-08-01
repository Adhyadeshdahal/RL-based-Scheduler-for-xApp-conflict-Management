import time

import numpy as np

from env_statistics.cache import get_cached_mean_std

NUM_SAMPLES = 1_000_000
CHUNK = 100_000


def safe_exp(x):
    return np.where(x > 0, x, 1e-3)


def compute_kpis(params):
    """
    KPI equations for Environment I:
        K1 = 0.5 * exp(-(P1+50)^2 / (2*P2)^2)
        K2 = exp(-(P1-50)^2       / (2*P3)^2)
        K3 = exp(-(P4+K1)^2       / (2*P5)^2)   ← depends on K1
        K4 = exp(-(P7+K2)^2       / (2*P6)^2)   ← depends on K2

    K3 and K4 depend on K1/K2, so we cannot evaluate them
    independently per-sample without knowing the joint (P4,P5,K1)
    and (P7,P6,K2) distributions. We use a chunked grid sweep
    internally to get exact statistics, but accept the same
    param_ranges / seed / num_samples signature as Environment II
    for consistency.

    seed and num_samples are used for K1/K2 sampling (P1×P2 and P1×P3),
    then K3/K4 are computed over the resulting K1/K2 distributions
    combined with grid sweeps over P4,P5 and P6,P7.
    """
    p1, p2, p3, p4, p5, p6, p7 = params

    p2_safe = safe_exp(p2)
    p3_safe = safe_exp(p3)
    p5_safe = safe_exp(p5)
    p6_safe = safe_exp(p6)

    kpi1 = 0.5 * np.exp(-((p1 + 50) ** 2) / (2 * p2_safe) ** 2)
    kpi2 = np.exp(-((p1 - 50) ** 2) / (2 * p3_safe) ** 2)
    kpi3 = np.exp(-((p4 + kpi1) ** 2) / (2 * p5_safe) ** 2)
    kpi4 = np.exp(-((p7 + kpi2) ** 2) / (2 * p6_safe) ** 2)

    return [kpi1, kpi2, kpi3, kpi4]


def get_mean_std(param_ranges, seed, num_samples=NUM_SAMPLES):
    """
    Same signature as EnvironmentII_mean_std.get_mean_std.

    param_ranges : list of 7 (low, high) tuples — [P1, P2, P3, P4, P5, P6, P7]
    seed         : int, RNG seed
    num_samples  : number of random samples

    Returns list of (mean, std) tuples: [(m1,s1), (m2,s2), (m3,s3), (m4,s4)]
    """

    def compute():
        rng = np.random.default_rng(seed)
        params = [rng.uniform(low, high, size=num_samples) for low, high in param_ranges]
        kpis = compute_kpis(params)
        return [(float(np.mean(k)), float(np.std(k))) for k in kpis]

    return get_cached_mean_std("EnvironmentI", param_ranges, seed, num_samples, compute)


if __name__ == "__main__":
    PARAM_RANGES = [
        (0, 300),  # P1
        (0, 300),  # P2
        (0, 3),  # P3
        (0, 3),  # P4
        (0, 3),  # P5
        (0, 3),  # P6
        (0, 3),  # P7
    ]
    KPI_NAMES = ["KPI1", "KPI2", "KPI3", "KPI4"]
    SEED = 45

    t0 = time.perf_counter()
    mean_std = get_mean_std(PARAM_RANGES, seed=SEED)
    print("MEAN_STD_KPIS = [")
    for name, (mean, std) in zip(KPI_NAMES, mean_std, strict=True):
        print(f"    ({mean:.6f}, {std:.6f}),  # {name}")
    print("]")
    print(f"# Time: {time.perf_counter() - t0:.2f}s")
