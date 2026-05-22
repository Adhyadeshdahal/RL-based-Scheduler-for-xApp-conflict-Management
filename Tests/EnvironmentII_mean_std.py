import numpy as np
import time


NUM_SAMPLES = 1_000_000


def safe_exp(x):
    return np.where(np.abs(x) > 1e-1, x, 1e-1)


def compute_kpis(params):
    p1, p2, p3, p4, p5, p6, p7, p8 = params

    p2_safe = safe_exp(p2)
    p4_safe = safe_exp(p4)
    p5_safe = safe_exp(p5)
    p7_safe = safe_exp(p7)

    kpi1  = 80  * np.exp(-(p1) ** 2           / (2 * (p2_safe ** 2)))
    kpi2  = 100 * np.exp(-(p1 + p3) ** 2      / (2 * (p2_safe ** 2)))
    kpi3  = 120 * np.exp(-(p1 + 45) ** 2      / (2 * (p4_safe ** 2)))
    kpi41 = 120 * np.exp(-(p6 + p2 - 30) ** 2 / (2 * (p5_safe ** 2)))
    kpi42 = 150 * np.exp(-(p6 + p2 - 50) ** 2 / (2 * (p5_safe ** 2)))
    kpi5  = -35 * np.exp(-(p8 + p1 - 25) ** 2 / (2 * (p7_safe ** 2)))

    return [kpi1, kpi2, kpi3, kpi41, kpi42, kpi5]


def get_mean_std(param_ranges, seed, num_samples=NUM_SAMPLES):
    """
    Given a list of (low, high) param ranges, returns MEAN_STD_KPIS
    as a list of (mean, std) tuples, one per KPI.
    """
    rng    = np.random.default_rng(seed)
    params = [rng.uniform(low, high, size=num_samples) for low, high in param_ranges]
    kpis   = compute_kpis(params)
    return [(float(np.mean(k)), float(np.std(k))) for k in kpis]


if __name__ == "__main__":
    PARAM_RANGES = [
        [(-100,100), (-10,50), (-20,20), (-60,60), (-20,20), (-50,150), (-60,65), (-100,150)],
        [(-100,100), (-10,50), (-20,-19), (60,61), (-20,-19), (-50,150), (60,61), (-100,150)],
    ]
    KPI_NAMES = ["KPI1", "KPI2", "KPI3", "KPI41", "KPI42", "KPI5"]

    for idx, ranges in enumerate(PARAM_RANGES):
        t0 = time.perf_counter()
        mean_std = get_mean_std(ranges)
        print(f"\n# PARAM_RANGES[{idx}] ({'Train' if idx == 0 else 'Test'}):")
        print("MEAN_STD_KPIS = [")
        for name, (mean, std) in zip(KPI_NAMES, mean_std):
            print(f"    ({mean:.6f}, {std:.6f}),  # {name}")
        print("]")
        print(f"# Time: {time.perf_counter() - t0:.2f}s")