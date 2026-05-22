import numpy as np
import time

P1_range = (0, 300)
P2_range = (0, 300)
P3_range = (0, 3)
P4_range = (0, 3)
P5_range = (0, 3)
P6_range = (0, 3)
P7_range = (0, 3)

STEP = 0.1
CHUNK = 100_000


def kpi1_vec(P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    safe = np.where(P2 > 0, P2, 1e-3)
    return (
        0.5 * np.exp(-((P1[:, None] + 50) ** 2) / (2 * safe[None, :] ** 2))
    ).astype(np.float32)


def kpi2_vec(P1: np.ndarray, P3: np.ndarray) -> np.ndarray:
    safe = np.where(P3 > 0, P3, 1e-3)
    return (
        np.exp(-((P1[:, None] - 50) ** 2) / (2 * safe[None, :] ** 2))
    ).astype(np.float32)


def kpi3_vec_chunked(P4: np.ndarray, P5: np.ndarray, K1_flat: np.ndarray) -> tuple:
    safe_P5 = np.where(P5 > 0, P5, 1e-3).astype(np.float32)
    P4 = P4.astype(np.float32)

    n_total = len(P4) * len(P5) * len(K1_flat)

    running_sum = np.float64(0.0)
    running_sum_sq = np.float64(0.0)

    for start in range(0, len(K1_flat), CHUNK):
        chunk = K1_flat[start:start + CHUNK].astype(np.float32)

        num = -(P4[:, None, None] + chunk[None, None, :]) ** 2
        den = 2 * safe_P5[None, :, None] ** 2

        block = np.exp(num / den)

        running_sum += block.sum(dtype=np.float64)
        running_sum_sq += (block ** 2).sum(dtype=np.float64)

    mean = running_sum / n_total
    std = np.sqrt(running_sum_sq / n_total - mean ** 2)

    return float(mean), float(std)


def kpi4_vec_chunked(P7: np.ndarray, P6: np.ndarray, K2_flat: np.ndarray) -> tuple:
    safe_P6 = np.where(P6 > 0, P6, 1e-3).astype(np.float32)
    P7 = P7.astype(np.float32)

    n_total = len(P7) * len(P6) * len(K2_flat)

    running_sum = np.float64(0.0)
    running_sum_sq = np.float64(0.0)

    for start in range(0, len(K2_flat), CHUNK):
        chunk = K2_flat[start:start + CHUNK].astype(np.float32)

        num = -(P7[:, None, None] + chunk[None, None, :]) ** 2
        den = 2 * safe_P6[None, :, None] ** 2

        block = np.exp(num / den)

        running_sum += block.sum(dtype=np.float64)
        running_sum_sq += (block ** 2).sum(dtype=np.float64)

    mean = running_sum / n_total
    std = np.sqrt(running_sum_sq / n_total - mean ** 2)

    return float(mean), float(std)


def main():
    t0 = time.perf_counter()

    P1 = np.arange(P1_range[0], P1_range[1] + STEP, STEP)
    P2 = np.arange(P2_range[0], P2_range[1] + STEP, STEP)
    P3 = np.arange(P3_range[0], P3_range[1] + STEP, STEP)
    P4 = np.arange(P4_range[0], P4_range[1] + STEP, STEP)
    P5 = np.arange(P5_range[0], P5_range[1] + STEP, STEP)
    P6 = np.arange(P6_range[0], P6_range[1] + STEP, STEP)
    P7 = np.arange(P7_range[0], P7_range[1] + STEP, STEP)

    print("Computing KPI1 & KPI2 ...")

    K1_grid = kpi1_vec(P1, P2)
    K2_grid = kpi2_vec(P1, P3)

    print(f"KPI1 mean={K1_grid.mean():.6f} std={K1_grid.std():.6f}")
    print(f"KPI2 mean={K2_grid.mean():.6f} std={K2_grid.std():.6f}")

    K1_flat = K1_grid.ravel()
    K2_flat = K2_grid.ravel()

    del K1_grid, K2_grid

    print("Computing KPI3 ...")

    m3, s3 = kpi3_vec_chunked(P4, P5, K1_flat)

    print(f"KPI3 mean={m3:.6f} std={s3:.6f}")

    print("Computing KPI4 ...")

    m4, s4 = kpi4_vec_chunked(P7, P6, K2_flat)

    print(f"KPI4 mean={m4:.6f} std={s4:.6f}")

    print(f"\nTotal time: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()