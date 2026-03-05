from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

def finite_mask(arr_band: np.ndarray) -> np.ndarray:
    return np.isfinite(arr_band)

def robust_range(values: np.ndarray, p_lo: float = 1, p_hi: float = 99) -> Tuple[float, float]:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return (np.nan, np.nan)
    lo = np.percentile(v, p_lo)
    hi = np.percentile(v, p_hi)
    if lo == hi:
        hi = lo + 1e-6
    return float(lo), float(hi)

def descriptive_stats(values: np.ndarray) -> Dict[str, float]:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return {k: np.nan for k in ["n","min","p25","median","mean","p75","max","std","iqr"]}
    p25 = np.percentile(v, 25)
    p75 = np.percentile(v, 75)
    return {
        "n": int(v.size),
        "min": float(np.min(v)),
        "p25": float(p25),
        "median": float(np.percentile(v, 50)),
        "mean": float(np.mean(v)),
        "p75": float(p75),
        "max": float(np.max(v)),
        "std": float(np.std(v)),
        "iqr": float(p75 - p25),
    }

def plot_histogram(values: np.ndarray, title: str, out_png,
                   nbins: int = 80, p_lo: float = 1, p_hi: float = 99) -> None:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return
    lo, hi = robust_range(v, p_lo, p_hi)
    v_clip = v[(v >= lo) & (v <= hi)]
    plt.figure()
    plt.hist(v_clip, bins=nbins)
    plt.title(title)
    plt.xlabel("Valor")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()