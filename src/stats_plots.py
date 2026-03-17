from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_band_stats(df_long: pd.DataFrame, bands: list[str]) -> pd.DataFrame:
    rows = []
    for band in bands:
        d = df_long[df_long["band"] == band]["value"]
        stats = d.describe(percentiles=[0.25, 0.5, 0.75])
        rows.append({
            "band": band,
            "count": float(stats["count"]),
            "mean": float(stats["mean"]),
            "std": float(stats["std"]),
            "min": float(stats["min"]),
            "p25": float(stats["25%"]),
            "median": float(stats["50%"]),
            "p75": float(stats["75%"]),
            "max": float(stats["max"]),
            "nan_count": int(pd.isna(d).sum())
        })
    return pd.DataFrame(rows)


def pivot_band_time(df: pd.DataFrame, band: str, order_images: list[str]) -> pd.DataFrame:
    d = df[(df["band"] == band) & (df["image_name"].isin(order_images))].copy()
    pv = d.pivot_table(index="image_name", columns="point_id", values="value", aggfunc="first")
    return pv.reindex(order_images)


def plot_multiline(pv: pd.DataFrame, band: str, out_png: Path):
    plt.figure(figsize=(14, 6))
    x = np.arange(len(pv.index))

    for pid in pv.columns:
        plt.plot(x, pv[pid].values, linewidth=1)

    plt.xticks(x, pv.index, rotation=90)
    plt.xlabel("Imagen")
    plt.ylabel("Valor (dB)" if band != "angle" else "angle")
    plt.title(f"Serie multitemporal por punto — {band}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def choose_sample_points(
    df_long: pd.DataFrame,
    order_images: list[str],
    bands_time: list[str],
    n_sample: int,
    min_valid_frac: float,
    random_state: int
) -> list[int]:
    min_valid_count = int(np.ceil(len(order_images) * min_valid_frac))

    valid_counts = {}
    for band in bands_time:
        d = df_long[(df_long["band"] == band) & (df_long["image_name"].isin(order_images))].copy()
        cnt = d.groupby("point_id")["value"].apply(lambda s: s.notna().sum())
        valid_counts[band] = cnt

    df_valid = pd.DataFrame(valid_counts).fillna(0).astype(int)
    eligible_mask = (df_valid[bands_time] >= min_valid_count).all(axis=1)
    eligible_points = df_valid.index[eligible_mask].tolist()

    if len(eligible_points) < n_sample:
        df_valid["sum_valid"] = df_valid[bands_time].sum(axis=1)
        return df_valid.sort_values("sum_valid", ascending=False).head(n_sample).index.tolist()

    rng = np.random.default_rng(random_state)
    return rng.choice(eligible_points, size=n_sample, replace=False).tolist()


def pivot_band_time_subset(df: pd.DataFrame, band: str, point_ids: list[int], order_images: list[str]) -> pd.DataFrame:
    d = df[
        (df["band"] == band)
        & (df["image_name"].isin(order_images))
        & (df["point_id"].isin(point_ids))
    ].copy()
    pv = d.pivot_table(index="image_name", columns="point_id", values="value", aggfunc="first")
    return pv.reindex(order_images)


def plot_multiline_subset(pv: pd.DataFrame, band: str, out_png: Path):
    plt.figure(figsize=(14, 6))
    x = np.arange(len(pv.index))

    for pid in pv.columns:
        plt.plot(x, pv[pid].values, linewidth=2, label=f"point_id={pid}")

    plt.xticks(x, pv.index, rotation=90)
    plt.xlabel("Imagen")
    plt.ylabel("Valor (dB)" if band != "angle" else "angle")
    plt.title(f"Serie multitemporal — {band} (solo puntos seleccionados)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()