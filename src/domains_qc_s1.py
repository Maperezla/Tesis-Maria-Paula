from typing import Dict, List
import numpy as np


def get_fixed_band_mapping() -> Dict[str, int]:
    """
    Mapeo fijo y determinístico de bandas Sentinel-1:
    1 = VV
    2 = VH
    3 = angle
    4 = VVVH_ratio
    5 = VV_Difference
    """
    return {
        "VV": 1,
        "VH": 2,
        "angle": 3,
        "VVVH_ratio": 4,
        "VV_Difference": 5,
    }


def finite_mask(arr_band: np.ndarray) -> np.ndarray:
    return np.isfinite(arr_band)


def compute_domain_metrics(
    arr: np.ndarray,
    domains: Dict[str, List[str]],
    band_mapping: Dict[str, int]
) -> Dict[str, float]:
    """
    Calcula métricas de validez por dominio usando mapeo fijo de bandas.
    """
    required_bands = sorted(set(sum(domains.values(), [])))
    named = {name: arr[band_mapping[name] - 1] for name in required_bands}

    total_px = int(next(iter(named.values())).size)

    row = {
        "total_px_aoi": total_px,
    }

    for dom_name, bands in domains.items():
        vmask = np.ones_like(next(iter(named.values())), dtype=bool)
        for b in bands:
            vmask &= finite_mask(named[b])

        valid_px = int(np.sum(vmask))
        null_px = total_px - valid_px

        row[f"{dom_name}_bands"] = ",".join(bands)
        row[f"{dom_name}_valid_px"] = valid_px
        row[f"{dom_name}_valid_pct"] = 100.0 * valid_px / total_px
        row[f"{dom_name}_null_px"] = null_px
        row[f"{dom_name}_null_pct"] = 100.0 * null_px / total_px

    domain_keys = sorted(
        [k for k in domains.keys() if k.startswith("D")],
        key=lambda s: int(s.split("_")[0][1:])
    )

    for prev_k, next_k in zip(domain_keys[:-1], domain_keys[1:]):
        row[f"gain_{prev_k}_to_{next_k}_pct"] = (
            row[f"{next_k}_valid_pct"] - row[f"{prev_k}_valid_pct"]
        )

    if "D2_VV_VH" in domain_keys:
        for k in domain_keys:
            if k != "D2_VV_VH":
                row[f"gain_D2_to_{k}_pct"] = (
                    row[f"{k}_valid_pct"] - row["D2_VV_VH_valid_pct"]
                )

    return row