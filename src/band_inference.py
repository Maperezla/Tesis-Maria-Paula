from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

@dataclass
class BandInference:
    mapping: Dict[str, int]      # nombre -> índice 1-based
    confidence: Dict[str, float]
    notes: List[str]

def infer_band_order(arr5: np.ndarray) -> BandInference:
    """
    Intenta inferir VV, VH, angle, VVVH_ratio, VV_Difference.
    Aunque excluyamos VV_Difference del análisis, lo dejamos aquí
    para no romper el esquema de 5 bandas.
    """
    notes = []
    mapping = {}
    conf = {}

    band_stats = []
    for b in range(5):
        v = arr5[b].ravel()
        v = v[np.isfinite(v)]
        if v.size == 0:
            band_stats.append({"b": b, "n": 0})
            continue
        band_stats.append({
            "b": b,
            "n": int(v.size),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "p10": float(np.percentile(v, 10)),
            "p50": float(np.percentile(v, 50)),
            "p90": float(np.percentile(v, 90)),
            "neg_frac": float(np.mean(v < 0.0)),
        })

    # angle
    angle_candidates = []
    for s in band_stats:
        if s.get("n", 0) == 0:
            continue
        rng = s["max"] - s["min"]
        if (s["min"] >= 0) and (s["max"] <= 90) and (rng <= 30):
            angle_candidates.append((rng, s["b"], s))
    if angle_candidates:
        angle_candidates.sort(key=lambda x: x[0])
        b = angle_candidates[0][1]
        mapping["angle"] = b + 1
        conf["angle"] = 0.85
    else:
        notes.append("No fue posible identificar 'angle' con alta confianza.")
        conf["angle"] = 0.0

    used = set(mapping.values())

    # VV_Difference (lo inferimos, aunque no se use luego)
    diff_candidates = []
    for s in band_stats:
        if s.get("n", 0) == 0:
            continue
        if (s["neg_frac"] > 0.05) or (s["min"] < 0):
            score = s["neg_frac"] + (0.1 if abs(s["mean"]) < 0.2 else 0.0)
            diff_candidates.append((score, s["b"], s))
    if diff_candidates:
        diff_candidates.sort(key=lambda x: x[0], reverse=True)
        b = diff_candidates[0][1]
        if (b + 1) not in used:
            mapping["VV_Difference"] = b + 1
            conf["VV_Difference"] = 0.8
            used.add(b + 1)
    else:
        conf["VV_Difference"] = 0.0

    # ratio
    ratio_candidates = []
    for s in band_stats:
        if s.get("n", 0) == 0:
            continue
        idx = s["b"] + 1
        if idx in used:
            continue
        if s["min"] >= 0 and s["p50"] > 0:
            tail = s["p90"] / max(s["p50"], 1e-6)
            ratio_candidates.append((tail, s["b"], s))
    if ratio_candidates:
        ratio_candidates.sort(key=lambda x: x[0], reverse=True)
        b = ratio_candidates[0][1]
        mapping["VVVH_ratio"] = b + 1
        conf["VVVH_ratio"] = 0.75
        used.add(b + 1)
    else:
        conf["VVVH_ratio"] = 0.0

    # VV y VH
    remaining = [s for s in band_stats if s.get("n", 0) > 0 and (s["b"] + 1) not in used]
    if len(remaining) == 2:
        remaining_sorted = sorted(remaining, key=lambda s: s["mean"], reverse=True)
        mapping["VV"] = remaining_sorted[0]["b"] + 1
        mapping["VH"] = remaining_sorted[1]["b"] + 1
        conf["VV"] = 0.7
        conf["VH"] = 0.7
    else:
        # fallback
        for name in ["VV", "VH"]:
            if name not in mapping:
                for b in range(1, 6):
                    if b not in used:
                        mapping[name] = b
                        used.add(b)
                        conf[name] = 0.3
                        break

    return BandInference(mapping=mapping, confidence=conf, notes=notes)