from pathlib import Path
from typing import Dict, List
import pandas as pd


def build_ranked_table(df_dom: pd.DataFrame, domains: Dict[str, List[str]]) -> pd.DataFrame:
    domain_bandcount = {k: len(v) for k, v in domains.items()}
    best_domain = max(domain_bandcount, key=domain_bandcount.get)
    rank_col = f"{best_domain}_valid_pct"

    df_rank = df_dom.sort_values(
        by=[rank_col, "geom_match_ref"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return df_rank


def write_report(
    df_rank: pd.DataFrame,
    domains: Dict[str, List[str]],
    out_path: Path
) -> None:
    best = df_rank.iloc[0].to_dict()
    domain_bandcount = {k: len(v) for k, v in domains.items()}
    best_domain = max(domain_bandcount, key=domain_bandcount.get)
    rank_col = f"{best_domain}_valid_pct"

    lines = []
    lines.append("REPORTE — DOMINIOS DE PIXELES VÁLIDOS (Sentinel-1)")
    lines.append("=" * 78)
    lines.append(f"Total escenas evaluadas: {len(df_rank)}")
    lines.append("")
    lines.append("Definición de validez: píxel válido = valor finito (no NaN/Inf) tras recorte al AOI.")
    lines.append("Los dominios representan la intersección de validez entre bandas.")
    lines.append("")

    lines.append("DOMINIOS reportados:")
    for dom_name, bands in domains.items():
        lines.append(f"- {dom_name}: {', '.join(bands)}")
    lines.append("")

    lines.append(f"MEJOR ESCENA PARA RANDOM FOREST MULTIBANDA (prioriza {best_domain}):")
    lines.append(f"- Variable objetivo de ranking: {rank_col}")
    lines.append(f"- Archivo: {best['file']}")
    lines.append(f"- Coherencia geométrica con referencia: {best['geom_match_ref']}")
    lines.append("")

    for dom_name in domains.keys():
        lines.append(f"- {dom_name}: {best[f'{dom_name}_valid_pct']:.2f}% válido")

    lines.append("")
    lines.append(f"Top 5 (por {best_domain}):")

    top5 = df_rank.head(5)
    for i, r in top5.iterrows():
        summary = [f"{k}={r[f'{k}_valid_pct']:.2f}%" for k in domains.keys()]
        lines.append(
            f"  {i+1}. {r['file']} | " + " | ".join(summary) + f" | geom_ok={r['geom_match_ref']}"
        )

    lines.append("")
    lines.append("Interpretación para Random Forest:")
    lines.append("- El conjunto de entrenamiento efectivo queda acotado por el dominio multibanda más exigente.")
    lines.append("- Si ese dominio es bajo, conviene evaluar modelos por niveles de bandas.")
    lines.append("- Este reporte permite trazabilidad: un píxel solo entra al análisis si está válido en todas las bandas del dominio.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")