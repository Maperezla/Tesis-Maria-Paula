from pathlib import Path
import pandas as pd

def write_report(df_rank: pd.DataFrame, bands_used, out_txt: Path) -> None:
    best = df_rank.iloc[0].to_dict()

    def fmt_pct(x): return f"{x:.2f}%"

    lines = []
    lines.append("JUSTIFICACIÓN TÉCNICA — SELECCIÓN DE ESCENA SENTINEL-1")
    lines.append("="*78)
    lines.append(f"Total escenas evaluadas: {len(df_rank)}")
    lines.append("")
    lines.append("Bandas evaluadas:")
    lines.append(f"- {', '.join(bands_used)}")
    lines.append("")
    lines.append("ESCENA SELECCIONADA (mejor ranking):")
    lines.append(f"- Archivo: {best['file']}")
    lines.append(f"- % válido (bandas evaluadas): {fmt_pct(best['valid_allbands_pct'])} | % null: {fmt_pct(best['null_anyband_pct'])}")
    lines.append(f"- Coherencia geométrica con referencia: {best['geom_match_ref']}")
    lines.append("")

    out_txt.write_text("\n".join(lines), encoding="utf-8")