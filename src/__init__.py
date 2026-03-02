"""
Paquete de preprocesamiento Sentinel-1 (GRD) en Google Earth Engine.

Incluye:
- Construcción de colección (VV, VH, angle)
- Correcciones radiométricas (gamma0, terrain flattening)
- Filtrado speckle espacial (Lee)
- Índices SAR (VV/VH ratio, VV difference)

Convención:
- VV y VH se mantienen en dB en todo el pipeline.
- Conversiones a lineal se hacen SOLO en operaciones específicas.
"""

# =========================
# COLECCIÓN
# =========================
from .collection import build_collection

# =========================
# RADIOMETRÍA
# =========================
from .radiometry import (
    border_noise_mask,
    gamma0_db,
    terrain_flattening,
)

# =========================
# SPECKLE
# =========================
from .speckle import refined_lee_spatial

# =========================
# ÍNDICES
# =========================
from .indices import (
    add_ratio_db,
    add_vv_difference_1y_signed_db,
)

# =========================
# API PÚBLICA
# =========================
__all__ = [
    # collection
    "build_collection",

    # radiometry
    "border_noise_mask",
    "gamma0_db",
    "terrain_flattening",

    # speckle
    "refined_lee_spatial",

    # indices
    "add_ratio_db",
    "add_vv_difference_1y_signed_db",
]