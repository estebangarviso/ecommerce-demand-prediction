"""
Paquete de aplicación del Sistema Predictivo de Demanda.
"""

# Importar módulos principales
from app.config import (
    CLUSTER_MAP,
    DEFAULT_PRICE,
    DEFAULT_PRICE_MIN,
    DEFAULT_PRICE_MAX,
    PRICE_RANGE_MULTIPLIER,
    PRICE_RANGE_MAX_MULTIPLIER,
    DARK_THEME_BG_COLOR,
    DARK_THEME_TEXT_COLOR,
    LIGHT_THEME_BG_COLOR,
    LIGHT_THEME_TEXT_COLOR,
    CHART_COLORS,
)
from app.state_manager import SessionStateManager

__all__ = [
    "CLUSTER_MAP",
    "DEFAULT_PRICE",
    "DEFAULT_PRICE_MIN",
    "DEFAULT_PRICE_MAX",
    "PRICE_RANGE_MULTIPLIER",
    "PRICE_RANGE_MAX_MULTIPLIER",
    "DARK_THEME_BG_COLOR",
    "DARK_THEME_TEXT_COLOR",
    "LIGHT_THEME_BG_COLOR",
    "LIGHT_THEME_TEXT_COLOR",
    "CHART_COLORS",
    "SessionStateManager",
]
