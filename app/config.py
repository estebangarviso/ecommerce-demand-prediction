"""Configuración de la aplicación."""

from typing import Dict

# Configuración de mapeos
CLUSTER_MAP: Dict[int, str] = {
    0: "Tienda Pequeña / Kiosco (Bajo Volumen)",
    1: "Supermercado / Mall (Volumen Medio)",
    2: "Megatienda / Online (Alto Volumen)",
}

# Configuración de precios
DEFAULT_PRICE: float = 1500.0
DEFAULT_PRICE_MIN: float = 0.0
DEFAULT_PRICE_MAX: float = 50000.0
PRICE_RANGE_MULTIPLIER: float = 0.33  # Para calcular el mínimo (33% del promedio)
PRICE_RANGE_MAX_MULTIPLIER: float = 3.0  # Para calcular el máximo (300% del promedio)

# Configuración de tema
DARK_THEME_BG_COLOR: str = "#0E1117"
DARK_THEME_TEXT_COLOR: str = "#FAFAFA"
LIGHT_THEME_BG_COLOR: str = "#FFFFFF"
LIGHT_THEME_TEXT_COLOR: str = "#262730"

# Configuración de colores
CHART_COLORS = {
    "historical": "#B0BEC5",
    "positive": "#4CAF50",
    "negative": "#F44336",
    "shap_negative": "#FF6B6B",
    "shap_positive": "#90EE90",
}
