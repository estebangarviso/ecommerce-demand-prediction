"""Renderizador de gráficos SHAP con soporte de temas."""

from typing import Dict, Any, Optional
import streamlit as st
import shap
import streamlit.components.v1 as components


class SHAPRenderer:
    """Renderizador de gráficos SHAP con soporte de temas."""

    def __init__(self, theme_config: Dict[str, str]):
        """
        Inicializa el renderizador SHAP.

        Args:
            theme_config: Diccionario con configuración de colores del tema
        """
        self.theme_config = theme_config

    def detect_theme(self) -> str:
        """Detecta el tema actual de Streamlit."""
        return "dark" if st.get_option("theme.base") == "dark" else "light"

    def render(self, plot: Any, height: Optional[int] = None) -> None:
        """
        Renderiza un gráfico SHAP con estilos adaptados al tema.

        Args:
            plot: Gráfico SHAP a renderizar
            height: Altura del componente en píxeles
        """
        theme = self.detect_theme()

        bg_color = (
            self.theme_config["dark_bg"] if theme == "dark" else self.theme_config["light_bg"]
        )
        text_color = (
            self.theme_config["dark_text"] if theme == "dark" else self.theme_config["light_text"]
        )

        shap_html = f"""
        <head>
            {shap.getjs()}
            <style>
                body {{
                    background-color: {bg_color} !important;
                    color: {text_color} !important;
                }}
                text {{
                    fill: {text_color} !important;
                }}
                .additive-force-array-wrapper {{
                    background-color: {bg_color} !important;
                }}
            </style>
        </head>
        <body>{plot.html()}</body>
        """
        components.html(shap_html, height=height if height else 300)
