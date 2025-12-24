"""
Sistema Predictivo de Demanda - Aplicación Principal.

Esta aplicación implementa un sistema de predicción de demanda utilizando
técnicas de Machine Learning.
"""

import sys
import os

# Agregar el directorio padre al path para imports absolutos
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

# Configuración de página
st.set_page_config(
    page_title="Sistema Predictivo de Demanda",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Importar módulos de la aplicación
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
)
from app.state_manager import SessionStateManager
from app.services import PricingService, PredictionService
from app.components import SHAPRenderer
from app.views import PredictionView, MonitoringView, AboutView, TechnicalAnalysisView
from app.ui_components import Sidebar, Header


def initialize_application():
    """Inicializa la aplicación cargando recursos desde la API."""
    # Inicializar servicio de predicción (se conecta a la API)
    prediction_service = PredictionService()

    # Verificar que la API esté disponible
    api_connected, health_data = prediction_service.check_api_health()
    if not api_connected:
        st.error(
            "No se puede conectar a la API. Asegúrate de que esté corriendo en http://localhost:8000",
            icon=":material/cloud_off:",
        )
        st.stop()

    # Extraer datos del health check
    model_metrics = health_data.get("model_metrics", {})
    rolling_windows = model_metrics.get("rolling_windows", [3, 6])

    # Cargar categorías y precios desde la API (mediante PredictionService)
    category_map = prediction_service.get_categories()
    cat_prices = prediction_service.get_category_prices()

    if not category_map:
        st.error(
            "No se pudieron cargar las categorías desde la API.",
            icon=":material/folder_off:",
        )
        st.stop()

    # Inicializar estado
    first_category = list(category_map.keys())[0] if category_map else None
    SessionStateManager.initialize_state(
        DEFAULT_PRICE, DEFAULT_PRICE_MIN, DEFAULT_PRICE_MAX, cat_prices, first_category
    )

    # Configurar rolling windows desde la API
    SessionStateManager.update_rolling_windows(rolling_windows)

    # Inicializar servicio de pricing
    pricing_service = PricingService(
        cat_prices,
        DEFAULT_PRICE,
        DEFAULT_PRICE_MIN,
        DEFAULT_PRICE_MAX,
        PRICE_RANGE_MULTIPLIER,
        PRICE_RANGE_MAX_MULTIPLIER,
    )

    # Configuración de tema para SHAP
    theme_config = {
        "dark_bg": DARK_THEME_BG_COLOR,
        "dark_text": DARK_THEME_TEXT_COLOR,
        "light_bg": LIGHT_THEME_BG_COLOR,
        "light_text": LIGHT_THEME_TEXT_COLOR,
    }
    shap_renderer = SHAPRenderer(theme_config)

    return category_map, pricing_service, prediction_service, shap_renderer


def main():
    """Función principal de la aplicación."""
    # Inicializar aplicación (conecta a la API)
    category_map, pricing_service, prediction_service, shap_renderer = initialize_application()

    # Renderizar header
    Header.render()

    # Renderizar sidebar y obtener inputs
    sidebar = Sidebar(category_map, CLUSTER_MAP, pricing_service)
    (
        item_category_id,
        shop_cluster,
        item_price,
        lag_1,
        lag_2,
        lag_3,
        rolling_windows,
        predict_btn,
    ) = sidebar.render()

    # Preparar datos de entrada
    # Las rolling_windows configuradas en el sidebar se envían a la API
    # para calcular las features dinámicamente
    input_data = {
        "shop_cluster": shop_cluster,
        "item_category_id": item_category_id,
        "item_price": item_price,
        "item_cnt_lag_1": lag_1,
        "item_cnt_lag_2": lag_2,
        "item_cnt_lag_3": lag_3,
        "rolling_windows": rolling_windows,
    }

    # Renderizar tabs principales
    tabs = ["Predicción", "Monitoreo", "Análisis Técnico", "Acerca de"]
    tab_objects = st.tabs(tabs)

    with tab_objects[0]:
        prediction_view = PredictionView(prediction_service, shap_renderer)
        prediction_view.render(predict_btn, input_data)

    with tab_objects[1]:
        monitoring_view = MonitoringView(prediction_service)
        monitoring_view.render()

    with tab_objects[2]:
        technical_view = TechnicalAnalysisView()
        technical_view.render()

    with tab_objects[3]:
        about_view = AboutView(prediction_service)
        about_view.render()


if __name__ == "__main__":
    main()
