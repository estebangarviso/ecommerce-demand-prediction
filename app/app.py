"""
Sistema Predictivo de Demanda - Aplicación Principal.

Esta aplicación implementa un sistema de predicción de demanda utilizando
técnicas de Machine Learning.
"""

import sys
import os

import streamlit as st

# Configuración de página
st.set_page_config(
    page_title="Sistema Predictivo de Demanda",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Agregar directorio raíz al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference import load_system, get_unique_categories

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
from app.views import PredictionView, MonitoringView, AboutView
from app.ui_components import Sidebar, Header


# Carga de recursos del sistema
@st.cache_data
def load_categories_map():
    """Carga el mapa de categorías desde el sistema."""
    return get_unique_categories()


@st.cache_resource
def load_cached_system():
    """Carga el sistema de predicción (modelo, features, etc.)."""
    return load_system()


def initialize_application():
    """Inicializa la aplicación cargando recursos y configurando el estado."""
    # Cargar recursos
    category_map = load_categories_map()
    model, _features, shap_model, cat_prices = load_cached_system()

    # Validar modelo
    if model is None:
        st.error(
            "Modelo no encontrado. Ejecuta `pipenv run train`.",
            icon=":material/folder_off:",
        )
        st.stop()

    # Inicializar estado
    first_category = list(category_map.keys())[0] if category_map else None
    SessionStateManager.initialize_state(
        DEFAULT_PRICE, DEFAULT_PRICE_MIN, DEFAULT_PRICE_MAX, cat_prices, first_category
    )

    # Sincronizar rolling windows con la API si no está inicializado
    if SessionStateManager.get_value(SessionStateManager.ROLLING_WINDOWS) is None:
        try:
            import httpx

            with httpx.Client(timeout=5.0) as client:
                response = client.get("http://localhost:8000/health")
                if response.status_code == 200:
                    health_data = response.json()
                    api_rolling_windows = health_data.get("model_metrics", {}).get(
                        "rolling_windows", [3, 6]
                    )
                    SessionStateManager.update_rolling_windows(api_rolling_windows)
                else:
                    SessionStateManager.update_rolling_windows([3, 6])
        except Exception:
            # Si falla, usar valor por defecto
            SessionStateManager.update_rolling_windows([3, 6])

    # Inicializar servicios
    pricing_service = PricingService(
        cat_prices,
        DEFAULT_PRICE,
        DEFAULT_PRICE_MIN,
        DEFAULT_PRICE_MAX,
        PRICE_RANGE_MULTIPLIER,
        PRICE_RANGE_MAX_MULTIPLIER,
    )

    # Inicializar servicio de predicción (solo API REST)
    prediction_service = PredictionService(shap_model)

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
    # Inicializar aplicación
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
    tab_pred, tab_monitor, tab_info = st.tabs(["Predicción", "Monitoreo", "Acerca de"])

    with tab_pred:
        prediction_view = PredictionView(prediction_service, shap_renderer)
        prediction_view.render(predict_btn, input_data)

    with tab_monitor:
        monitoring_view = MonitoringView()
        monitoring_view.render()

    with tab_info:
        AboutView.render()


if __name__ == "__main__":
    main()
