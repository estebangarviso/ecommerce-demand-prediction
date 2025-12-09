"""Gestión del estado de la aplicación Streamlit."""

from typing import Dict, Optional, Any
import streamlit as st


class SessionStateManager:
    """Gestiona el estado de la sesión de Streamlit siguiendo el patrón Singleton."""

    # Keys para el estado
    PRICE_SLIDER = "price_slider"
    PRICE_MIN = "price_min"
    PRICE_MAX = "price_max"
    LAST_CATEGORY = "last_category"
    SELECTED_CATEGORY = "selected_category_key"
    ROLLING_WINDOWS = "rolling_windows"

    @staticmethod
    def initialize_state(
        default_price: float,
        default_min: float,
        default_max: float,
        cat_prices: Optional[Dict[int, float]],
        first_category: Optional[int],
    ) -> None:
        """Inicializa el estado de la sesión con valores por defecto."""
        if SessionStateManager.PRICE_SLIDER not in st.session_state:
            if cat_prices and first_category and first_category in cat_prices:
                st.session_state[SessionStateManager.PRICE_SLIDER] = float(
                    cat_prices[first_category]
                )
            else:
                st.session_state[SessionStateManager.PRICE_SLIDER] = default_price

        if SessionStateManager.PRICE_MIN not in st.session_state:
            st.session_state[SessionStateManager.PRICE_MIN] = default_min

        if SessionStateManager.PRICE_MAX not in st.session_state:
            st.session_state[SessionStateManager.PRICE_MAX] = default_max

        if SessionStateManager.LAST_CATEGORY not in st.session_state:
            st.session_state[SessionStateManager.LAST_CATEGORY] = None

    @staticmethod
    def update_price_range(avg_price: float, min_multiplier: float, max_multiplier: float) -> None:
        """Actualiza el rango de precios basado en el precio promedio."""
        st.session_state[SessionStateManager.PRICE_SLIDER] = avg_price
        st.session_state[SessionStateManager.PRICE_MIN] = max(0.0, avg_price * min_multiplier)
        st.session_state[SessionStateManager.PRICE_MAX] = avg_price * max_multiplier

    @staticmethod
    def reset_price_range(default_price: float, default_min: float, default_max: float) -> None:
        """Resetea el rango de precios a valores por defecto."""
        st.session_state[SessionStateManager.PRICE_SLIDER] = default_price
        st.session_state[SessionStateManager.PRICE_MIN] = default_min
        st.session_state[SessionStateManager.PRICE_MAX] = default_max

    @staticmethod
    def get_value(key: str, default: Any = None) -> Any:
        """Obtiene un valor del estado de la sesión."""
        return st.session_state.get(key, default)

    @staticmethod
    def set_value(key: str, value: Any) -> None:
        """Establece un valor en el estado de la sesión."""
        st.session_state[key] = value

    @staticmethod
    def get_current_rolling_windows() -> list:
        """Obtiene las rolling windows actuales del modelo."""
        return st.session_state.get(SessionStateManager.ROLLING_WINDOWS, [3, 6])

    @staticmethod
    def update_rolling_windows(rolling_windows: list) -> None:
        """Actualiza las rolling windows en el estado de sesión."""
        st.session_state[SessionStateManager.ROLLING_WINDOWS] = rolling_windows
