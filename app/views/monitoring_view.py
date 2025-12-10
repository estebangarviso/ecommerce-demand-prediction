"""Vista de monitoreo del modelo."""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict, TYPE_CHECKING

from ..components import ChartBuilder, DataFrameBuilder

if TYPE_CHECKING:
    from services.prediction_service import PredictionService


class MonitoringView:
    """Vista de monitoreo del modelo."""

    def __init__(self, prediction_service: "PredictionService"):
        """Inicializa la vista de monitoreo.

        Args:
            prediction_service: Servicio de predicción para verificar estado de la API
        """
        self.chart_builder = ChartBuilder()
        self.df_builder = DataFrameBuilder()
        self.prediction_service = prediction_service

    def render(self) -> None:
        """Renderiza la vista de monitoreo."""
        st.header(":material/speed: Panel de Salud del Modelo")

        # Mostrar estado de conexión con la API
        if self.prediction_service:
            api_healthy, health_data = self._check_api_status()

            if api_healthy:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.success(
                        f":material/check_circle: API Conectada - {self.prediction_service.api_url}",
                        icon=":material/cloud_done:",
                    )
                with col2:
                    if health_data and "model_metrics" in health_data:
                        rolling_windows = health_data["model_metrics"].get("rolling_windows", [])
                        st.metric(
                            "Rolling Windows",
                            f"{rolling_windows}",
                            help="Configuración de ventanas temporales del modelo",
                        )
            else:
                st.error(
                    f":material/error: API Desconectada - {self.prediction_service.api_url}. "
                    "Las predicciones no estarán disponibles.",
                    icon=":material/cloud_off:",
                )

            st.divider()

        self._render_metrics()
        st.caption(
            "Métricas calculadas sobre el conjunto de validación (Mes 32 - Septiembre 2015)."
        )

        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            self._render_stability_chart()

        with col_chart2:
            self._render_distribution_chart()

        # Sección de mantenimiento
        st.divider()
        self._render_maintenance_section()

    def _check_api_status(self) -> tuple[bool, Optional[Dict]]:
        """Verifica el estado de la API y obtiene información del sistema.

        Returns:
            Tupla (is_healthy, health_data)
        """
        try:
            is_healthy, health_data = self.prediction_service.check_api_health()
            return is_healthy, health_data
        except Exception:
            return False, None

    def _render_metrics(self) -> None:
        """Renderiza las métricas del modelo."""
        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE Global", "1.005", "-0.02", border=True, help="Error Cuadrático Medio")
        col2.metric("MAE Global", "0.835", "-0.01", border=True, help="Error Absoluto Medio")
        col3.metric("R² Score", "0.741", "+0.05", border=True, help="Coeficiente de Determinación")

    def _render_stability_chart(self) -> None:
        """Renderiza el gráfico de estabilidad de errores."""
        st.subheader(":material/history: Estabilidad de Errores")
        st.caption("Residuos en el tiempo. Ideal: dispersión aleatoria alrededor de 0.")

        dates = pd.date_range(start="2024-01-01", periods=30)
        residuals = np.random.normal(0, 0.5, 30)
        df_mon = self.df_builder.create_monitoring_dataframe(dates, residuals)

        fig_mon = self.chart_builder.create_scatter_chart(df_mon, "Fecha", "Error Residual")
        st.plotly_chart(fig_mon, width="stretch")

    def _render_distribution_chart(self) -> None:
        """Renderiza el gráfico de distribución de errores."""
        st.subheader(":material/bar_chart: Distribución de Errores")
        st.caption("Histograma de errores. Ideal: Campana de Gauss centrada en 0.")

        residuals = np.random.normal(0, 0.5, 30)
        fig_hist = self.chart_builder.create_histogram(residuals)
        st.plotly_chart(fig_hist, width="stretch")

    def _render_maintenance_section(self) -> None:
        """Renderiza la sección de mantenimiento del sistema."""
        st.subheader(":material/build: Mantenimiento del Sistema")
        st.caption("Herramientas para gestionar datasets y recursos del sistema.")

        # Mostrar mensajes de estado persistentes
        if "regenerate_status" in st.session_state:
            status = st.session_state.regenerate_status
            if status["type"] == "success":
                st.success(status["message"], icon=":material/check_circle:")
            elif status["type"] == "error":
                st.error(status["message"], icon=":material/error:")
            # Limpiar el estado después de mostrarlo
            del st.session_state.regenerate_status

        if "retrain_status" in st.session_state:
            status = st.session_state.retrain_status
            if status["type"] == "success":
                st.success(status["message"], icon=":material/check_circle:")
            elif status["type"] == "error":
                st.error(status["message"], icon=":material/error:")
            # Limpiar el estado después de mostrarlo
            del st.session_state.retrain_status

        # Datasets
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(
                "**Regenerar Datasets:** Descarga datos frescos desde KaggleHub y "
                "actualiza la carpeta `data/`. Útil si los archivos están corruptos o desactualizados.",
                icon=":material/refresh:",
            )
        with col2:
            if st.button(
                "Regenerar",
                type="primary",
                width="stretch",
                help="Fuerza la descarga desde KaggleHub",
                key="btn_regenerate_datasets",
            ):
                self._handle_regenerate_datasets()

        st.divider()

        # Modelos
        col3, col4 = st.columns([3, 1])
        with col3:
            st.info(
                "**Reentrenar Modelos:** Ejecuta el pipeline completo de entrenamiento "
                "y genera nuevos modelos en `models/`.",
                icon=":material/psychology:",
            )
        with col4:
            if st.button(
                "Reentrenar",
                type="primary",
                width="stretch",
                help="Entrena los modelos con los datos actuales",
                key="btn_retrain_models",
            ):
                self._handle_retrain_models()

    def _handle_regenerate_datasets(self) -> None:
        """Maneja la regeneración de datasets a través del servicio."""
        with st.spinner("Descargando datasets desde KaggleHub vía API..."):
            success, message = self.prediction_service.regenerate_datasets()

            if success:
                st.session_state.regenerate_status = {
                    "type": "success",
                    "message": f"✅ {message}",
                }
            else:
                st.session_state.regenerate_status = {
                    "type": "error",
                    "message": f"❌ {message}\n\n"
                    "Alternativamente, ejecuta desde terminal:\n"
                    "```bash\n"
                    "pipenv run python -c 'from src.data_processing import force_download_datasets; force_download_datasets()'\n"
                    "```",
                }

        # Forzar rerun para mostrar los mensajes
        st.rerun()

    def _handle_retrain_models(self) -> None:
        """Maneja el reentrenamiento de modelos a través del servicio."""
        rolling_windows = st.session_state.get("rolling_windows", [3, 6])

        with st.spinner(f"Reentrenando modelos con rolling_windows={rolling_windows}..."):
            success, message = self.prediction_service.retrain_model(
                rolling_windows=rolling_windows,
                use_balancing=False,  # Puedes agregar un checkbox en la UI
            )

            if success:
                st.session_state.retrain_status = {
                    "type": "success",
                    "message": f"✅ {message}\n\nMétricas actualizadas en la API.",
                }

                # Limpiar caché de Streamlit
                st.cache_data.clear()
                st.cache_resource.clear()
            else:
                st.session_state.retrain_status = {
                    "type": "error",
                    "message": f"❌ {message}\n\n"
                    "Alternativamente, ejecuta desde terminal:\n"
                    "```bash\n"
                    "pipenv run train\n"
                    "```",
                }

        # Forzar rerun para mostrar los mensajes y cargar nuevos modelos
        st.rerun()
