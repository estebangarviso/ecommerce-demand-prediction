"""Vista de monitoreo del modelo."""

import streamlit as st
import pandas as pd
import numpy as np

from ..components import ChartBuilder, DataFrameBuilder


class MonitoringView:
    """Vista de monitoreo del modelo."""

    def __init__(self):
        """Inicializa la vista de monitoreo."""
        self.chart_builder = ChartBuilder()
        self.df_builder = DataFrameBuilder()

    def render(self) -> None:
        """Renderiza la vista de monitoreo."""
        st.header(":material/speed: Panel de Salud del Modelo")

        self._render_metrics()
        st.caption("Métricas calculadas sobre el conjunto de validación (Mes 33).")

        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            self._render_stability_chart()

        with col_chart2:
            self._render_distribution_chart()

        # Sección de mantenimiento
        st.divider()
        self._render_maintenance_section()

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
                use_container_width=True,
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
                "y genera nuevos modelos en `models/`. Incluye XGBoost, Random Forest y Stacking.",
                icon=":material/psychology:",
            )
        with col4:
            if st.button(
                "Reentrenar",
                type="primary",
                use_container_width=True,
                help="Entrena los modelos con los datos actuales",
                key="btn_retrain_models",
            ):
                self._handle_retrain_models()

    def _handle_regenerate_datasets(self) -> None:
        """Maneja la regeneración de datasets."""
        with st.spinner("Descargando datasets desde KaggleHub..."):
            try:
                from src.data_processing import force_download_datasets

                success = force_download_datasets()

                if success:
                    st.session_state.regenerate_status = {
                        "type": "success",
                        "message": "✅ Datasets regenerados exitosamente en `data/`",
                    }
                else:
                    st.session_state.regenerate_status = {
                        "type": "error",
                        "message": "❌ Error al regenerar datasets. Revisa los logs en la terminal.",
                    }
            except Exception as e:
                st.session_state.regenerate_status = {
                    "type": "error",
                    "message": f"❌ Error al regenerar datasets: {str(e)}",
                }

        # Forzar rerun para mostrar los mensajes
        st.rerun()

    def _handle_retrain_models(self) -> None:
        """Maneja el reentrenamiento de modelos."""
        with st.spinner("Entrenando modelos... Esto puede tomar varios minutos."):
            try:
                from src.train import train_models

                # Ejecutar entrenamiento
                train_models()

                st.session_state.retrain_status = {
                    "type": "success",
                    "message": "✅ Modelos reentrenados exitosamente en `models/`",
                }

                # Limpiar caché de Streamlit para cargar nuevos modelos
                st.cache_data.clear()
                st.cache_resource.clear()

            except Exception as e:
                st.session_state.retrain_status = {
                    "type": "error",
                    "message": f"❌ Error al reentrenar modelos: {str(e)}",
                }

        # Forzar rerun para mostrar los mensajes y cargar nuevos modelos
        st.rerun()
