"""Vista de predicción de demanda."""

from typing import Dict, Any, Literal
import streamlit as st
import shap

from ..components import ChartBuilder, SHAPRenderer, DataFrameBuilder
from ..services import PredictionService, TrendAnalyzer
from ..config import CHART_COLORS


class PredictionView:
    """Vista de predicción de demanda."""

    def __init__(
        self,
        prediction_service: PredictionService,
        shap_renderer: SHAPRenderer,
    ):
        """
        Inicializa la vista de predicción.

        Args:
            prediction_service: Servicio de predicciones
            shap_renderer: Renderizador de gráficos SHAP
        """
        self.prediction_service = prediction_service
        self.shap_renderer = shap_renderer
        self.chart_builder = ChartBuilder()
        self.df_builder = DataFrameBuilder()
        self.trend_analyzer = TrendAnalyzer()

    def render(self, predict_btn: bool, input_data: Dict[str, Any]) -> None:
        """
        Renderiza la vista de predicción.

        Args:
            predict_btn: Estado del botón de predicción
            input_data: Datos de entrada del formulario
        """
        st.header(":material/lightbulb: Análisis Predictivo")

        if predict_btn:
            self._render_prediction_results(input_data)
        else:
            self._render_waiting_state()

    def _render_prediction_results(self, input_data: Dict[str, Any]) -> None:
        """Renderiza los resultados de la predicción."""
        with st.spinner("Procesando en el Backend..."):
            prediction = self.prediction_service.predict(input_data)

        col_kpi, col_shap = st.columns([1, 2])

        with col_kpi:
            self._render_kpi_section(prediction, input_data["item_cnt_lag_1"])

        with col_shap:
            self._render_shap_section(input_data)

        st.divider()
        self._render_temporal_projection(
            input_data["item_cnt_lag_3"],
            input_data["item_cnt_lag_2"],
            input_data["item_cnt_lag_1"],
            prediction,
        )

    def _render_kpi_section(self, prediction: float, last_value: float) -> None:
        """Renderiza la sección de KPIs."""
        st.markdown("#### Proyección")
        delta = self.trend_analyzer.calculate_delta(prediction, last_value)
        message, icon, delta_mode = self.trend_analyzer.get_trend_status(delta)

        if delta > 0:
            st.success(message, icon=icon)
        elif delta < 0:
            st.error(message, icon=icon)
        else:
            st.info(message, icon=icon)

        # Convertir string a tipo literal
        delta_color: Literal["normal", "inverse", "off"] = delta_mode  # type: ignore
        st.metric(
            label="Ventas Estimadas (Mes t)",
            value=f"{prediction:.2f} u.",
            delta=f"{delta:+.2f} vs mes anterior",
            delta_color=delta_color,
            border=True,
        )

    def _render_shap_section(self, input_data: Dict[str, Any]) -> None:
        """Renderiza la sección de análisis SHAP."""
        st.markdown("#### :material/fact_check: Factores de Influencia (SHAP)")

        try:
            shap_values, feat_df, expected_value = self.prediction_service.calculate_shap_values(
                input_data
            )
            self.shap_renderer.render(
                shap.force_plot(
                    expected_value,
                    shap_values[0],
                    feat_df.iloc[0],
                    plot_cmap=[CHART_COLORS["shap_negative"], CHART_COLORS["shap_positive"]],
                ),
                height=140,
            )
        except (ValueError, TypeError, AttributeError) as e:
            st.error(f"Error generando SHAP: {e}", icon=":material/error:")

        st.info(
            "**Interpretación:** Verde = Empuja la venta hacia arriba | Rojo = Empuja hacia abajo.",
            icon=":material/help_outline:",
        )

    def _render_temporal_projection(
        self, lag_3: int, lag_2: int, lag_1: int, prediction: float
    ) -> None:
        """Renderiza la proyección temporal."""
        st.subheader(":material/timeline: Proyección Temporal")
        st.caption("Continuidad de la serie de tiempo: Histórico (Gris) vs Predicción (Color).")

        lags_df = self.df_builder.create_temporal_dataframe(lag_3, lag_2, lag_1, prediction)
        colors = self.trend_analyzer.get_chart_colors(
            prediction,
            lag_1,
            CHART_COLORS["historical"],
            CHART_COLORS["positive"],
            CHART_COLORS["negative"],
        )

        fig_trend = self.chart_builder.create_temporal_chart(lags_df, colors)
        st.plotly_chart(fig_trend, width="stretch")

    def _render_waiting_state(self) -> None:
        """Renderiza el estado de espera."""
        st.container(border=True).markdown(
            """
            <div style="text-align: center; padding: 20px;">
                <h3>Esperando parámetros</h3>
                <p>Configure las variables en el menú lateral y presione <b>Calcular Demanda</b>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
