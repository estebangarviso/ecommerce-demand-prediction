"""Vista de predicción de demanda."""

from typing import Dict, Any, Literal
import streamlit as st
import shap
import numpy as np

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
        """Inicializa la vista con los servicios de predicción y rendering."""
        self.prediction_service = prediction_service
        self.shap_renderer = shap_renderer
        self.chart_builder = ChartBuilder()
        self.df_builder = DataFrameBuilder()
        self.trend_analyzer = TrendAnalyzer()

    def render(self, predict_btn: bool, input_data: Dict[str, Any]) -> None:
        """Renderiza la vista principal de predicción."""
        st.header(":material/lightbulb: Análisis Predictivo")

        if predict_btn:
            self._render_prediction_results(input_data)
        else:
            self._render_waiting_state()

    def _render_prediction_results(self, input_data: Dict[str, Any]) -> None:
        """Renderiza los resultados de la predicción."""
        from ..state_manager import SessionStateManager

        # Verificar compatibilidad de rolling windows
        rolling_windows = input_data.get("rolling_windows", [3, 6])
        current_model_windows = SessionStateManager.get_current_rolling_windows()

        if rolling_windows != current_model_windows:
            st.warning(
                f"**Reentrenando modelo con ventanas {rolling_windows}...**\n\n"
                f"El modelo actual fue entrenado con ventanas `{current_model_windows}`. "
                f"Se está reentrenando automáticamente con `{rolling_windows}`.\n\n"
                f"**Este proceso puede tomar varios minutos. Por favor espera...**",
                icon=":material/autorenew:",
            )

            # Intentar reentrenar
            if not self._retrain_model(rolling_windows):
                return

            # Actualizar estado con las nuevas rolling windows
            SessionStateManager.update_rolling_windows(rolling_windows)

            st.success(
                "Modelo reentrenado exitosamente. Recargando sistema...",
                icon=":material/check_circle:",
            )

            # Invalidar cache de recursos para recargar modelos
            st.cache_resource.clear()

            # Forzar re-renderizado completo para recargar modelos
            st.rerun()

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
        from ..state_manager import SessionStateManager

        st.markdown("#### :material/fact_check: Factores de Influencia (SHAP)")

        # Verificar si las rolling windows coinciden con el modelo actual
        rolling_windows = input_data.get("rolling_windows", [3, 6])
        current_model_windows = SessionStateManager.get_current_rolling_windows()

        if rolling_windows != current_model_windows:
            st.warning(
                f"SHAP no disponible durante el cambio de configuración.\n\n"
                f"El modelo fue reentrenado con `{rolling_windows}` pero el análisis SHAP "
                f"aún usa la configuración anterior `{current_model_windows}`.\n\n"
                f"Presiona 'Calcular Demanda' nuevamente para actualizar.",
                icon=":material/sync_problem:",
            )
            return

        try:
            # IMPORTANTE: Calcular rolling features si no están presentes
            complete_input = input_data.copy()

            # Calcular aproximaciones de rolling features desde los lags
            lags = [
                input_data["item_cnt_lag_1"],
                input_data["item_cnt_lag_2"],
                input_data["item_cnt_lag_3"],
            ]

            for window in rolling_windows:
                mean_key = f"rolling_mean_{window}"
                std_key = f"rolling_std_{window}"

                # Si no están en el input, calcular aproximaciones
                if mean_key not in complete_input:
                    complete_input[mean_key] = float(np.mean(lags))
                if std_key not in complete_input:
                    complete_input[std_key] = float(np.std(lags))

            # Calcular SHAP con input completo
            shap_values, feat_df, expected_value = self.prediction_service.calculate_shap_values(
                complete_input
            )

            # Detectar si el modelo tiene features nuevas o antiguas
            num_features = len(feat_df.columns)
            has_pricing_features = num_features > (6 + len(rolling_windows) * 2)

            if has_pricing_features:
                # Modelo nuevo con pricing features
                expected_features = 14 + (len(rolling_windows) * 2)
                feature_type = "pricing (precio relativo, descuento, elasticidad)"
            else:
                # Modelo antiguo sin pricing features
                expected_features = 6 + (len(rolling_windows) * 2)
                feature_type = "básicas"

            if num_features != expected_features:
                st.info(
                    f"SHAP calculado con {num_features} features ({feature_type}). "
                    f"Modelo detectado: {'Nuevo' if has_pricing_features else 'Antiguo'}.\n\n"
                    f"Features: {', '.join(feat_df.columns.tolist())}",
                    icon=":material/info:",
                )

            self.shap_renderer.render(
                shap.force_plot(
                    expected_value,
                    shap_values[0],
                    feat_df.iloc[0],
                    plot_cmap=[CHART_COLORS["shap_negative"], CHART_COLORS["shap_positive"]],
                ),
                height=150,
            )

            # Expandible con detalles de features
            with st.expander("Ver features utilizadas", icon=":material/search:"):
                st.caption("**Features incluidas en el análisis SHAP:**")
                # Crear DataFrame con valores y contribuciones SHAP
                feature_impact = []
                for i, col in enumerate(feat_df.columns):
                    impact_value = shap_values[0][i]
                    if impact_value > 0:
                        direction = ":green[:material/arrow_upward:]"
                    elif impact_value < 0:
                        direction = ":red[:material/arrow_downward:]"
                    else:
                        direction = ":gray[:material/remove:]"
                    feature_impact.append(
                        {
                            "Feature": col,
                            "Valor": f"{feat_df.iloc[0][col]:.2f}",
                            "Impacto SHAP": f"{impact_value:+.4f}",
                            "Dirección": direction,
                        }
                    )

                # Ordenar por impacto absoluto (mayor a menor)
                feature_impact.sort(key=lambda x: abs(float(x["Impacto SHAP"])), reverse=True)

                # Mostrar tabla
                import pandas as pd

                impact_df = pd.DataFrame(feature_impact)
                st.table(impact_df)

                st.caption(f"**Valor base del modelo:** {expected_value:.4f}")
                st.caption(f"**Suma de contribuciones:** {sum(shap_values[0]):.4f}")
                st.caption(f"**Predicción final:** {expected_value + sum(shap_values[0]):.4f}")

        except (ValueError, TypeError, AttributeError) as e:
            st.error(f"Error generando SHAP: {e}", icon=":material/error:")

        # Detectar tipo de modelo para mensaje informativo
        try:
            # Re-calcular para obtener num_features (sin triggear error)
            test_input = input_data.copy()
            lags = [
                input_data["item_cnt_lag_1"],
                input_data["item_cnt_lag_2"],
                input_data["item_cnt_lag_3"],
            ]
            for window in rolling_windows:
                if f"rolling_mean_{window}" not in test_input:
                    test_input[f"rolling_mean_{window}"] = float(np.mean(lags))
                if f"rolling_std_{window}" not in test_input:
                    test_input[f"rolling_std_{window}"] = float(np.std(lags))

            _, test_df, _ = self.prediction_service.calculate_shap_values(test_input)
            num_features = len(test_df.columns)
            has_pricing = num_features > (6 + len(rolling_windows) * 2)

            if has_pricing:
                info_text = (
                    f"Incluye {num_features} features: lags normalizados, pricing (precio relativo, "
                    "descuento, elasticidad), cluster, categoría y ventanas rolling."
                )
            else:
                info_text = (
                    f"Incluye {num_features} features: lags básicos, precio, "
                    "cluster, categoría y ventanas rolling. "
                    "**Reentrena el modelo** para ver features de pricing."
                )
        except:
            # Fallback si hay error
            info_text = "Features: lags, precio, categoría, cluster y ventanas rolling."

        st.info(
            f"**Interpretación:** Verde = Empuja la venta hacia arriba | Rojo = Empuja hacia abajo. {info_text}",
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

    def _retrain_model(self, rolling_windows: list) -> bool:
        """Llama al endpoint de reentrenamiento de la API.

        Returns:
            True si el reentrenamiento fue exitoso, False en caso contrario.
        """
        import httpx

        try:
            api_url = self.prediction_service.api_url

            with st.spinner(
                f"🔄 Reentrenando modelo con ventanas {rolling_windows}. Esto puede tomar varios minutos..."
            ):
                with httpx.Client(timeout=600.0) as client:  # 10 minutos timeout
                    response = client.post(
                        f"{api_url}/retrain",
                        json={"rolling_windows": rolling_windows, "use_balancing": False},
                    )
                    response.raise_for_status()
                    result = response.json()

                    st.success(f"✅ {result['message']}", icon=":material/check_circle:")
                    return True

        except httpx.TimeoutException:
            st.error(
                "⏱️ El reentrenamiento está tardando demasiado. "
                "El proceso continúa en segundo plano. "
                "Por favor espera unos minutos e intenta nuevamente.",
                icon=":material/timer:",
            )
            return False
        except httpx.HTTPError as e:
            st.error(
                f"❌ Error al reentrenar el modelo: {e}\n\n"
                "Verifica que la API esté ejecutándose correctamente.",
                icon=":material/error:",
            )
            return False
        except Exception as e:
            st.error(f"❌ Error inesperado: {e}", icon=":material/error:")
            return False

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
