"""Vista de predicción de demanda."""

from typing import Dict, Any, Literal, Optional
import pandas as pd
import streamlit as st
import shap
import numpy as np

from app.ui_components import Table

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
        from ..state_manager import SessionStateManager

        st.header(":material/lightbulb: Análisis Predictivo")

        # Verificar si hay una predicción pendiente después de reentrenamiento
        has_pending = SessionStateManager.has_pending_prediction()

        if predict_btn or has_pending:
            # Limpiar el flag si estaba pendiente
            if has_pending:
                SessionStateManager.clear_pending_prediction()

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

            # IMPORTANTE: Actualizar estado ANTES de reentrenar
            # Esto asegura que cuando se recargue después del rerun, tenga las ventanas correctas
            SessionStateManager.update_rolling_windows(rolling_windows)

            # Marcar que hay una predicción pendiente para mostrar después del rerun
            SessionStateManager.set_pending_prediction(True)

            # Intentar reentrenar
            if not self._retrain_model(rolling_windows):
                # Si falla el reentrenamiento, revertir el estado
                SessionStateManager.update_rolling_windows(current_model_windows)
                SessionStateManager.clear_pending_prediction()
                return

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

        # Calcular SHAP una sola vez para usar en ambas columnas
        shap_explanation = self._calculate_shap_explanation(input_data)

        col_kpi, col_shap = st.columns([1, 2])

        with col_kpi:
            self._render_kpi_section(prediction, input_data["item_cnt_lag_1"], shap_explanation)

        with col_shap:
            self._render_shap_section(input_data, shap_explanation)

        st.divider()
        self._render_temporal_projection(
            input_data["item_cnt_lag_3"],
            input_data["item_cnt_lag_2"],
            input_data["item_cnt_lag_1"],
            prediction,
        )

    def _render_kpi_section(
        self,
        prediction: float,
        last_value: float,
        shap_explanation: Optional[shap.Explanation] = None,
    ) -> None:
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

        # Interpretación en lenguaje natural (si hay datos SHAP)
        if shap_explanation is not None:
            self._render_textual_interpretation(shap_explanation)

    def _calculate_shap_explanation(self, input_data: Dict[str, Any]) -> Optional[shap.Explanation]:
        """Calcula el objeto SHAP Explanation con todas las features necesarias."""
        from ..state_manager import SessionStateManager

        rolling_windows = input_data.get("rolling_windows", [3, 6])
        current_model_windows = SessionStateManager.get_current_rolling_windows()

        if rolling_windows != current_model_windows:
            return None

        try:
            # Calcular rolling features si no están presentes
            complete_input = input_data.copy()

            lags = [
                input_data["item_cnt_lag_1"],
                input_data["item_cnt_lag_2"],
                input_data["item_cnt_lag_3"],
            ]

            for window in rolling_windows:
                mean_key = f"rolling_mean_{window}"
                std_key = f"rolling_std_{window}"

                if mean_key not in complete_input:
                    complete_input[mean_key] = float(np.mean(lags))
                if std_key not in complete_input:
                    complete_input[std_key] = float(np.std(lags))

            # Calcular SHAP con input completo
            return self.prediction_service.calculate_shap_values(complete_input)
        except (ValueError, TypeError, AttributeError) as e:
            st.error(f"Error calculando SHAP: {type(e).__name__}: {str(e)}", icon=":material/bug_report:")
            return None

    def _render_shap_section(
        self, input_data: Dict[str, Any], shap_explanation: Optional[shap.Explanation] = None
    ) -> None:
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

        if shap_explanation is None:
            st.error("No se pudo calcular SHAP para esta configuración.", icon=":material/error:")
            return

        try:
            # Renderizar gráfico de cascada (waterfall plot)
            st.markdown("**:material/waterfall_chart: Análisis de Contribución por Variable**")

            import matplotlib.pyplot as plt
            from PIL import Image
            
            # Aumentar límite de PIL para imágenes grandes generadas por SHAP
            Image.MAX_IMAGE_PIXELS = None
            
            # Crear figura con tamaño y DPI controlados
            fig, _ = plt.subplots(figsize=(10, 6), dpi=100)
            shap.plots.waterfall(shap_explanation[0], show=False)
            st.pyplot(fig, width="stretch")
            plt.close()

            # Mensaje explicativo sobre cómo interpretar el gráfico
            st.info(
                "**Cómo interpretar este gráfico:** "
                "Las barras **azules** incrementan la predicción de demanda (impulsan las ventas). "
                "Las barras **rojas** la disminuyen (frenan las ventas). "
                "El gráfico muestra cómo cada variable contribuye para llegar desde el valor base E[f(X)] "
                "hasta la predicción final f(x).",
                icon=":material/info:",
            )

            # Expandible con detalles técnicos de features
            with st.expander("Ver detalles técnicos de features", icon=":material/analytics:"):
                st.caption("**:material/table_chart: Tabla detallada de contribuciones SHAP:**")

                # Extraer datos del objeto Explanation
                shap_values = shap_explanation.values[0]

                # Los nombres de features están en el atributo feature_names del Explanation
                feature_names = shap_explanation.feature_names

                # Los valores de las features están en data
                feat_data = shap_explanation.data
                if hasattr(feat_data, "iloc"):
                    feature_values = feat_data.iloc[0].tolist()
                else:
                    feature_values = (
                        feat_data[0].tolist()
                        if hasattr(feat_data[0], "tolist")
                        else list(feat_data[0])
                    )

                total_impact = sum(abs(shap_values))

                feature_impact = []
                for i, feature_name in enumerate(feature_names):
                    impact_value = shap_values[i]
                    impact_percentage = (
                        (abs(impact_value) / total_impact) if total_impact > 0 else 0
                    )

                    feature_impact.append(
                        {
                            "feature_name": feature_name,
                            "feature_value": f"{feature_values[i]:.2f}",
                            "impacto_sharp": f"{impact_value:+.4f}",
                            "impacto_sharp_perc": Table.create_bar(
                                impact_percentage, max_width=75, height=20
                            ),
                        }
                    )

                # Ordenar por impacto absoluto descendente
                feature_impact.sort(
                    key=lambda x: abs(float(x["impacto_sharp"])),
                    reverse=True,
                )

                impact_df = pd.DataFrame(feature_impact)
                impact_table = Table(None, auto_align=True, width="stretch")
                impact_table.render(
                    impact_table.get_table(impact_df)
                    .tab_spanner(label="Feature", columns=["feature_name", "feature_value"])
                    .tab_spanner(label="SHAP", columns=["impacto_sharp", "impacto_sharp_perc"])
                    .cols_label(
                        feature_name="Nombre",
                        feature_value="Valor",
                        impacto_sharp="Valor",
                        impacto_sharp_perc="%",
                    )
                    .opt_stylize(style=6)
                )

                st.caption(
                    f":material/insights: **Valor base del modelo:** {shap_explanation.base_values[0]:.4f}"
                )
                st.caption(
                    f":material/calculate: **Suma de contribuciones:** {sum(shap_values):.4f}"
                )
                st.caption(
                    f":material/done_all: **Predicción final:** {shap_explanation.base_values[0] + sum(shap_values):.4f}"
                )

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

            test_explanation = self.prediction_service.calculate_shap_values(test_input)
            num_features = (
                len(test_explanation.data.columns)
                if hasattr(test_explanation.data, "columns")
                else len(test_explanation.values[0])
            )
            has_pricing = num_features > (6 + len(rolling_windows) * 2)

            if has_pricing:
                info_text = (
                    f":material/check_circle: Modelo avanzado con **{num_features} features**: "
                    "lags normalizados, pricing inteligente (precio relativo, descuento, elasticidad), "
                    "cluster de tienda, categoría y ventanas rolling."
                )
            else:
                info_text = (
                    f":material/info: Modelo básico con **{num_features} features**: "
                    "lags históricos, precio, cluster, categoría y ventanas rolling. "
                    "Reentrena el modelo para activar features de pricing avanzadas."
                )
        except (ValueError, TypeError, AttributeError):
            info_text = ":material/dataset: Features estándar: lags, precio, categoría, cluster y ventanas rolling."

        st.caption(info_text)

    def _render_textual_interpretation(self, explanation: shap.Explanation) -> None:
        """Genera una interpretación en lenguaje natural de los valores SHAP.

        Analiza las contribuciones más significativas y las traduce a insights
        accionables para usuarios no técnicos.
        """
        st.markdown("---")
        st.markdown("**:material/psychology: Interpretación en Lenguaje Natural**")

        try:
            shap_values = explanation.values[0]

            # Obtener nombres reales de las features del objeto Explanation
            feature_names = explanation.feature_names

            # Obtener valores de las features
            feat_data = explanation.data
            if hasattr(feat_data, "iloc"):
                feature_values = feat_data.iloc[0].tolist()
            else:
                feature_values = (
                    feat_data[0].tolist() if hasattr(feat_data[0], "tolist") else list(feat_data[0])
                )

            # Identificar las contribuciones más relevantes
            max_positive_idx = np.argmax(shap_values)
            min_negative_idx = np.argmin(shap_values)

            max_positive_value = shap_values[max_positive_idx]
            min_negative_value = shap_values[min_negative_idx]

            max_positive_feature = feature_names[max_positive_idx]
            min_negative_feature = feature_names[min_negative_idx]

            max_positive_feat_value = feature_values[max_positive_idx]
            min_negative_feat_value = feature_values[min_negative_idx]

            interpretations = []

            # Analizar el factor más positivo
            if max_positive_value > 0.01:
                if "lag_1" in max_positive_feature.lower():
                    interpretations.append(
                        f":green[**✓ Impulsor clave:**] Las ventas del mes anterior "
                        f"(**{max_positive_feat_value:.1f} unidades**) están generando un fuerte impulso "
                        f"en la predicción actual (**+{max_positive_value:.2f} puntos**). "
                        "Esta inercia positiva sugiere una demanda estable y confiable."
                    )
                elif (
                    "price" in max_positive_feature.lower()
                    and "discount" not in max_positive_feature.lower()
                ):
                    interpretations.append(
                        f":green[**✓ Precio competitivo:**] El precio configurado "
                        f"(**${max_positive_feat_value:.0f}**) está favoreciendo significativamente "
                        f"la demanda (**+{max_positive_value:.2f} puntos**). "
                        "Este nivel de precio resulta atractivo para los clientes y potencia las ventas."
                    )
                elif "category" in max_positive_feature.lower():
                    interpretations.append(
                        f":green[**✓ Categoría demandada:**] La categoría seleccionada "
                        f"muestra un comportamiento históricamente fuerte, contribuyendo positivamente "
                        f"(**+{max_positive_value:.2f} puntos**). Este tipo de productos tiene alta aceptación en el mercado."
                    )
                elif "rolling" in max_positive_feature.lower():
                    interpretations.append(
                        f":green[**✓ Tendencia favorable:**] El análisis de la tendencia temporal "
                        f"indica un momentum positivo reciente (**+{max_positive_value:.2f} puntos**). "
                        "Los patrones de venta recientes están favoreciendo la predicción."
                    )
                elif "discount" in max_positive_feature.lower():
                    interpretations.append(
                        f":green[**✓ Descuento efectivo:**] La estrategia de descuento actual "
                        f"está impulsando fuertemente la demanda (**+{max_positive_value:.2f} puntos**). "
                        "Los clientes están respondiendo positivamente a esta promoción."
                    )
                else:
                    interpretations.append(
                        f":green[**✓ Factor positivo:**] `{max_positive_feature}` "
                        f"(valor: **{max_positive_feat_value:.2f}**) está incrementando "
                        f"la predicción en **+{max_positive_value:.2f} puntos**."
                    )

            # Analizar el factor más negativo
            if min_negative_value < -0.01:
                if (
                    "price" in min_negative_feature.lower()
                    and "discount" not in min_negative_feature.lower()
                ):
                    interpretations.append(
                        f":red[**⚠ Limitante principal:**] El precio actual "
                        f"(**${min_negative_feat_value:.0f}**) está reduciendo la demanda proyectada "
                        f"(**{min_negative_value:.2f} puntos**). "
                        "Considera ajustar el precio o implementar descuentos para aumentar las ventas."
                    )
                elif "lag" in min_negative_feature.lower():
                    interpretations.append(
                        f":red[**⚠ Historial débil:**] Las ventas previas "
                        f"(**{min_negative_feat_value:.1f} unidades**) están limitando la proyección actual "
                        f"(**{min_negative_value:.2f} puntos**). "
                        "El bajo rendimiento histórico está afectando las expectativas del modelo."
                    )
                elif "cluster" in min_negative_feature.lower():
                    interpretations.append(
                        f":red[**⚠ Perfil de tienda:**] El perfil de la tienda seleccionada "
                        f"está reduciendo la predicción (**{min_negative_value:.2f} puntos**). "
                        "Este tipo de tienda históricamente registra menor volumen de ventas para este producto."
                    )
                elif "rolling" in min_negative_feature.lower():
                    interpretations.append(
                        f":red[**⚠ Tendencia desfavorable:**] El análisis temporal muestra "
                        f"volatilidad o tendencia negativa reciente (**{min_negative_value:.2f} puntos**). "
                        "Los patrones de venta recientes no son favorables."
                    )
                elif "elasticity" in min_negative_feature.lower():
                    interpretations.append(
                        f":red[**⚠ Baja elasticidad:**] La elasticidad precio-demanda "
                        f"está limitando el potencial de ventas (**{min_negative_value:.2f} puntos**). "
                        "Los clientes son sensibles a cambios de precio en este producto."
                    )
                else:
                    interpretations.append(
                        f":red[**⚠ Factor limitante:**] `{min_negative_feature}` "
                        f"(valor: **{min_negative_feat_value:.2f}**) está reduciendo "
                        f"la demanda en **{min_negative_value:.2f} puntos**."
                    )

            # Mostrar las interpretaciones
            if interpretations:
                for interpretation in interpretations:
                    st.markdown(interpretation)
            else:
                st.success(
                    ":material/balance: Las contribuciones de las variables están equilibradas. "
                    "No hay factores dominantes que estén afectando significativamente la predicción.",
                    icon=":material/check_circle:",
                )

            # Resumen cuantitativo
            total_positive = sum([v for v in shap_values if v > 0])
            total_negative = sum([v for v in shap_values if v < 0])

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label=":material/trending_up: Contribución positiva total",
                    value=f"+{total_positive:.2f}",
                    delta="Impulsa la demanda",
                    delta_color="normal",
                )
            with col2:
                st.metric(
                    label=":material/trending_down: Contribución negativa total",
                    value=f"{total_negative:.2f}",
                    delta="Reduce la demanda",
                    delta_color="inverse",
                )

        except Exception as e:
            st.warning(
                f":material/error: No fue posible generar la interpretación textual. "
                f"Detalle técnico: {str(e)}",
                icon=":material/warning:",
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
        """Llama al endpoint de reentrenamiento a través del servicio.

        Returns:
            True si el reentrenamiento fue exitoso, False en caso contrario.
        """
        with st.spinner(
            f"🔄 Reentrenando modelo con ventanas {rolling_windows}. Esto puede tomar varios minutos..."
        ):
            success, message = self.prediction_service.retrain_model(
                rolling_windows=rolling_windows, use_balancing=False
            )

            if success:
                st.success(f"✅ {message}", icon=":material/check_circle:")
                return True
            else:
                st.error(
                    f"❌ {message}\n\n" "Verifica que la API esté ejecutándose correctamente.",
                    icon=":material/error:",
                )
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
