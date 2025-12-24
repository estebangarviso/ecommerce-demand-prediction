"""Vista de análisis técnico de modelos."""

import streamlit as st
import pandas as pd
import numpy as np
import os

from ..services.model_analyzer import ModelAnalyzer
from ..services.data_exporter import DataExporter


class TechnicalAnalysisView:
    """Vista de análisis técnico de modelos."""

    def __init__(self):
        """Inicializa la vista de análisis técnico."""
        self.exports_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "exports"
        )
        self.analyzer = None

    def _load_analyzer(self) -> bool:
        """
        Carga el analizador de modelos.

        Returns:
            True si se cargó correctamente, False en caso contrario
        """
        if not os.path.exists(self.exports_dir):
            return False

        try:
            self.analyzer = ModelAnalyzer(self.exports_dir)
            return True
        except Exception as e:
            st.error(f"Error al cargar datos: {str(e)}")
            return False

    def render(self) -> None:
        """Renderiza la vista de análisis técnico."""

        st.header("📊 Análisis Técnico de Modelos")
        st.markdown(
            """
        Análisis exhaustivo de desempeño, distribución de errores y explicabilidad de los modelos entrenados.
        """
        )

        # Export button at the top
        st.divider()
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown("### 🔄 Regenerar Datos de Análisis")
            st.caption(
                "Ejecuta la exportación de métricas, predicciones y análisis SHAP desde los modelos entrenados."
            )

        with col2:
            if st.button("🚀 Exportar Datos", type="primary", use_container_width=True):
                with st.spinner("Exportando datos..."):
                    exporter = DataExporter()
                    success, message = exporter.export_all()

                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)

        st.divider()

        # Check if exports directory exists
        if not os.path.exists(self.exports_dir):
            st.warning(
                """
            ⚠️ **Directorio de exportación no encontrado**
            
            Haz clic en el botón **🚀 Exportar Datos** arriba para generar los archivos necesarios.
            """
            )
            return

        # Load analyzer
        if not self._load_analyzer():
            st.warning(
                """
            ⚠️ **Datos no disponibles**
            
            Haz clic en el botón **🚀 Exportar Datos** arriba para generar los archivos de análisis.
            """
            )
            return

        # Check if data is available
        if self.analyzer.metrics_df is None or not self.analyzer.predictions:
            st.warning(
                """
            ⚠️ **Datos incompletos**
            
            Haz clic en el botón **🚀 Exportar Datos** arriba para regenerar todos los archivos necesarios.
            """
            )
            return

        # Sidebar: Model selection
        st.sidebar.header("⚙️ Análisis Técnico")

        # Get best model (lowest RMSE)
        available_models = list(self.analyzer.predictions.keys())
        best_model_row = self.analyzer.metrics_df.sort_values("rmse").iloc[0]
        best_model_name = best_model_row["model"].lower().replace(" ", "")

        # Find best model in available models (handle name variations)
        default_index = 0
        for i, model in enumerate(available_models):
            if best_model_name in model.lower() or model.lower() in best_model_name:
                default_index = i
                break

        selected_model = st.sidebar.selectbox(
            "Modelo a analizar",
            options=available_models,
            format_func=lambda x: (
                f"🏆 {x.title()}" if x == available_models[default_index] else x.title()
            ),
            index=default_index,
        )

        # Show best model indicator
        if selected_model == available_models[default_index]:
            st.sidebar.success(f"✅ Mejor modelo según RMSE: {best_model_row['rmse']:.4f}")

        # Render analysis tabs
        self._render_analysis_tabs(selected_model)

    def _render_analysis_tabs(self, selected_model: str) -> None:
        """
        Renderiza las pestañas de análisis.

        Args:
            selected_model: Nombre del modelo seleccionado
        """
        # Tab layout
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📈 Métricas Globales",
                "📊 Distribución de Errores",
                "🔍 Análisis por Segmento",
                "🧠 Explicabilidad (SHAP)",
            ]
        )

        with tab1:
            self._render_global_metrics()

        with tab2:
            self._render_error_distribution(selected_model)

        with tab3:
            self._render_segment_analysis(selected_model)

        with tab4:
            self._render_shap_analysis(selected_model)

        # Technical report download
        self._render_report_download(selected_model)

    def _render_global_metrics(self) -> None:
        """Renderiza métricas globales."""
        st.subheader("Comparación de Modelos")

        # Metrics comparison table
        metrics_df = self.analyzer.get_metrics_comparison()

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("**📊 Tabla de Métricas**")

            # Build format dict only for existing columns
            format_dict = {}
            if "rmse" in metrics_df.columns:
                format_dict["rmse"] = "{:.4f}"
            if "mae" in metrics_df.columns:
                format_dict["mae"] = "{:.4f}"
            if "r2" in metrics_df.columns:
                format_dict["r2"] = "{:.4f}"
            if "train_time_s" in metrics_df.columns:
                format_dict["train_time_s"] = "{:.2f}"
            if "model_size_mb" in metrics_df.columns:
                format_dict["model_size_mb"] = "{:.2f}"

            styled_df = metrics_df.style.format(format_dict)

            # Apply highlighting only if columns exist
            highlight_min_cols = [col for col in ["rmse", "mae"] if col in metrics_df.columns]
            highlight_max_cols = [col for col in ["r2"] if col in metrics_df.columns]

            if highlight_min_cols:
                styled_df = styled_df.highlight_min(subset=highlight_min_cols, color="lightgreen")
            if highlight_max_cols:
                styled_df = styled_df.highlight_max(subset=highlight_max_cols, color="lightgreen")

            st.dataframe(styled_df, use_container_width=True)

        with col2:
            st.markdown("**🏆 Mejor Modelo**")
            best_model = metrics_df.iloc[0]
            st.metric("Modelo", best_model["model"])
            st.metric("RMSE", f"{best_model['rmse']:.4f}")
            st.metric("R²", f"{best_model['r2']:.4f}")

            # Optional columns
            if "train_time_s" in best_model.index and not pd.isna(best_model["train_time_s"]):
                st.metric("Tiempo (s)", f"{best_model['train_time_s']:.2f}")

        # RMSE comparison chart
        st.markdown("**📈 Comparación Visual de RMSE**")
        fig_metrics = self.analyzer.plot_metrics_comparison()
        st.plotly_chart(fig_metrics, use_container_width=True)

        # Interpretation
        st.info(
            f"""
        **Interpretación:**
        
        - El modelo **{best_model['model']}** tiene el mejor desempeño con RMSE de **{best_model['rmse']:.4f}**
        - En escala original, esto representa un error relativo de ~**{(np.exp(best_model['rmse'])-1)*100:.2f}%**
        - R² de **{best_model['r2']:.4f}** indica que el modelo explica **{best_model['r2']*100:.2f}%** de la varianza
        """
        )

    def _render_error_distribution(self, model_name: str) -> None:
        """
        Renderiza distribución de errores.

        Args:
            model_name: Nombre del modelo
        """
        st.subheader(f"Distribución de Errores - {model_name.title()}")

        # Get error statistics
        stats = self.analyzer.get_error_statistics(model_name)

        # Display key statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("RMSE", f"{stats['rmse']:.4f}")
        with col2:
            st.metric("MAE", f"{stats['mae']:.4f}")
        with col3:
            st.metric("Media Residual", f"{stats['mean_residual']:.4f}")
        with col4:
            st.metric("Desv. Est.", f"{stats['std_residual']:.4f}")

        # Residuals distribution plot
        st.markdown("**📊 Histograma de Residuos**")
        fig_residuals = self.analyzer.plot_residuals_distribution(model_name)
        st.plotly_chart(fig_residuals, use_container_width=True)

        # Statistical analysis
        st.markdown("**📈 Análisis Estadístico**")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Cuartiles**")
            st.write(f"- 25%: {stats['q25']:.4f}")
            st.write(f"- 50% (mediana): {stats['median_residual']:.4f}")
            st.write(f"- 75%: {stats['q75']:.4f}")
            st.write(f"- IQR: {stats['iqr']:.4f}")

        with col2:
            st.markdown("**Forma de la Distribución**")
            st.write(f"- Asimetría: {stats['skewness']:.2f}")
            st.write(f"- Curtosis: {stats['kurtosis']:.2f}")
            st.write(f"- Rango: [{stats['min_residual']:.4f}, {stats['max_residual']:.4f}]")

        # Interpretation
        if abs(stats["mean_residual"]) < 0.01:
            st.success(
                "✅ **Modelo no sesgado**: La media de residuos cercana a cero indica predicciones balanceadas."
            )
        else:
            bias_dir = "subestima" if stats["mean_residual"] > 0 else "sobreestima"
            st.warning(f"⚠️ **Sesgo detectado**: El modelo {bias_dir} ligeramente las predicciones.")

        if abs(stats["skewness"]) < 0.5:
            st.success("✅ **Distribución simétrica**: Los errores se distribuyen equitativamente.")
        else:
            skew_dir = (
                "positiva (colas a la derecha)"
                if stats["skewness"] > 0
                else "negativa (colas a la izquierda)"
            )
            st.info(f"ℹ️ **Asimetría {skew_dir}**: Puede haber outliers en la dirección indicada.")

    def _render_segment_analysis(self, model_name: str) -> None:
        """
        Renderiza análisis por segmento.

        Args:
            model_name: Nombre del modelo
        """
        st.subheader(f"Análisis por Segmento - {model_name.title()}")

        # Boxplot by segment
        st.markdown("**📦 Distribución de Residuos por Tipo de Tienda**")
        fig_segments = self.analyzer.plot_residuals_by_segment(model_name)
        st.plotly_chart(fig_segments, use_container_width=True)

        # Segment statistics table
        st.markdown("**📊 Métricas por Segmento**")

        stats = self.analyzer.get_error_statistics(model_name)
        segment_df = pd.DataFrame(stats["segment_errors"])

        st.dataframe(
            segment_df.style.format({"rmse": "{:.4f}", "samples": "{:,}"}).highlight_min(
                subset=["rmse"], color="lightgreen"
            ),
            use_container_width=True,
        )

        # Stability analysis
        segment_rmses = segment_df["rmse"].values
        rmse_range = segment_rmses.max() - segment_rmses.min()

        st.markdown("**🎯 Análisis de Estabilidad**")

        if rmse_range < 0.05:
            st.success(
                f"""
            ✅ **Alta estabilidad entre segmentos**
            
            - Rango de RMSE: {rmse_range:.4f}
            - El modelo mantiene consistencia entre tipos de tienda
            - Recomendado para producción general
            """
            )
        elif rmse_range < 0.15:
            st.info(
                f"""
            ℹ️ **Estabilidad moderada**
            
            - Rango de RMSE: {rmse_range:.4f}
            - Variación aceptable entre segmentos
            - Considerar calibración específica para segmentos con mayor error
            """
            )
        else:
            st.warning(
                f"""
            ⚠️ **Alta variabilidad entre segmentos**
            
            - Rango de RMSE: {rmse_range:.4f}
            - Diferencias significativas en performance
            - Recomendación: Entrenar modelos específicos por segmento
            """
            )

    def _render_shap_analysis(self, model_name: str) -> None:
        """
        Renderiza análisis SHAP.

        Args:
            model_name: Nombre del modelo
        """
        st.subheader(f"Explicabilidad (SHAP) - {model_name.title()}")

        available_shap = list(self.analyzer.shap_summary.keys())

        if model_name not in self.analyzer.shap_summary:
            st.warning(
                f"""
            ⚠️ **Valores SHAP no disponibles para el modelo '{model_name}'**
            
            Modelos con SHAP disponibles: {', '.join(available_shap) if available_shap else 'ninguno'}
            
            Haz clic en el botón **🚀 Exportar Datos** arriba para regenerar el análisis SHAP.
            """
            )
            return

        # Slider for number of features
        top_n = st.slider(
            "Número de features a mostrar", 5, 20, 15, key=f"shap_slider_{model_name}"
        )

        # SHAP Summary Plot (Scatter)
        st.markdown("**🎯 SHAP Summary Plot: Distribución de Impactos**")
        st.caption(
            "Cada punto representa una observación. El color indica el valor de la feature: "
            "🔴 rojo = valor alto, 🔵 azul = valor bajo"
        )

        fig_summary = self.analyzer.plot_shap_summary_scatter(model_name, top_n)
        st.plotly_chart(fig_summary, use_container_width=True)

        st.divider()

        # SHAP importance bar chart
        st.markdown("**📊 Importancia Agregada de Features**")
        st.caption("Promedio del valor absoluto de SHAP por feature")

        fig_shap = self.analyzer.plot_shap_importance(model_name, top_n)
        st.plotly_chart(fig_shap, use_container_width=True)

        st.divider()

        # SHAP interpretations
        st.markdown("**🔍 Interpretación de Features Principales**")

        interpretations = self.analyzer.get_shap_interpretation(model_name, top_n=5)

        for i, interp in enumerate(interpretations, 1):
            with st.expander(
                f"**{i}. {interp['feature_name']}** (Impacto: {interp['importance']:.4f})"
            ):
                st.markdown(interp["text"])

                # Feature-specific insights
                if interp["feature"] == "item_cnt_lag_1_log":
                    st.info(
                        """
                    **Insight**: Las ventas del mes anterior son el mejor predictor debido a la **inercia temporal**.
                    Productos con alta demanda histórica tienden a mantener esa demanda.
                    """
                    )
                elif interp["feature"] == "price_rel_category":
                    st.info(
                        """
                    **Insight**: El precio relativo a la categoría captura **elasticidad precio-demanda**.
                    Productos caros relativamente tienen menor demanda.
                    """
                    )
                elif interp["feature"] == "shop_cluster":
                    st.info(
                        """
                    **Insight**: El tipo de tienda (cluster) refleja **volumen base** de ventas.
                    Megatiendas tienen mayor demanda promedio.
                    """
                    )

        # SHAP summary table
        st.markdown("**📋 Tabla Completa de Features**")

        shap_df = self.analyzer.shap_summary[model_name].head(top_n)
        st.dataframe(
            shap_df.style.format(
                {
                    "mean_abs_shap_value": "{:.4f}",
                    "mean_shap_value": "{:.4f}",
                    "std_shap_value": "{:.4f}",
                }
            ),
            use_container_width=True,
        )

    def _render_report_download(self, model_name: str) -> None:
        """
        Renderiza sección de descarga de reporte.

        Args:
            model_name: Nombre del modelo
        """
        st.divider()
        st.subheader("📄 Reporte Técnico Completo")

        report = self.analyzer.generate_technical_report(model_name)

        col1, col2 = st.columns([3, 1])

        with col1:
            # Mostrar reporte directamente (sin expander)
            st.markdown(report)

        with col2:
            st.download_button(
                label="⬇️ Descargar Reporte",
                data=report,
                file_name=f"reporte_tecnico_{model_name}.md",
                mime="text/markdown",
            )
