"""Vista de información del sistema."""

import streamlit as st


class AboutView:
    """Vista de información del sistema."""

    @staticmethod
    def render() -> None:
        """Renderiza la vista de información."""
        st.header("Acerca del Sistema")

        # Metodología
        st.subheader("Metodología CRISP-DM")
        st.markdown(
            """
        Este sistema sigue la metodología **CRISP-DM** (Cross-Industry Standard Process for Data Mining), 
        un estándar internacional para proyectos de ciencia de datos:

        1. **Comprensión del Negocio**
           - Objetivo: Predecir la demanda futura en retail para optimizar inventario y ventas
           - Beneficio: Reducir costos de almacenamiento y evitar pérdida de ventas por falta de stock

        2. **Comprensión de Datos**
           - Dataset: Registros históricos de ventas de retail (2013-2015)
           - Fuente: Kaggle - "Predict Future Sales"
           - Variables: Tiendas, categorías de productos, precios, ventas mensuales

        3. **Preparación de Datos**
           - Limpieza de outliers y valores extremos
           - Segmentación de tiendas mediante clustering K-Means (k=2)
           - Generación de features temporales (lags: t-1, t-2, t-3)
           - Transformación logarítmica para normalizar distribución

        4. **Modelado**
           - Técnica: Ensemble Learning con Stacking
           - Modelos base: XGBoost + Random Forest
           - Meta-modelo: Regresión Lineal
           - Explicabilidad: Valores SHAP para interpretar predicciones

        5. **Evaluación**
           - RMSE: 1.005 (Error cuadrático medio)
           - MAE: 0.835 (Error absoluto medio)
           - R²: 0.741 (74.1% de la varianza explicada)

        6. **Despliegue**
           - Aplicación web interactiva con Streamlit
           - Predicciones en tiempo real
           - Explicabilidad visual con gráficos SHAP
        """
        )

        st.divider()

        # Guía de uso
        st.subheader("Guía de Uso")
        st.markdown(
            """
        **Pasos para realizar una predicción:**

        1. **Seleccionar Categoría de Producto**
           - En el panel lateral, elige la categoría del producto que deseas analizar
           - El sistema ajustará automáticamente el rango de precio sugerido

        2. **Configurar Tipo de Tienda**
           - Selecciona el tipo de tienda según su volumen:
             - **Pequeña/Kiosco**: Bajo volumen de ventas
             - **Supermercado/Mall**: Volumen medio
             - **Megatienda/Online**: Alto volumen

        3. **Ajustar Precio Unitario**
           - El sistema sugiere un precio promedio histórico
           - Puedes ajustarlo según tu escenario de negocio (±200%)

        4. **Ingresar Historial de Ventas**
           - Ingresa las ventas de los últimos 3 meses:
             - Hace 3 meses (t-3)
             - Hace 2 meses (t-2)
             - Mes anterior (t-1)

        5. **Generar Predicción**
           - Haz clic en el botón "Predecir Demanda"
           - El sistema calculará la demanda esperada para el próximo mes

        **Resultados que obtendrás:**

        - **Demanda Predicha**: Unidades esperadas para el próximo mes
        - **Ventas Esperadas**: Ingreso estimado (demanda × precio)
        - **Tendencia**: Indicador de crecimiento o decrecimiento
        - **Factores de Influencia (SHAP)**: Qué variables impactan más en la predicción
        - **Proyección Temporal**: Visualización de la tendencia histórica y futura
        """
        )

        st.divider()

        # Interpretación
        st.subheader("Interpretación de Resultados")
        st.markdown(
            """
        **KPIs Principales:**

        - **Demanda Predicha**: Número de unidades que se espera vender
          - Úsalo para planificar compras de inventario
          - Considera un margen de seguridad (±10-15%)

        - **Ventas Esperadas**: Ingreso proyectado en pesos
          - Multiplica demanda × precio unitario
          - Útil para proyecciones financieras

        - **Tendencia**: Comparación con el mes anterior
          - Verde ↑: Crecimiento esperado
          - Rojo ↓: Decrecimiento esperado
          - Considera factores estacionales

        **Factores de Influencia (SHAP):**

        El gráfico SHAP muestra qué variables tienen más impacto en la predicción:

        - **Barras rojas**: Incrementan la demanda predicha
        - **Barras azules**: Reducen la demanda predicha
        - **Longitud de barra**: Magnitud del impacto

        Variables típicas con alto impacto:
        - **Ventas del mes anterior (t-1)**: La más influyente, refleja inercia de la demanda
        - **Categoría de producto**: Algunas categorías tienen mayor demanda base
        - **Precio**: Relación inversa (precio ↑, demanda ↓)
        - **Tipo de tienda**: Megatiendas tienen mayor volumen

        **Proyección Temporal:**

        El gráfico muestra:
        - **Barras grises**: Ventas históricas (últimos 3 meses)
        - **Barra verde/roja**: Predicción para el próximo mes
        - **Línea**: Tendencia general

        Úsalo para:
        - Identificar patrones estacionales
        - Validar si la predicción es coherente con el historial
        - Detectar anomalías o cambios bruscos
        """
        )

        st.divider()

        # Métricas del modelo
        st.subheader("Métricas de Rendimiento del Modelo")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "RMSE",
                "1.005",
                help="Error Cuadrático Medio: Penaliza más los errores grandes. Valor bajo indica mayor precisión.",
            )

        with col2:
            st.metric(
                "MAE",
                "0.835",
                help="Error Absoluto Medio: Promedio de la diferencia entre predicciones y valores reales.",
            )

        with col3:
            st.metric(
                "R² Score",
                "0.741",
                help="Coeficiente de Determinación: El modelo explica el 74.1% de la variabilidad en las ventas.",
            )

        st.caption("Métricas calculadas sobre el conjunto de validación (Mes 33)")

        st.divider()

        # Limitaciones
        st.subheader("Limitaciones y Consideraciones")
        st.markdown(
            """
        **Limitaciones del modelo:**

        1. **Datos históricos**: El modelo fue entrenado con datos de 2013-2015
           - Los patrones de consumo pueden haber cambiado
           - Considera factores externos actuales (economía, tendencias)

        2. **Alcance temporal**: Predicción a 1 mes
           - No proyecta más allá del próximo mes
           - Para planificación de largo plazo, consulta con expertos

        3. **Factores no considerados**:
           - Promociones especiales o descuentos
           - Estacionalidad específica (Navidad, Black Friday)
           - Eventos externos (crisis económica, pandemias)
           - Competencia local

        4. **Rango de confianza**: Las predicciones tienen incertidumbre
           - RMSE de 1.005 indica margen de error
           - Considera un rango de ±10-15% en la demanda

        **Recomendaciones:**

        - Usa las predicciones como guía, no como verdad absoluta
        - Combina con conocimiento experto del negocio
        - Actualiza el modelo periódicamente con datos recientes
        - Valida las predicciones contra ventas reales
        - Ajusta estrategias según retroalimentación del mercado
        """
        )

        st.divider()

        # Footer
        st.caption(
            "**Desarrollado por:** [Esteban Garviso](https://github.com/estebangarviso) & [Felipe Ortega](https://github.com/piwinsi)"
        )
