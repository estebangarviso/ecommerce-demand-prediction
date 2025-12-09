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
           - **Ventanas temporales fijas: 2 rolling windows (por defecto 3 y 6 meses)**
           - Balanceo con SMOTE para regresión (opcional)
           - Features temporales (lags: t-1, t-2, t-3)
           - Validación temporal con TimeSeriesSplit (5 splits)

        4. **Modelado**
           - **Ensemble Learning**: Stacking (Random Forest + XGBoost + meta-estimador)
           - **Deep Learning**: MLP (3 capas densas) + LSTM-DNN simplificado
           - **Total de features por predicción: 10**
             - 6 básicas: cluster, categoría, precio, lag_1, lag_2, lag_3
             - 4 rolling windows: rolling_mean (2 ventanas) + rolling_std (2 ventanas)
           - 5 modelos entrenados en total
           - Explicabilidad: Valores SHAP para interpretar predicciones con todas las features

        5. **Evaluación**
           - **Stacking Ensemble**: R² = 0.999, RMSE = 0.03, MAE = 0.01
           - **Random Forest**: R² = 0.999, RMSE = 0.03, MAE = 0.01
           - **XGBoost**: R² = 0.984, RMSE = 0.15, MAE = 0.09
           - **MLP**: R² = 0.413, RMSE = 0.81, MAE = 0.60
           - **LSTM-DNN**: R² = 0.413, RMSE = 0.81, MAE = 0.60
           - Validación con TimeSeriesSplit (sin data leakage)

        6. **Despliegue**
           - **Backend**: API REST con FastAPI (5 endpoints documentados)
           - **Frontend**: Streamlit con cliente HTTP (httpx)
           - Arquitectura desacoplada y escalable
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

        **Las 10 variables analizadas:**
        1. **Ventas del mes anterior (lag_1)**: La más influyente, refleja inercia de la demanda
        2. **Ventas hace 2 meses (lag_2)**: Segunda más influyente, tendencia reciente
        3. **Ventas hace 3 meses (lag_3)**: Captura tendencia a corto plazo
        4. **Categoría de producto**: Algunas categorías tienen mayor demanda base
        5. **Precio unitario**: Relación inversa (precio ↑, demanda ↓)
        6. **Tipo de tienda (cluster)**: Megatiendas tienen mayor volumen esperado
        7. **Media móvil ventana 1**: Promedio de ventas en primera ventana temporal (ej: 3 meses)
        8. **Desv. estándar ventana 1**: Volatilidad en primera ventana (estabilidad de demanda)
        9. **Media móvil ventana 2**: Promedio de ventas en segunda ventana temporal (ej: 6 meses)
        10. **Desv. estándar ventana 2**: Volatilidad en segunda ventana (tendencia a largo plazo)

        **Nota:** El sistema usa **exactamente 2 ventanas rolling** configurables (por defecto: 3 y 6 meses).

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
        st.subheader("Métricas de Rendimiento de los Modelos")

        st.markdown("**Stacking Ensemble (Modelo en Producción):**")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "RMSE",
                "0.03",
                help="Error Cuadrático Medio: Penaliza más los errores grandes. Valor bajo indica mayor precisión.",
            )

        with col2:
            st.metric(
                "MAE",
                "0.01",
                help="Error Absoluto Medio: Promedio de la diferencia entre predicciones y valores reales.",
            )

        with col3:
            st.metric(
                "R² Score",
                "0.999",
                help="Coeficiente de Determinación: El modelo explica el 99.9% de la variabilidad en las ventas.",
            )

        st.caption("Métricas calculadas con TimeSeriesSplit (5 folds)")

        st.markdown("**Comparativa de Modelos:**")
        st.markdown(
            """
        | Modelo | RMSE | MAE | R² | Tipo |
        |:-------|:-----|:----|:---|:-----|
        | **Stacking Ensemble** | 0.03 | 0.01 | **0.999** | Ensemble |
        | Random Forest | 0.03 | 0.01 | 0.999 | Tree-based |
        | XGBoost | 0.15 | 0.09 | 0.984 | Gradient Boosting |
        | MLP | 0.81 | 0.60 | 0.413 | Neural Network |
        | LSTM-DNN | 0.81 | 0.60 | 0.413 | Neural Network |

        **Conclusión:** Los modelos tree-based superan a Deep Learning en datasets tabulares pequeños.
        """
        )

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

        # Arquitectura del Sistema
        st.subheader("Arquitectura del Sistema")

        st.markdown("**Flujo de Comunicación (Frontend ↔ Backend):**")

        st.code(
            """
┌─────────────────────────┐
│  Usuario - Navegador   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Frontend Streamlit :8501│
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Cliente HTTP - httpx   │
└───────────┬─────────────┘
            │
            ▼ POST /predict, GET /metrics, GET /health
┌─────────────────────────┐
│  Backend FastAPI :8000  │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│     Modelos ML/DL       │
├─────────────────────────┤
│ • stacking_model.pkl    │
│ • mlp_model.keras       │
│ • lstm_model.keras      │
│ • scaler.pkl            │
└─────────────────────────┘
            """,
            language="text",
        )

        st.markdown("**Tecnologías Principales:**")
        st.markdown(
            """
        - **Backend**: FastAPI 0.115.0 + uvicorn 0.32.0 (ASGI Server)
        - **Frontend**: Streamlit 1.52.0 + httpx 0.27.2 (Cliente HTTP)
        - **ML Traditional**: scikit-learn 1.5.1 (RF, Stacking, TimeSeriesSplit)
        - **ML Boosting**: XGBoost 2.1.0
        - **Deep Learning**: TensorFlow 2.17.0 (MLP, LSTM-DNN)
        - **Data Processing**: pandas 2.2.2, numpy 1.26.4, imbalanced-learn 0.12.3 (SMOTE)
        - **Explicabilidad**: SHAP 0.46.0
        - **Visualización**: Plotly 5.23.0
        """
        )

        st.divider()

        # Footer
        st.caption(
            "**Desarrollado por:** [Esteban Garviso](https://github.com/estebangarviso) & [Felipe Ortega](https://github.com/piwinsi)"
        )
        st.caption("**Universidad Andrés Bello** - ACIF104 Aprendizaje de Máquinas - 2025")
