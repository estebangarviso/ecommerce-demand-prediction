"""Vista de documentación técnica del sistema - Dashboard Ejecutivo.

Este módulo proporciona una interfaz interactiva que documenta la arquitectura,
rendimiento y capacidades del sistema de predicción de demanda. Toda la información
se carga dinámicamente desde los modelos y configuraciones del sistema.
"""

import streamlit as st
from streamlit_mermaid import st_mermaid
import pandas as pd
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from services.prediction_service import PredictionService


class AboutView:
    """Dashboard de documentación técnica para evaluadores y stakeholders.

    Esta vista carga dinámicamente las métricas desde la API REST
    y presenta la información de forma interactiva sin valores hardcodeados.
    """

    def __init__(self, prediction_service: "PredictionService"):
        """Inicializa la vista con el servicio de predicción para acceder a la API.

        Args:
            prediction_service: Servicio que contiene la URL de la API y métodos helper
        """
        self.prediction_service = prediction_service

    def _load_metrics_from_api(self) -> Optional[List[Dict]]:
        """Carga las métricas de rendimiento desde la API REST.

        Returns:
            Lista de diccionarios con métricas por modelo, o None si hay error
        """
        try:
            data = self.prediction_service.get_metrics()
            if data:
                return data.get("models", [])
            return None
        except Exception as e:
            st.warning(
                f":material/error: No se pudieron cargar métricas desde la API: {str(e)}",
                icon=":material/warning:",
            )
            return None

    def render(self) -> None:
        """Renderiza el dashboard completo de documentación técnica."""
        st.header(":material/analytics: Dashboard Técnico del Sistema")

        st.info(
            "Esta documentación presenta la arquitectura, rendimiento y capacidades "
            "del sistema. Todos los datos se cargan dinámicamente desde las configuraciones actuales.",
            icon=":material/info:",
        )

        # Resumen Ejecutivo
        st.subheader(":material/summarize: Resumen Ejecutivo")

        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(
                """
            **Problema de Negocio:**
            
            Predecir la demanda futura de productos en comercio electrónico para optimizar 
            la gestión de inventario, reducir costos de almacenamiento y evitar pérdidas 
            por desabastecimiento.
            
            **Solución Implementada:**
            
            Sistema de Machine Learning con arquitectura desacoplada Cliente-Servidor que
            separa la lógica de inferencia (Backend REST API) de la interfaz de usuario
            (Frontend Streamlit).
            """
            )

        with col2:
            # Cargar métricas para mostrar el mejor modelo
            metrics_data = self._load_metrics_from_api()
            if metrics_data:
                best_model = max(metrics_data, key=lambda x: x["r2"])
                st.metric(
                    ":material/emoji_events: Mejor Modelo",
                    best_model["model"],
                    help="Modelo con mayor R² en validación",
                )
                st.metric(
                    ":material/speed: R² Score",
                    f"{best_model['r2']:.3f}",
                    help="Coeficiente de determinación",
                )

        st.divider()

        # Arquitectura del Sistema
        st.subheader(":material/account_tree: Arquitectura Cliente-Servidor")

        st.markdown(
            """
        El sistema implementa el patrón arquitectónico **Cliente-Servidor** con separación 
        clara de responsabilidades entre la capa de presentación y la lógica de negocio.
        """
        )

        # Diagrama de arquitectura con Mermaid
        try:
            theme = (
                st.context.theme.type
                if hasattr(st, "context") and hasattr(st.context, "theme")
                else "light"
            )
        except:
            theme = "light"

        st_mermaid(
            f"""
%%{{init: {{'theme':'{theme}'}}}}%%
sequenceDiagram
    participant U as 👤 Usuario
    participant F as 🖥️ Frontend<br/>(Streamlit :8501)
    participant B as ☁️ Backend<br/>(FastAPI :8000)
    participant M as 🧠 Modelos<br/>(.pkl/.keras)
    
    U->>F: Configura parámetros<br/>(categoría, precio, lags)
    F->>F: Valida inputs
    F->>B: POST /predict<br/>JSON {{features}}
    B->>B: Validación Pydantic
    B->>B: Feature Engineering<br/>(24+ features)
    B->>M: Carga modelos
    M->>M: Inferencia ML/DL
    M-->>B: Predicción
    B->>M: SHAP TreeExplainer
    M-->>B: Valores SHAP
    B-->>F: JSON Response<br/>{{prediction, shap_values}}
    F->>F: Renderiza KPIs
    F->>F: Waterfall Plot SHAP
    F->>F: Interpretación textual
    F-->>U: Visualización completa<br/>(KPIs + Gráficos + SHAP)
    
    Note over U,M: Arquitectura Cliente-Servidor Desacoplada
""",
            pan=False,
            show_controls=False,
            zoom=False,
        )

        st.markdown(
            """
        **Características de la Arquitectura:**
        - 🔄 **Comunicación HTTP:** Frontend stateless que consume REST API
        - ⚡ **Procesamiento Backend:** Feature engineering centralizado en FastAPI
        - 🧠 **Modelos en Servidor:** No se cargan modelos en el cliente
        - 📊 **Explicabilidad:** SHAP calculado en backend y renderizado en frontend
        - 🎯 **Separación de Responsabilidades:** UI, Lógica de Negocio y ML/DL desacoplados
        """
        )

        st.success(
            """
        **Ventajas de esta Arquitectura:**
        
        - :material/trending_up: **Escalabilidad:** Backend y Frontend escalan independientemente
        - :material/build: **Mantenibilidad:** Lógica de negocio centralizada en la API
        - :material/security: **Seguridad:** Modelos no expuestos al cliente
        - :material/apps: **Deployment:** Contenedores Docker separados
        - :material/bug_report: **Testing:** Unit tests aislados por capa
        """,
            icon=":material/check_circle:",
        )

        st.divider()

        # Rendimiento de Modelos (Dinámico)
        st.subheader(":material/leaderboard: Evaluación de Rendimiento")

        metrics_data = self._load_metrics_from_api()

        if metrics_data:
            # Convertir a DataFrame
            df_metrics = pd.DataFrame(metrics_data)

            # Ordenar por R² descendente
            df_metrics = df_metrics.sort_values("r2", ascending=False)

            # Formatear valores numéricos
            df_metrics_display = df_metrics.copy()
            df_metrics_display["rmse"] = df_metrics_display["rmse"].apply(lambda x: f"{x:.4f}")
            df_metrics_display["mae"] = df_metrics_display["mae"].apply(lambda x: f"{x:.4f}")
            df_metrics_display["r2"] = df_metrics_display["r2"].apply(lambda x: f"{x:.4f}")

            # Renombrar columnas para mejor presentación
            df_metrics_display = df_metrics_display.rename(
                columns={"model": "Modelo", "rmse": "RMSE", "mae": "MAE", "r2": "R²"}
            )

            st.markdown("**Comparativa de Modelos (Validación Temporal con TimeSeriesSplit):**")

            # Configurar columnas con ayuda contextual
            st.dataframe(
                df_metrics_display,
                width="stretch",
                hide_index=True,
                column_config={
                    "Modelo": st.column_config.TextColumn(
                        "Modelo",
                        width="medium",
                        help="Algoritmo de Machine Learning o Deep Learning",
                    ),
                    "RMSE": st.column_config.TextColumn(
                        "RMSE",
                        help="Root Mean Squared Error - Menor es mejor",
                    ),
                    "MAE": st.column_config.TextColumn(
                        "MAE", help="Mean Absolute Error - Menor es mejor"
                    ),
                    "R²": st.column_config.TextColumn(
                        "R²",
                        help="Coeficiente de Determinación - Más cercano a 1.0 es mejor",
                    ),
                },
            )

            # Encontrar el mejor modelo dinámicamente
            best_model = df_metrics.loc[df_metrics["r2"].idxmax()]

            st.caption(
                f":material/info: **Nota Metodológica:** El modelo **{best_model['model']}** "
                f"muestra el mejor rendimiento estadístico (R² = {best_model['r2']:.4f}). "
                "Los modelos basados en árboles (Random Forest, XGBoost) generalmente superan "
                "a las redes neuronales en datasets tabulares de tamaño moderado. "
                "La validación se realizó con TimeSeriesSplit (5 folds) para prevenir data leakage temporal."
            )

        else:
            st.error(
                ":material/error: No se encontró el archivo `models/metrics.json`. "
                "Ejecuta el entrenamiento con `pipenv run train` para generar las métricas.",
                icon=":material/warning:",
            )

        st.divider()

        # Ingeniería de Características
        st.subheader(":material/construction: Ingeniería de Características")

        st.markdown(
            """
        El modelo procesa múltiples categorías de features para capturar patrones complejos
        en la demanda. Las features se calculan dinámicamente en cada predicción.
        """
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                """
            **:material/dataset: Features Base**
            - Cluster de tienda (K-Means)
            - ID de categoría
            - Precio del producto
            - Ventas históricas (lags)
            
            **:material/timeline: Rolling Windows**
            - Media móvil (configurable)
            - Desviación estándar
            - Captura tendencias temporales
            """
            )

        with col2:
            st.markdown(
                """
            **:material/trending_up: Momentum**
            - Deltas entre períodos
            - Evolución de tendencia
            - Dirección de crecimiento
            - Promedio de momentum
            
            **:material/attach_money: Pricing**
            - Cambio porcentual de precio
            - Elasticidad precio-demanda
            - Precio relativo a categoría
            - Detección de descuentos
            """
            )

        with col3:
            st.markdown(
                """
            **:material/functions: Desviaciones**
            - Diferencia vs promedio
            - Normalización Z-score
            - Coeficiente de volatilidad
            
            **:material/transform: Normalizaciones**
            - Transformaciones logarítmicas
            - Escalado estándar
            - Mejora estabilidad numérica
            """
            )

        st.info(
            """
        **:material/rule: Restricciones Monotónicas en XGBoost:**
        
        Se aplican restricciones de monotonicidad para asegurar coherencia económica:
        - Precio :material/arrow_upward: → Demanda :material/arrow_downward: (restricción negativa)
        - Ventas previas :material/arrow_upward: → Demanda :material/arrow_upward: (restricción positiva)
        
        Esto previene predicciones contradictorias con las leyes de oferta y demanda.
        """,
            icon=":material/verified:",
        )

        st.divider()

        # Metodología CRISP-DM
        st.subheader(":material/science: Metodología CRISP-DM")

        tab1, tab2, tab3 = st.tabs(
            [
                ":material/business: Fases 1-3",
                ":material/model_training: Fases 4-5",
                ":material/rocket_launch: Fase 6",
            ]
        )

        with tab1:
            st.markdown(
                """
            **1. Business Understanding**
            - Objetivo: Predecir demanda mensual para optimización de inventario
            - Métrica de éxito: Maximizar R² y minimizar RMSE/MAE
            
            **2. Data Understanding**
            - Dataset: Registros históricos de ventas (2013-2015)
            - Fuente: Kaggle - "Predict Future Sales"
            - Variables: Tiendas, categorías, precios, ventas mensuales
            
            **3. Data Preparation**
            - Limpieza: Outliers detectados con Z-score
            - Clustering: K-Means para segmentación de tiendas
            - Balanceo: SMOTE opcional para clases minoritarias
            - Validación: TimeSeriesSplit para series temporales
            """
            )

        with tab2:
            st.markdown(
                """
            **4. Modeling**
            - Stacking Ensemble: Random Forest + XGBoost + meta-estimador
            - Deep Learning: MLP (capas densas) + LSTM-DNN
            - Hiperparámetros: Optimización con grid search
            - Restricciones: Monotonicidad en features económicas
            
            **5. Evaluation**
            - Métricas: RMSE, MAE, R² (coeficiente de determinación)
            - Validación: TimeSeriesSplit (prevenir data leakage)
            - Explicabilidad: SHAP TreeExplainer
            - Comparación: Baseline vs modelos avanzados
            """
            )

        with tab3:
            st.markdown(
                """
            **6. Deployment**
            
            Arquitectura de producción con microservicios:
            
            - **Backend:** FastAPI + uvicorn (servidor ASGI)
            - **Frontend:** Streamlit + httpx (cliente REST)
            - **Modelos:** Serializados con joblib y keras
            - **Infraestructura:** Docker Compose (multi-container)
            - **Monitoreo:** Health checks + logs estructurados
            
            **Endpoints Implementados:**
            
            | Endpoint | Método | Descripción |
            |:---------|:-------|:------------|
            | `/health` | GET | Health check del sistema |
            | `/predict` | POST | Predicción de demanda |
            | `/metrics` | GET | Métricas de modelos |
            | `/schema` | GET | Schema dinámico |
            | `/retrain` | POST | Reentrenamiento automático |
            """
            )

        st.divider()

        # Explicabilidad con SHAP
        st.subheader(":material/psychology: Explicabilidad con SHAP")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(
                """
            **SHAP (SHapley Additive exPlanations)** permite interpretar las predicciones
            mediante valores de Shapley basados en teoría de juegos coalicionales:
            
            - **TreeExplainer:** Método optimizado para modelos basados en árboles
            - **Waterfall Plots:** Visualización de contribución marginal por feature
            - **Interpretación Automática:** Traducción a lenguaje natural de insights
            - **Base Teórica:** Valores de Shapley garantizan propiedades deseables
            
            **Interpretación de Gráficos SHAP:**
            
            - Barras azules :material/arrow_forward: Incrementan la predicción
            - Barras rojas :material/arrow_back: Disminuyen la predicción
            - E[f(X)]: Valor base del modelo (promedio poblacional)
            - f(x): Predicción final para la instancia específica
            """
            )

        with col2:
            st.success(
                """
            **Ventajas de SHAP:**
            
            - :material/visibility: Transparencia algorítmica
            - :material/gavel: Cumplimiento regulatorio
            - :material/bug_report: Debugging de modelos
            - :material/thumb_up: Confianza del usuario
            - :material/lightbulb: Insights accionables
            """,
                icon=":material/verified_user:",
            )

        st.divider()

        # Validación Temporal
        st.subheader(":material/schedule: Validación Temporal")

        st.markdown(
            """
        Se utiliza **TimeSeriesSplit** para validación en series temporales, 
        respetando el orden cronológico de los datos:
        
        ```
        Fold 1: Train [Período 1-24] → Test [Período 25-30]
        Fold 2: Train [Período 1-25] → Test [Período 26-31]
        Fold 3: Train [Período 1-26] → Test [Período 27-32]
        Fold 4: Train [Período 1-27] → Test [Período 28-33]
        Fold 5: Train [Período 1-28] → Test [Período 29-34]
        ```
        
        **Ventajas:**
        
        - :material/check: Previene data leakage (modelo nunca ve datos futuros)
        - :material/check: Simula comportamiento en producción
        - :material/check: Métricas realistas sin sobreajuste
        - :material/check: Respeta dependencia temporal de los datos
        """
        )

        st.divider()

        # Stack Tecnológico
        st.subheader(":material/terminal: Stack Tecnológico")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
            **:material/cloud: Backend (Servidor API)**
            - FastAPI: Framework web moderno
            - Pydantic: Validación de datos
            - uvicorn: Servidor ASGI
            - httpx: Cliente HTTP asíncrono
            
            **:material/psychology: Machine Learning**
            - scikit-learn: Modelos tradicionales
            - XGBoost: Gradient boosting
            - imbalanced-learn: SMOTE
            - SHAP: Explicabilidad
            
            **:material/memory: Deep Learning**
            - TensorFlow: Framework principal
            - Keras: API de alto nivel
            """
            )

        with col2:
            st.markdown(
                """
            **:material/web: Frontend (Cliente Web)**
            - Streamlit: UI interactiva
            - Plotly: Gráficos interactivos
            - matplotlib: Visualizaciones
            
            **:material/storage: Data Processing**
            - pandas: Manipulación de datos
            - numpy: Cálculos numéricos
            - joblib: Serialización
            
            **:material/build_circle: DevOps**
            - Docker: Contenedores
            - pytest: Testing
            - Black/Pylint: Code quality
            """
            )

        st.divider()

        # Limitaciones
        st.subheader(":material/warning: Limitaciones y Consideraciones")

        st.warning(
            """
        **Limitaciones del Sistema:**
        
        1. **Horizonte Temporal:** Predicción limitada a 1 mes adelante
        2. **Datos Históricos:** Entrenado con datos de períodos específicos
        3. **Factores Externos:** No considera eventos excepcionales o promociones
        4. **Incertidumbre:** Las métricas indican el margen de error esperado
        
        **Recomendaciones:**
        
        - :material/tips_and_updates: Usar predicciones como guía complementaria
        - :material/groups: Combinar con conocimiento experto del negocio
        - :material/update: Reentrenar periódicamente con datos recientes
        - :material/monitoring: Monitorear drift de datos en producción
        - :material/tune: Ajustar thresholds según retroalimentación real
        """,
            icon=":material/info:",
        )

        st.divider()

        # Guía de Uso
        with st.expander(":material/help: Guía de Uso del Sistema", expanded=False):
            st.markdown(
                """
            **Inicialización del Sistema:**
            
            El sistema requiere dos terminales simultáneas para operar:
            
            **Terminal 1 - Backend API:**
            ```bash
            pipenv run api
            # Backend disponible en http://localhost:8000
            ```
            
            **Terminal 2 - Frontend UI:**
            ```bash
            pipenv run start
            # Frontend disponible en http://localhost:8501
            ```
            
            **Flujo de Predicción:**
            
            1. Configurar parámetros en el sidebar
            2. Seleccionar categoría y tipo de tienda
            3. Ajustar precio y ventas históricas
            4. Presionar "Calcular Demanda"
            5. Interpretar resultados (KPIs, SHAP, gráficos)
            
            **Documentación Adicional:**
            
            - :material/api: [API Docs](http://localhost:8000/docs) - Swagger UI interactivo
            - :material/code: [GitHub](https://github.com/estebangarviso/acif104_s9_equipo9) - Código fuente
            """
            )

        st.divider()

        # Footer
        st.caption("---")

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.caption(
                """
            **Desarrollado por:**  
            [Esteban Garviso](https://github.com/estebangarviso) & 
            [Felipe Ortega](https://github.com/piwinsi)
            """
            )

        with col2:
            st.caption(
                """
            **Universidad Andrés Bello**  
            ACIF104 - 2025
            """
            )

        with col3:
            st.caption(
                """
            **Versión:** 1.0.0  
            **Fecha:** Enero 2025
            """
            )
