# 🛒 Predicción de Demanda en E-commerce

Sistema de predicción de demanda para retail con **Machine Learning Avanzado** e **Interpretabilidad SHAP**, desarrollado para la asignatura **Aprendizaje de Máquinas (ACIF104)** de la Universidad Andrés Bello.

**Equipo 9:**
* **Esteban Garviso** - [GitHub](https://github.com/estebangarviso)
* **Felipe Ortega** - [GitHub](https://github.com/piwinsi)

---

## ✨ Características Principales

- 🤖 **5 Modelos ML/DL**: Random Forest, XGBoost, MLP, LSTM-DNN, Stacking Ensemble
- 🧠 **Explicabilidad SHAP**: Interpretación visual y textual de predicciones
- 🌐 **Arquitectura Cliente-Servidor**: FastAPI (backend) + Streamlit (frontend)
- 📊 **24+ Features Engineered**: Clustering, rolling windows, elasticidad de precio
- 🔄 **Validación Temporal**: TimeSeriesSplit para prevenir data leakage
- 📈 **Análisis Técnico**: Exportación de métricas, SHAP y residuales

## Estructura del Proyecto

El proyecto sigue una arquitectura modular que desacopla la lógica de negocio (Backend REST API) de la capa de presentación (Frontend Streamlit), facilitando la mantenibilidad y escalabilidad:

```text
acif104_s9_equipo9/
│
├── README.md               # Documentación completa del proyecto
├── Pipfile                 # Gestión de dependencias con Pipenv
├── Pipfile.lock            # Árbol de dependencias exacto (reproducibilidad)
├── requirements.txt        # Dependencias (generado automáticamente)
├── requirements-dev.txt    # Dependencias de desarrollo (generado automáticamente)
├── Makefile                # Comandos de automatización (install, train, api, start)
├── pyproject.toml          # Configuración de QA (Black, Isort, Mypy)
│
├── .githooks/              # Git hooks personalizados
│   └── pre-commit          # Auto-sincronización de requirements.txt al commitear
│
├── data/                   # Datasets con sistema de respaldo automático
│   ├── .gitkeep            # Los datos se descargan automáticamente vía KaggleHub
│   └── [*.csv]             # Respaldo local: sales_train, items, shops, item_categories
│
├── models/                 # Modelos entrenados y metadatos
│   ├── stacking_model.pkl  # Ensemble Stacking (Random Forest + XGBoost)
│   ├── mlp_model.keras     # Red Neuronal MLP (3 capas densas)
│   ├── lstm_model.keras    # Red Neuronal LSTM-DNN simplificada
│   ├── scaler.pkl          # StandardScaler para normalización
│   └── metrics.json        # Métricas comparativas (RMSE, MAE, R²)
│
├── notebooks/              # Prototipado y análisis exploratorio
│   ├── 01_EDA_Clustering.ipynb      # K-Means, Outliers y patrones temporales
│   └── 02_Modelado_Ensemble.ipynb   # Experimentos con Stacking y Deep Learning
│
├── src/                    # Backend: Lógica de Negocio y Modelado
│   ├── __init__.py         # Inicialización del paquete
│   ├── data_processing.py  # Pipeline ETL: SMOTE, Rolling Windows, TimeSeriesSplit
│   ├── train.py            # Entrenamiento de 5 modelos (RF, XGB, MLP, LSTM, Stacking)
│   ├── inference.py        # Motor de inferencia con sistema de respaldo
│   └── api.py              # FastAPI REST API (5 endpoints con Pydantic)
│
├── app/                    # Frontend: Interfaz de Usuario con Streamlit
│   ├── README.md           # Documentación de arquitectura modular
│   ├── app.py              # Punto de entrada principal
│   ├── config.py           # Configuraciones centralizadas
│   ├── state_manager.py    # Gestión de estado (Singleton)
│   │
│   ├── services/           # Lógica de negocio
│   │   ├── data_exporter.py         # Exportación de métricas y SHAP a CSV
│   │   ├── model_analyzer.py        # Análisis de métricas de modelos
│   │   ├── pricing_service.py       # Precios dinámicos por categoría
│   │   ├── prediction_service.py    # Cliente HTTP para API REST
│   │   └── trend_analyzer.py        # Análisis de tendencias
│   │
│   ├── components/         # Componentes de visualización
│   │   ├── chart_builder.py         # Gráficos Plotly reutilizables
│   │   ├── shap_renderer.py         # Renderizado SHAP (dark/light theme)
│   │   └── dataframe_builder.py     # Construcción de DataFrames
│   │
│   ├── ui_components/      # Componentes UI
│   │   ├── header.py       # Encabezado con branding
│   │   └── sidebar.py      # Formulario de predicción
│   │
│   └── views/              # Vistas de navegación
│       ├── technical_analysis_view.py  # Análisis técnico y métricas
│       ├── prediction_view.py          # Vista principal de predicción
│       ├── monitoring_view.py          # Dashboard de monitoreo
│       └── about_view.py               # Información del proyecto
│
└── models/                 # Artefactos serializados (Persistencia)
    ├── lstm_model.keras   # Modelo LSTM-DNN entrenado
    ├── mlp_model.keras    # Modelo MLP entrenado
    ├── stacking_model.pkl  # Modelo final de ensamble (RF + XGBoost)
    ├── scaler.pkl         # StandardScaler serializado
    ├── features.pkl        # Metadatos de columnas
    ├── xgb_simple_shap.pkl # Modelo proxy para explicabilidad
    └── category_prices.pkl # Precios promedio por categoría
```

## Inicio Rápido

```bash
# 1. Clonar repositorio
git clone https://github.com/estebangarviso/acif104_s9_equipo9.git
cd acif104_s9_equipo9

# 2. Instalar dependencias
pipenv install --ignore-pipfile

# 3. Iniciar Backend (Terminal 1)
pipenv run api

# 4. Iniciar Frontend (Terminal 2)
pipenv run start
```

📖 **Documentación completa:** Ver [docs/INSTALLATION.md](docs/INSTALLATION.md)

## Características Principales

- **5 Modelos ML/DL:** Random Forest, XGBoost, MLP, LSTM-DNN, Stacking Ensemble
- **Ingeniería de Features Avanzada (24+ variables):**
  - **Momentum:** Deltas (delta_1_2, evolution_3m), promedios y dirección de tendencia
  - **Sensibilidad al Precio:** Cambios porcentuales, elasticidad precio-demanda, ingreso potencial
  - **Desviaciones:** Z-scores, diferencias vs promedio, coeficientes de volatilidad
  - **Rolling Windows:** 2 ventanas temporales configurables (default: 3 y 6 meses)
  - **Clustering K-Means:** Segmentación automática de tiendas
  - **Balanceo SMOTE:** Opcional para regresión con clases desbalanceadas
- **API REST con FastAPI:** 5 endpoints documentados con Swagger UI interactivo
- **Frontend Streamlit:** Interfaz moderna con explicabilidad SHAP waterfall + interpretación textual
- **Validación Temporal:** TimeSeriesSplit (5 folds) para prevenir data leakage
- **Restricciones Monotónicas:** En XGBoost para coherencia económica (precio ↑ → demanda ↓)
- **Sistema de Respaldo:** Gestión automática de datasets con KaggleHub

📖 **Documentación Técnica Completa:** Ejecuta la aplicación y ve a la pestaña "Acerca de"  
📖 **Detalles de Implementación:** Ver [docs/TECHNICAL_DETAILS.md](docs/TECHNICAL_DETAILS.md)  
📖 **API Endpoints:** Ver [docs/API.md](docs/API.md) o http://localhost:8000/docs

## Capturas de Pantalla

### Vista de Predicción con SHAP Waterfall + Interpretación Textual
![Vista de Predicción](docs/screenshots/prediction-view.png)

*La vista muestra KPIs principales (demanda predicha, ventas esperadas, tendencia), gráfico SHAP waterfall con contribución de features, e interpretación automática en lenguaje natural.*

### Panel de Monitoreo con Métricas Dinámicas
![Panel de Monitoreo](docs/screenshots/monitoring-view.png)

*Dashboard de salud del sistema mostrando métricas de todos los modelos, comparativas de rendimiento y estado del servicio backend.*

**📖 Ver documentación técnica completa en la pestaña "Acerca de" dentro de la aplicación Streamlit.**

## Tecnologías Utilizadas

**Machine Learning:** scikit-learn, XGBoost, TensorFlow, imbalanced-learn, SHAP  
**Backend:** FastAPI, Pydantic, uvicorn  
**Frontend:** Streamlit, Plotly, httpx  
**Data:** pandas, numpy, KaggleHub  
**QA:** Black, Pylint, Mypy, Isort, pytest

📖 **Ver versiones completas:** [docs/INSTALLATION.md](docs/INSTALLATION.md)

## Métricas de los Modelos

Para ver las **métricas actualizadas** de todos los modelos entrenados (RMSE, MAE, R²), consulta la **sección "Dashboard Técnico"** dentro de la aplicación Streamlit en la pestaña **"Acerca de"**.

Las métricas se cargan dinámicamente desde `models/metrics.json` y reflejan el rendimiento real validado con **TimeSeriesSplit** (5 folds).

**Modelos Evaluados:**
- Random Forest (Tree-based)
- XGBoost (Gradient Boosting con restricciones monotónicas)
- MLP (Red Neuronal Densa)
- LSTM-DNN (Red Neuronal Recurrente)
- Stacking Ensemble (Random Forest + XGBoost + Meta-estimador)

**Nota:** Los modelos basados en árboles (Random Forest, XGBoost) generalmente muestran mejor rendimiento en datasets tabulares de tamaño moderado. Consulta la documentación técnica en la app para análisis detallado.

## Arquitectura del Sistema

El sistema implementa el patrón **Cliente-Servidor** con separación clara de responsabilidades:

**Backend (FastAPI):**
- Servidor ASGI con uvicorn
- 5 endpoints REST: `/predict`, `/health`, `/metrics`, `/schema`, `/retrain`
- Validación de datos con Pydantic
- Carga de modelos serializados (.pkl, .keras)
- Feature engineering centralizado

**Frontend (Streamlit):**
- Cliente HTTP con httpx
- UI interactiva con 3 vistas principales
- Visualización SHAP con waterfall plots
- Interpretación en lenguaje natural de predicciones
- KPIs y gráficos temporales con Plotly

**Comunicación:**
```
Usuario → Streamlit UI → HTTP Request (JSON) → FastAPI → Modelos ML/DL
         ← Streamlit UI ← HTTP Response (JSON) ← FastAPI ← Predicción + SHAP
```

Ver **diagrama Mermaid completo** en la pestaña "Acerca de" dentro de la aplicación.

## Documentación Adicional

- 📘 [Guía de Instalación](docs/INSTALLATION.md) - Configuración completa del entorno
- 🔧 [Detalles Técnicos](docs/TECHNICAL_DETAILS.md) - Metodología, arquitectura y features
- 🌐 [Documentación API](docs/API.md) - Endpoints y ejemplos de uso
- 🏗️ [Arquitectura Frontend](app/README.md) - Patrones SOLID y estructura modular
- 🐳 [Deployment con Docker](docs/DOCKER.md) - Guía de Deployment con Docker

## Universidad Andrés Bello - 2025

**Asignatura:** ACIF104 - Aprendizaje de Máquinas  
**Docente:** OMAR IVÁN SALINAS SILVA  
**Periodo:** Sexto Trimestre 2025
