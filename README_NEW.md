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
- 📈 **Análisis Técnico**: Exportación automática de métricas, SHAP y residuales

---

## 🚀 Inicio Rápido

```bash
# 1. Clonar repositorio
git clone https://github.com/estebangarviso/acif104_s9_equipo9.git
cd acif104_s9_equipo9

# 2. Instalar dependencias
pipenv install --ignore-pipfile

# 3. Entrenar modelos (opcional si ya existen en models/)
pipenv run train

# 4. Iniciar Backend (Terminal 1)
pipenv run api

# 5. Iniciar Frontend (Terminal 2)
pipenv run start
```

**🌐 Acceso a la aplicación:**
- **Frontend:** http://localhost:8501
- **API Docs:** http://localhost:8000/docs

📖 **Instalación detallada:** [docs/INSTALLATION.md](docs/INSTALLATION.md)

---

## 📁 Estructura del Proyecto

```text
├── app/                    # Frontend Streamlit (componentes, vistas, servicios)
├── src/                    # Backend FastAPI (API, entrenamiento, inferencia)
├── models/                 # Modelos entrenados (.pkl, .keras, metrics.json)
├── notebooks/              # Análisis exploratorio (EDA, clustering, SHAP)
├── data/                   # Datasets (descarga automática vía KaggleHub)
├── exports/                # Análisis técnico (métricas, predicciones, SHAP)
└── docs/                   # Documentación técnica completa
```

📖 **Arquitectura detallada:** [docs/TECHNICAL_DETAILS.md](docs/TECHNICAL_DETAILS.md)

---

## 📊 Análisis Técnico

El sistema genera **archivos CSV** con métricas detalladas para análisis profundo:

### Generar Exports

```bash
# Ejecutar notebook de análisis SHAP
pipenv run jupyter nbconvert --to notebook --execute notebooks/03_SHAP_Analysis.ipynb

# Los archivos se generan automáticamente en exports/
```

### Archivos Generados (`exports/`)

| Archivo | Contenido |
|:--------|:----------|
| `metrics_overall.csv` | RMSE, MAE, R², train_time, model_size |
| `predictions_<model>_val.csv` | y_true, y_pred, residuales, segmentos |
| `shap_summary_<model>_val.csv` | Importancia SHAP, ranking de features |
| `features_val.csv` | 24+ features procesadas (reproducibilidad) |
| `segments_map.csv` | Mapeo de clusters y categorías |

### Casos de Uso

**1. Análisis de Error por Segmento:**
```python
import pandas as pd

preds = pd.read_csv('exports/predictions_randomforest_val.csv')
segments = pd.read_csv('exports/segments_map.csv')

# Error por tipo de tienda
error_by_cluster = preds.merge(segments, on='shop_cluster') \
    .groupby('cluster_name')['residual'].agg(['mean', 'std'])
```

**2. Top Features SHAP:**
```python
shap = pd.read_csv('exports/shap_summary_randomforest_val.csv')
print(shap.head(5)[['feature', 'mean_abs_shap_value', 'rank']])
```

📖 **Análisis completo:** [docs/TECHNICAL_DETAILS.md - Sección 8](docs/TECHNICAL_DETAILS.md#8-análisis-técnico-y-exportación-de-métricas)

---

## 🖼️ Capturas de Pantalla

### Vista de Predicción con Explicabilidad SHAP
![Vista de Predicción](docs/screenshots/prediction-view.png)

*KPIs principales, gráfico SHAP waterfall con contribución de features e interpretación automática en lenguaje natural.*

### Panel de Monitoreo
![Panel de Monitoreo](docs/screenshots/monitoring-view.png)

*Dashboard de métricas comparativas, salud del sistema y estado del backend.*

---

## 📚 Documentación

| Documento | Descripción |
|:----------|:------------|
| [INSTALLATION.md](docs/INSTALLATION.md) | Guía de instalación y configuración completa |
| [TECHNICAL_DETAILS.md](docs/TECHNICAL_DETAILS.md) | Arquitectura, features engineering y metodología |
| [API.md](docs/API.md) | Endpoints REST y ejemplos de uso |
| [DOCKER.md](docs/DOCKER.md) | Deployment con contenedores Docker |
| [app/README.md](app/README.md) | Arquitectura frontend (patrones SOLID) |

---

## 🛠️ Tecnologías

**Machine Learning:** scikit-learn, XGBoost, TensorFlow, SHAP, imbalanced-learn  
**Backend:** FastAPI, Pydantic, uvicorn  
**Frontend:** Streamlit, Plotly, httpx  
**Data:** pandas, numpy, KaggleHub  
**DevOps:** Pipenv, Docker, pytest

---

## 🎓 Universidad Andrés Bello - 2025

**Asignatura:** ACIF104 - Aprendizaje de Máquinas  
**Docente:** OMAR IVÁN SALINAS SILVA  
**Periodo:** Sexto Trimestre 2025

---

## 📄 Licencia

Este proyecto es parte del trabajo académico para la Universidad Andrés Bello.
