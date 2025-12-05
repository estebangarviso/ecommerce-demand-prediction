import joblib
import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, Any, Optional

# --- Configuración de Directorios ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")


def get_data_path() -> str:
    """
    Obtiene la ruta de los datos con sistema de respaldo.
    Prioriza data/ local, luego intenta KaggleHub.
    """
    # Verificar si existe item_categories.csv en data/
    local_cats = os.path.join(DATA_DIR, "item_categories.csv")
    if os.path.exists(local_cats) and os.path.getsize(local_cats) > 0:
        return DATA_DIR

    # Si no existe localmente, intentar con KaggleHub
    try:
        import kagglehub

        path = kagglehub.dataset_download("jaklinmalkoc/predict-future-sales-retail-dataset-en")
        return path
    except Exception as e:
        print(f"Warning: No se pudo acceder a KaggleHub ({e}). Usando data/ local.")
        return DATA_DIR


def get_unique_categories() -> Dict[int, str]:
    """Carga dinámicamente las categorías desde el CSV."""
    try:
        path = get_data_path()
        cats = pd.read_csv(os.path.join(path, "item_categories.csv"))
        return dict(zip(cats.item_category_id, cats.item_category_name))
    except Exception as e:
        print(f"Warning: No se pudieron cargar categorías ({e}). Usando fallback.")
        return {0: "Categoría Genérica"}


def load_system() -> Tuple[Optional[Any], Optional[list], Optional[Any], Optional[Dict]]:
    """Carga los modelos y metadatos usando rutas absolutas."""
    try:
        # Carga robusta usando MODELS_DIR
        model = joblib.load(os.path.join(MODELS_DIR, "stacking_model.pkl"))
        features = joblib.load(os.path.join(MODELS_DIR, "features.pkl"))
        shap_model = joblib.load(os.path.join(MODELS_DIR, "xgb_simple_shap.pkl"))

        # Cargar precios (si no existe, retorna dict vacío)
        try:
            cat_prices = joblib.load(os.path.join(MODELS_DIR, "category_prices.pkl"))
        except FileNotFoundError:
            print("Warning: category_prices.pkl no encontrado. Ejecuta training.")
            cat_prices = {}

        return model, features, shap_model, cat_prices
    except FileNotFoundError as e:
        print(
            f"Error Crítico: No se encontraron los archivos del modelo en {MODELS_DIR}. Detalle: {e}"
        )
        return None, None, None, None


def predict_demand(model: Any, input_data: Dict[str, float]) -> float:
    """Realiza la predicción."""
    # Convertir a DataFrame asegurando el orden correcto si es necesario
    df = pd.DataFrame([input_data])

    # Predecir (el modelo devuelve log)
    pred_log = model.predict(df)

    # Invertir transformación logarítmica (expm1)
    pred_real = np.expm1(pred_log)

    # Retornar float asegurando no negativos
    return float(max(0.0, pred_real[0]))
