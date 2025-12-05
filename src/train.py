import pandas as pd
import numpy as np
import joblib
import os

from src.data_processing import prepare_full_pipeline
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- Configuración de Directorios ---
# Definimos la ruta absoluta a la carpeta 'models' en la raíz del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def train_models() -> None:
    # 1. Obtener datos procesados
    train, val, _ = prepare_full_pipeline()

    features = [
        "shop_cluster",
        "item_category_id",
        "item_price",
        "item_cnt_lag_1",
        "item_cnt_lag_2",
        "item_cnt_lag_3",
    ]
    target = "target_log"

    X_train = train[features]
    y_train = train[target]
    X_val = val[features]
    y_val = val[target]

    print(f"🚀 Iniciando entrenamiento con {X_train.shape[0]} muestras...")

    # --- NUEVO: Calcular y Guardar Precios Promedio por Categoría ---
    print("💾 Generando metadatos de precios (category_prices.pkl)...")
    # Calculamos la mediana para ser robustos ante outliers de precios
    category_prices = train.groupby("item_category_id")["item_price"].median().to_dict()
    joblib.dump(category_prices, os.path.join(MODELS_DIR, "category_prices.pkl"))
    # -------------------------------------------------------------

    # 2. Definir Modelos Base
    estimators = [
        ("rf", RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)),
        ("xgb", XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=42)),
    ]

    # 3. Stacking (Ensemble Learning)
    print("🧠 Entrenando Stacking Ensemble (XGBoost + Random Forest)...")
    stacking_model = StackingRegressor(
        estimators=estimators, final_estimator=LinearRegression(), n_jobs=-1
    )

    stacking_model.fit(X_train, y_train)

    # 4. Evaluación Rápida
    print("📉 Evaluando modelo...")
    preds_log = stacking_model.predict(X_val)
    preds = np.expm1(preds_log)  # Invertir logaritmo
    y_true = np.expm1(y_val)

    rmse = np.sqrt(mean_squared_error(y_true, preds))
    r2 = r2_score(y_true, preds)
    print(f"✅ Resultados Validación -> RMSE: {rmse:.4f} | R2: {r2:.4f}")

    # 5. Guardar Modelos
    joblib.dump(stacking_model, os.path.join(MODELS_DIR, "stacking_model.pkl"))
    joblib.dump(features, os.path.join(MODELS_DIR, "features.pkl"))
    print(f"💾 Modelo guardado en: {os.path.join(MODELS_DIR, 'stacking_model.pkl')}")

    # Guardar modelo simple para SHAP (TreeExplainer no soporta Stacking directo fácilmente)
    print("💾 Guardando modelo proxy para SHAP...")
    xgb_simple = XGBRegressor(n_estimators=50, max_depth=5).fit(X_train, y_train)
    joblib.dump(xgb_simple, os.path.join(MODELS_DIR, "xgb_simple_shap.pkl"))


if __name__ == "__main__":
    train_models()
