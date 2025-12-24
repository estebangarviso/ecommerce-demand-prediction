from typing import List, Optional
import pandas as pd
import numpy as np
import joblib
import os
import json
import warnings

from src.data_processing import (
    prepare_full_pipeline,
    DEFAULT_ROLLING_WINDOWS,
    validate_rolling_windows,
)
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin

# Deep Learning imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

warnings.filterwarnings("ignore", category=UserWarning)

# Configuración de directorios
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


class KerasRegressor(BaseEstimator, RegressorMixin):
    """Wrapper para modelos Keras compatible con sklearn (para Stacking)."""

    def __init__(self, model_builder, epochs=50, batch_size=32, verbose=0):
        self.model_builder = model_builder
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None

    def fit(self, X, y):
        """Entrena el modelo Keras."""
        self.model = self.model_builder(input_dim=X.shape[1])
        early_stop = callbacks.EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
        self.model.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            callbacks=[early_stop],
        )
        return self

    def predict(self, X):
        """Realiza predicciones."""
        return self.model.predict(X, verbose=0).flatten()


def build_mlp_model(input_dim: int) -> keras.Model:
    """Construye un Multi-Layer Perceptron para regresión.

    Arquitectura: 3 capas ocultas con dropout para regularización.
    """
    model = keras.Sequential(
        [
            layers.Dense(128, activation="relu", input_dim=input_dim),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),  # Salida: predicción continua
        ]
    )

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

    return model


def build_lstm_model(input_dim: int) -> keras.Model:
    """Construye una red LSTM simplificada para series temporales.

    Nota: En datasets tabulares pequeños, arquitecturas simples tipo DNN
    suelen funcionar mejor que LSTM tradicionales.
    """
    model = keras.Sequential(
        [
            layers.Dense(64, activation="relu", input_dim=input_dim),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(32, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),  # learning rate más bajo
        loss="mse",
        metrics=["mae"],
    )

    return model


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    """Calcula métricas de evaluación para un modelo.

    Retorna diccionario con RMSE, MAE y R2.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    metrics = {"model": model_name, "rmse": float(rmse), "mae": float(mae), "r2": float(r2)}

    print(f"  {model_name:20s} -> RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

    return metrics


def train_models(use_balancing: bool = False, rolling_windows: Optional[List[int]] = None) -> None:
    """Pipeline completo de entrenamiento con modelos tradicionales y Deep Learning.

    Parámetros:
        use_balancing: activar SMOTE en entrenamiento
        rolling_windows: tamaños de ventanas rolling (None = usar DEFAULT_ROLLING_WINDOWS)
    """
    # Validar y usar ventanas rolling
    if rolling_windows is None:
        rolling_windows = DEFAULT_ROLLING_WINDOWS
    rolling_windows = validate_rolling_windows(rolling_windows)

    # Obtener datos procesados (ahora con rolling windows parametrizados)
    train, val, _, tscv = prepare_full_pipeline(
        use_balancing=use_balancing, rolling_windows=rolling_windows
    )

    # Generar features dinámicamente basadas en rolling_windows
    features = [
        "shop_cluster",
        "item_category_id",
        "item_price_log",  # Precio normalizado con log
        "item_cnt_lag_1_log",  # Lags de ventas normalizados
        "item_cnt_lag_2_log",
        "item_cnt_lag_3_log",
        "price_rel_category",  # Precio relativo a categoría
        "price_rel_category_log",  # Versión normalizada
        "price_discount",  # Ratio de descuento
        "is_new_price",  # Indicador de cambio de precio
        "price_change_pct",  # Cambio porcentual de precio
        "price_change_2m_pct",  # Cambio de precio en 2 meses
        "revenue_potential_log",  # Ingreso potencial normalizado
        "price_demand_elasticity",  # Elasticidad precio-demanda
    ]

    # Agregar features de rolling windows dinámicamente
    for window in rolling_windows:
        features.append(f"rolling_mean_{window}")
        features.append(f"rolling_std_{window}")
    target = "target_log"

    X_train = train[features].values
    y_train = train[target].values
    X_val = val[features].values
    y_val = val[target].values

    print(f"🚀 Iniciando entrenamiento con {X_train.shape[0]} muestras...")
    print(f"📊 Features: {len(features)}")
    print(f"   - Lags normalizados: lag_1_log, lag_2_log, lag_3_log")
    print(f"   - Rolling windows: {rolling_windows}")
    print(f"   - Pricing: price_rel_category, price_discount, is_new_price")
    print(f"   - Elasticidad: price_demand_elasticity, price_change_pct")
    print(f"   - Otras: shop_cluster, item_category_id")

    # Calcular y guardar precios promedio por categoría para inferencia
    print("💾 Generando metadatos de precios (category_prices.pkl)...")
    category_prices = train.groupby("item_category_id")["item_price"].median().to_dict()
    joblib.dump(category_prices, os.path.join(MODELS_DIR, "category_prices.pkl"))

    # Diccionario para almacenar métricas de todos los modelos
    all_metrics = []

    # Entrenar modelos tradicionales
    print("\n🔨 Entrenando modelos tradicionales...")

    rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_val)
    all_metrics.append(evaluate_model(np.expm1(y_val), np.expm1(rf_preds), "Random Forest"))

    # Configurar restricciones monotónicas para XGBoost
    # Mapear nombres de features a índices para monotone_constraints
    feature_names = features
    monotone_dict = {}

    # Restricción decreciente (-1) para variables de precio
    # A mayor precio, menor demanda esperada
    price_features_monotone = ["item_price_log", "price_rel_category", "price_rel_category_log"]

    for feat in price_features_monotone:
        if feat in feature_names:
            idx = feature_names.index(feat)
            monotone_dict[idx] = -1  # Restricción decreciente

    # Convertir diccionario a tupla ordenada por índice
    monotone_constraints = tuple(monotone_dict.get(i, 0) for i in range(len(feature_names)))

    print(
        f"  Restricciones monotónicas aplicadas: {sum(1 for x in monotone_constraints if x != 0)} features"
    )

    xgb_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=7,
        random_state=42,
        monotone_constraints=monotone_constraints,
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_val)
    all_metrics.append(evaluate_model(np.expm1(y_val), np.expm1(xgb_preds), "XGBoost"))

    # Entrenar modelos de Deep Learning
    print("\n🧠 Entrenando modelos de Deep Learning...")

    # Normalizar features para Deep Learning (importante para convergencia)
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Guardar scaler para inferencia
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    # MLP (Multi-Layer Perceptron)
    print("  Entrenando MLP...")
    mlp_model = build_mlp_model(input_dim=X_train_scaled.shape[1])
    early_stop = callbacks.EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)
    mlp_model.fit(
        X_train_scaled, y_train, epochs=100, batch_size=64, verbose=0, callbacks=[early_stop]
    )
    mlp_preds = mlp_model.predict(X_val_scaled, verbose=0).flatten()
    all_metrics.append(evaluate_model(np.expm1(y_val), np.expm1(mlp_preds), "MLP"))

    # LSTM simplificada (arquitectura tipo DNN para datos tabulares)
    print("  Entrenando LSTM (DNN Architecture)...")
    lstm_model = build_lstm_model(input_dim=X_train_scaled.shape[1])
    lstm_model.fit(
        X_train_scaled, y_train, epochs=150, batch_size=32, verbose=0, callbacks=[early_stop]
    )
    lstm_preds = lstm_model.predict(X_val_scaled, verbose=0).flatten()
    all_metrics.append(evaluate_model(np.expm1(y_val), np.expm1(lstm_preds), "LSTM-DNN"))

    # Stacking Regressor con modelos tradicionales
    print("\n🏗️  Entrenando Stacking Ensemble...")
    estimators = [
        ("rf", RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)),
        ("xgb", XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=42)),
    ]

    stacking_model = StackingRegressor(
        estimators=estimators, final_estimator=LinearRegression(), n_jobs=-1
    )
    stacking_model.fit(X_train, y_train)
    stacking_preds = stacking_model.predict(X_val)
    all_metrics.append(
        evaluate_model(np.expm1(y_val), np.expm1(stacking_preds), "Stacking Ensemble")
    )

    # Guardar métricas en JSON
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n📊 Métricas guardadas en: {metrics_path}")

    # Guardar modelos principales y configuración
    print(f"\n💾 Guardando modelos y configuración...")
    joblib.dump(stacking_model, os.path.join(MODELS_DIR, "stacking_model.pkl"))
    joblib.dump(features, os.path.join(MODELS_DIR, "features.pkl"))
    joblib.dump(rolling_windows, os.path.join(MODELS_DIR, "rolling_windows.pkl"))

    # Guardar modelos de Deep Learning
    mlp_model.save(os.path.join(MODELS_DIR, "mlp_model.keras"))
    lstm_model.save(os.path.join(MODELS_DIR, "lstm_model.keras"))

    # Modelo simple para SHAP (TreeExplainer requiere modelos de árbol)
    xgb_simple = XGBRegressor(n_estimators=50, max_depth=5).fit(X_train, y_train)
    joblib.dump(xgb_simple, os.path.join(MODELS_DIR, "xgb_simple_shap.pkl"))

    print(f"✅ Entrenamiento completado. Modelos guardados en: {MODELS_DIR}")

    # ===========================================
    # Export predictions for technical analysis
    # ===========================================
    print(f"\n📦 Exporting predictions for analysis...")

    # Create exports directory
    EXPORTS_DIR = os.path.join(os.path.dirname(MODELS_DIR), "exports")
    os.makedirs(EXPORTS_DIR, exist_ok=True)

    # Export predictions with residuals for key models
    models_to_export = {"randomforest": rf_model, "xgboost": xgb_model, "stacking": stacking_model}

    for model_name, model in models_to_export.items():
        # Generate predictions on validation set
        y_pred_log = model.predict(X_val)

        # Convert to original scale
        y_true_original = np.expm1(y_val)
        y_pred_original = np.expm1(y_pred_log)

        # Create predictions DataFrame with metadata
        predictions_df = pd.DataFrame(
            {
                "y_true": y_true_original.values,
                "y_pred": y_pred_original,
                "residual": y_true_original.values - y_pred_original,
                "shop_cluster": val["shop_cluster"].values,
                "item_category_id": val["item_category_id"].values,
                "date_block_num": val["date_block_num"].values,
            }
        )

        # Save to CSV
        output_path = os.path.join(EXPORTS_DIR, f"predictions_{model_name}_val.csv")
        predictions_df.to_csv(output_path, index=False)
        print(f"  ✅ Exported: predictions_{model_name}_val.csv")

    print(f"📂 Predictions exported to: {EXPORTS_DIR}")

    # Mostrar ranking de modelos
    sorted_metrics = sorted(all_metrics, key=lambda x: x["r2"], reverse=True)
    print(f"\n🏆 Ranking de modelos (por R²):")
    for i, m in enumerate(sorted_metrics, 1):
        print(f"  {i}. {m['model']:25s} -> R²: {m['r2']:.4f}")

    # Nota sobre uso de modelos
    print("\n📝 Nota: El Stacking Ensemble se usa en producción por su balance")
    print("   entre rendimiento y estabilidad. Los modelos de Deep Learning")
    print("   están disponibles para comparación y futuros experimentos.")


if __name__ == "__main__":
    train_models()
