"""Servicio para exportar datos de análisis técnico."""

import os
import json
import sys
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np
import joblib
import shap

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src import data_processing


class DataExporter:
    """Servicio para exportar datos de modelos entrenados."""

    def __init__(self):
        """Inicializa el exportador de datos."""
        self.project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        self.models_dir = os.path.join(self.project_root, "models")
        self.exports_dir = os.path.join(self.project_root, "exports")

        # Model file mapping
        self.model_file_map = {
            "Random Forest": "stacking_model.pkl",
            "XGBoost": "xgb_simple_shap.pkl",
            "Stacking Ensemble": "stacking_model.pkl",
            "MLP": "mlp_model.keras",
            "LSTM-DNN": "lstm_model.keras",
        }

    def export_all(self) -> Tuple[bool, str]:
        """
        Ejecuta la exportación completa de datos.

        Returns:
            Tuple con (éxito, mensaje)
        """
        try:
            # Create exports directory
            os.makedirs(self.exports_dir, exist_ok=True)

            # Load data
            train, val, test, tscv = data_processing.prepare_full_pipeline()
            rolling_windows = joblib.load(os.path.join(self.models_dir, "rolling_windows.pkl"))

            # Define features
            features = [
                "shop_cluster",
                "item_category_id",
                "item_price_log",
                "item_cnt_lag_1_log",
                "item_cnt_lag_2_log",
                "item_cnt_lag_3_log",
                "price_rel_category",
                "price_rel_category_log",
                "price_discount",
                "is_new_price",
                "price_change_pct",
                "price_change_2m_pct",
                "revenue_potential_log",
                "price_demand_elasticity",
            ]

            # Add rolling window features
            for window in rolling_windows:
                features.append(f"rolling_mean_{window}")
                features.append(f"rolling_std_{window}")

            target = "target_log"
            X_val = val[features]
            y_val = val[target]

            # Export all data
            stats = {
                "metrics": self._export_metrics(),
                "predictions": self._export_predictions(X_val, y_val, val),
                "features": self._export_features(X_val, y_val, val),
                "shap": self._export_shap(X_val, y_val),
                "segments": self._export_segments(val),
            }

            success_count = sum(1 for v in stats.values() if v)
            message = f"✅ Exportación completada: {success_count}/{len(stats)} archivos generados en {self.exports_dir}"

            return True, message

        except Exception as e:
            return False, f"❌ Error en exportación: {str(e)}"

    def _export_metrics(self) -> bool:
        """Exporta métricas globales."""
        try:
            metrics_path = os.path.join(self.models_dir, "metrics.json")
            if not os.path.exists(metrics_path):
                return False

            with open(metrics_path, "r") as f:
                metrics_data = json.load(f)

            metrics_list = []

            if isinstance(metrics_data, list):
                metrics_list = metrics_data
                for item in metrics_list:
                    if "split" not in item:
                        item["split"] = "val"
                    if "model_size_mb" not in item:
                        item["model_size_mb"] = None
            elif isinstance(metrics_data, dict):
                for model_name, model_metrics in metrics_data.items():
                    metrics_list.append(
                        {
                            "model": model_name,
                            "split": "val",
                            "rmse": model_metrics.get("rmse", None),
                            "mae": model_metrics.get("mae", None),
                            "r2": model_metrics.get("r2", None),
                            "train_time_s": model_metrics.get("train_time", None),
                            "model_size_mb": None,
                        }
                    )

            metrics_df = pd.DataFrame(metrics_list)

            # Calculate model sizes
            for idx, row in metrics_df.iterrows():
                model_name = row["model"]

                if model_name in self.model_file_map:
                    model_file = os.path.join(self.models_dir, self.model_file_map[model_name])
                else:
                    model_file = os.path.join(
                        self.models_dir, f"{model_name.lower().replace(' ', '_')}_model.pkl"
                    )

                if os.path.exists(model_file):
                    size_mb = os.path.getsize(model_file) / (1024 * 1024)
                    metrics_df.at[idx, "model_size_mb"] = round(size_mb, 2)

            output_path = os.path.join(self.exports_dir, "metrics_overall.csv")
            metrics_df.to_csv(output_path, index=False)
            return True

        except Exception as e:
            print(f"Error exporting metrics: {e}")
            return False

    def _export_predictions(self, X_val: pd.DataFrame, y_val: pd.Series, val: pd.DataFrame) -> bool:
        """Exporta predicciones con residuales."""
        try:
            models_to_export = {
                "randomforest": os.path.join(self.models_dir, "stacking_model.pkl"),
                "xgboost": os.path.join(self.models_dir, "xgb_simple_shap.pkl"),
                "stacking": os.path.join(self.models_dir, "stacking_model.pkl"),
            }

            exported_count = 0
            for model_name, model_path in models_to_export.items():
                if os.path.exists(model_path):
                    try:
                        loaded_model = joblib.load(model_path)

                        # For stacking model, extract base estimator for RF
                        if model_name == "randomforest" and hasattr(loaded_model, "estimators_"):
                            loaded_model = loaded_model.estimators_[0]

                        # Generate predictions
                        y_pred_log = loaded_model.predict(X_val)

                        # Convert back to original scale
                        y_true_original = np.expm1(y_val)
                        y_pred_original = np.expm1(y_pred_log)

                        # Create predictions DataFrame
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

                        output_path = os.path.join(
                            self.exports_dir, f"predictions_{model_name}_val.csv"
                        )
                        predictions_df.to_csv(output_path, index=False)
                        exported_count += 1

                    except Exception as e:
                        print(f"Failed to export {model_name}: {e}")

            return exported_count > 0

        except Exception as e:
            print(f"Error exporting predictions: {e}")
            return False

    def _export_features(self, X_val: pd.DataFrame, y_val: pd.Series, val: pd.DataFrame) -> bool:
        """Exporta features de validación."""
        try:
            features_df = X_val.copy()
            features_df["shop_cluster"] = val["shop_cluster"].values
            features_df["item_category_id"] = val["item_category_id"].values
            features_df["date_block_num"] = val["date_block_num"].values
            features_df["target_log"] = y_val.values

            output_path = os.path.join(self.exports_dir, "features_val.csv")
            features_df.to_csv(output_path, index=False)
            return True

        except Exception as e:
            print(f"Error exporting features: {e}")
            return False

    def _export_shap(self, X_val: pd.DataFrame, y_val: pd.Series) -> bool:
        """Exporta análisis SHAP para RandomForest y XGBoost."""
        try:
            # Models to generate SHAP for
            models_for_shap = []

            # 1. Extract RandomForest from stacking model
            stacking_path = os.path.join(self.models_dir, "stacking_model.pkl")
            if os.path.exists(stacking_path):
                stacking_model = joblib.load(stacking_path)
                if hasattr(stacking_model, "estimators_"):
                    for estimator in stacking_model.estimators_:
                        if hasattr(estimator, "tree_") or hasattr(estimator, "estimators_"):
                            # Found RandomForest
                            models_for_shap.append(("randomforest", estimator))
                            break

            # 2. Load XGBoost model
            xgb_path = os.path.join(self.models_dir, "xgb_simple_shap.pkl")
            if os.path.exists(xgb_path):
                xgb_model = joblib.load(xgb_path)
                models_for_shap.append(("xgboost", xgb_model))

            if not models_for_shap:
                return False

            # Generate SHAP for each model
            exported_count = 0
            for model_name, model in models_for_shap:
                try:
                    # Calculate SHAP values
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_val)

                    if isinstance(shap_values, list):
                        shap_values = shap_values[0]

                    # Create SHAP summary
                    shap_summary_df = pd.DataFrame(
                        {
                            "feature": X_val.columns,
                            "mean_abs_shap_value": np.abs(shap_values).mean(axis=0),
                            "mean_shap_value": shap_values.mean(axis=0),
                            "std_shap_value": shap_values.std(axis=0),
                        }
                    ).sort_values("mean_abs_shap_value", ascending=False)

                    shap_summary_df["rank"] = range(1, len(shap_summary_df) + 1)

                    output_path = os.path.join(
                        self.exports_dir, f"shap_summary_{model_name}_val.csv"
                    )
                    shap_summary_df.to_csv(output_path, index=False)
                    exported_count += 1
                    print(f"✅ SHAP exported for {model_name}")

                except Exception as e:
                    print(f"⚠️ Failed to export SHAP for {model_name}: {e}")

            return exported_count > 0

        except Exception as e:
            print(f"Error exporting SHAP: {e}")
            return False

    def _export_segments(self, val: pd.DataFrame) -> bool:
        """Exporta mapa de segmentos."""
        try:
            segments_df = val[
                ["shop_cluster", "item_category_id", "date_block_num"]
            ].drop_duplicates()

            cluster_map = {0: "Tienda Pequeña", 1: "Supermercado Mediano", 2: "Megatienda"}
            segments_df["cluster_name"] = segments_df["shop_cluster"].map(cluster_map)

            output_path = os.path.join(self.exports_dir, "segments_map.csv")
            segments_df.to_csv(output_path, index=False)
            return True

        except Exception as e:
            print(f"Error exporting segments: {e}")
            return False
