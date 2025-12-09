"""Servicio para predicciones de demanda vía API REST."""

from typing import Dict, Tuple, Any, Optional
import pandas as pd
import numpy as np
import shap
import httpx
import os
import streamlit as st


class PredictionService:
    """Servicio para predicciones de demanda usando API REST exclusivamente."""

    def __init__(self, shap_model: Any, api_url: Optional[str] = None):
        """Inicializa el servicio con la URL de la API y modelo SHAP.

        Parámetros:
            shap_model: modelo local para análisis SHAP (no se usa para predicción)
            api_url: URL de la API REST (default: http://localhost:8000)
        """
        self.shap_model = shap_model
        self.api_url = api_url or os.getenv("API_URL", "http://localhost:8000")
        self._feature_names: Optional[list] = None
        self._rolling_windows: Optional[list] = None

    def predict(self, input_data: Dict) -> float:
        """Realiza una predicción de demanda consultando la API REST."""
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(f"{self.api_url}/predict", json=input_data)
                response.raise_for_status()
                result = response.json()
                return result["prediction"]
        except httpx.ConnectError:
            st.error(
                f"❌ No se pudo conectar con la API en {self.api_url}. "
                "Asegúrate de que el servidor esté corriendo con: `pipenv run api`",
                icon=":material/error:",
            )
            st.stop()
        except httpx.HTTPError as e:
            st.error(f"Error HTTP al consultar la API: {e}", icon=":material/warning:")
            st.stop()
        except Exception as e:
            st.error(f"❌ Error inesperado: {e}", icon=":material/error:")
            st.stop()

    def calculate_shap_values(self, input_data: Dict) -> Tuple[Any, pd.DataFrame, Any]:
        """Calcula los valores SHAP para explicar la predicción.

        Nota: SHAP siempre se calcula localmente por limitaciones de serialización.
        """
        # Obtener schema de la API si no lo tenemos
        if self._feature_names is None or self._rolling_windows is None:
            self._fetch_model_schema()

        # Asegurarse de que input_data tenga todas las features necesarias
        complete_input = self._prepare_features_for_shap(input_data)

        # Construir lista de features dinámicamente según el modelo cargado
        custom_windows = input_data.get("rolling_windows", self._rolling_windows)

        # Detectar si el modelo tiene features nuevas (con _log) o antiguas
        if self._feature_names and "item_price_log" in self._feature_names:
            # Modelo nuevo con features de pricing normalizadas
            feature_names = [
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
        else:
            # Modelo antiguo con features básicas
            feature_names = [
                "shop_cluster",
                "item_category_id",
                "item_price",
                "item_cnt_lag_1",
                "item_cnt_lag_2",
                "item_cnt_lag_3",
            ]

        # Agregar rolling features en el orden correcto
        for window in custom_windows:
            feature_names.append(f"rolling_mean_{window}")
            feature_names.append(f"rolling_std_{window}")

        explainer = shap.TreeExplainer(self.shap_model)
        feat_df = pd.DataFrame([complete_input])

        # Filtrar solo las features que existen en complete_input
        available_features = [f for f in feature_names if f in feat_df.columns]
        feat_df = feat_df[available_features]

        shap_values = explainer.shap_values(feat_df)
        return shap_values, feat_df, explainer.expected_value

    def _fetch_model_schema(self) -> None:
        """Obtiene el schema del modelo desde la API."""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.api_url}/health")
                response.raise_for_status()
                data = response.json()

                # Extraer rolling windows del health check
                metrics = data.get("model_metrics", {})
                self._rolling_windows = metrics.get("rolling_windows", [3, 6])

                # Construir nombres de features con pricing
                self._feature_names = [
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

                # Agregar rolling features dinámicamente
                for window in self._rolling_windows:
                    self._feature_names.append(f"rolling_mean_{window}")
                    self._feature_names.append(f"rolling_std_{window}")

        except Exception as e:
            # Fallback a configuración por defecto
            st.warning(f"No se pudo obtener schema de la API: {e}. Usando valores por defecto.")
            self._rolling_windows = [3, 6]
            self._feature_names = [
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
                "rolling_mean_3",
                "rolling_std_3",
                "rolling_mean_6",
                "rolling_std_6",
            ]

    def _prepare_features_for_shap(self, input_data: Dict) -> Dict:
        """Prepara los datos de entrada con todas las features necesarias incluyendo pricing."""
        # Features base
        item_price = input_data["item_price"]
        item_category_id = input_data["item_category_id"]
        item_cnt_lag_1 = input_data["item_cnt_lag_1"]
        item_cnt_lag_2 = input_data["item_cnt_lag_2"]
        item_cnt_lag_3 = input_data["item_cnt_lag_3"]

        # Calcular pricing features localmente (igual que en la API)
        price_rel_category = 1.0  # Neutral por defecto

        # Features de pricing normalizadas
        item_price_log = float(np.log1p(item_price))
        item_cnt_lag_1_log = float(np.log1p(item_cnt_lag_1))
        item_cnt_lag_2_log = float(np.log1p(item_cnt_lag_2))
        item_cnt_lag_3_log = float(np.log1p(item_cnt_lag_3))

        # Preparar features tanto antiguas como nuevas para compatibilidad
        complete_input = {
            "shop_cluster": input_data["shop_cluster"],
            "item_category_id": item_category_id,
            # Features antiguas (sin transformación)
            "item_price": item_price,
            "item_cnt_lag_1": item_cnt_lag_1,
            "item_cnt_lag_2": item_cnt_lag_2,
            "item_cnt_lag_3": item_cnt_lag_3,
            # Features nuevas (con transformación log)
            "item_price_log": item_price_log,
            "item_cnt_lag_1_log": item_cnt_lag_1_log,
            "item_cnt_lag_2_log": item_cnt_lag_2_log,
            "item_cnt_lag_3_log": item_cnt_lag_3_log,
            "price_rel_category": price_rel_category,
            "price_rel_category_log": float(np.log1p(price_rel_category)),
            "price_discount": 0.0,
            "is_new_price": 0,
            "price_change_pct": 0.0,
            "price_change_2m_pct": 0.0,
            "revenue_potential_log": float(np.log1p(item_cnt_lag_1 * item_price)),
            "price_demand_elasticity": 0.0,
        }

        # Usar rolling_windows del input_data si está presente, sino usar self._rolling_windows
        custom_windows = input_data.get("rolling_windows", self._rolling_windows)

        # Agregar rolling features según configuración
        if custom_windows:
            lags = [
                input_data["item_cnt_lag_1"],
                input_data["item_cnt_lag_2"],
                input_data["item_cnt_lag_3"],
            ]

            for window in custom_windows:
                mean_key = f"rolling_mean_{window}"
                std_key = f"rolling_std_{window}"

                # Si el input_data ya tiene estos valores, usarlos
                if mean_key in input_data:
                    complete_input[mean_key] = input_data[mean_key]
                    complete_input[std_key] = input_data.get(std_key, 0.0)
                else:
                    # Calcular aproximaciones
                    complete_input[mean_key] = float(np.mean(lags))
                    complete_input[std_key] = float(np.std(lags))

        return complete_input

    def check_api_health(self) -> bool:
        """Verifica si la API está disponible y saludable."""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.api_url}/health")
                response.raise_for_status()
                data = response.json()
                return data.get("status") == "healthy"
        except Exception:
            return False
