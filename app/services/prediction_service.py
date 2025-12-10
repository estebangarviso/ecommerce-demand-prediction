"""Servicio para predicciones de demanda vía API REST."""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import shap
import httpx
import os
import joblib
import streamlit as st


class PredictionService:
    """Servicio para predicciones de demanda usando API REST exclusivamente."""

    def __init__(self, shap_model: Optional[Any] = None, api_url: Optional[str] = None):
        """Inicializa el servicio con la URL de la API.

        Parámetros:
            shap_model: modelo local para análisis SHAP (opcional, se carga si es necesario)
            api_url: URL de la API REST (default: http://localhost:8000)
        """
        self.shap_model = shap_model
        self.api_url = api_url or os.getenv("API_URL", "http://localhost:8000")
        self._feature_names: Optional[list] = None
        self._rolling_windows: Optional[list] = None
        self._categories: Optional[Dict] = None
        self._category_prices: Optional[Dict] = None

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

    def calculate_shap_values(self, input_data: Dict) -> shap.Explanation:
        """Calcula los valores SHAP para explicar la predicción.

        Retorna un objeto shap.Explanation completo que permite visualizaciones
        avanzadas como waterfall plots y análisis detallados de contribución.

        Nota: SHAP siempre se calcula localmente por limitaciones de serialización.
        """
        # Cargar modelo SHAP si no está disponible
        self._load_shap_model_if_needed()

        if self.shap_model is None:
            st.error("⚠️ Modelo SHAP no disponible. Verifica que exista `models/stacking_model.pkl`")
            raise ValueError("Modelo SHAP no disponible")

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

        try:
            feat_df = pd.DataFrame([complete_input])

            # Filtrar solo las features que existen en complete_input
            available_features = [f for f in feature_names if f in feat_df.columns]

            if len(available_features) == 0:
                st.error(
                    f"❌ No hay features disponibles. Expected: {feature_names}, Got: {list(feat_df.columns)}"
                )
                raise ValueError("No features disponibles para SHAP")

            feat_df = feat_df[available_features]

            # CRÍTICO: Convertir todas las columnas a float para evitar errores de dtype
            for col in feat_df.columns:
                feat_df[col] = pd.to_numeric(feat_df[col], errors="coerce")

            # Verificar que no haya NaN después de la conversión
            if feat_df.isnull().any().any():
                st.error(
                    f"❌ Datos inválidos detectados después de conversión: {feat_df.isnull().sum().to_dict()}"
                )
                raise ValueError("Datos contienen valores no numéricos")

            # Para StackingRegressor, necesitamos usar KernelExplainer o LinearExplainer
            # porque el estimador final solo recibe las predicciones de los modelos base
            if hasattr(self.shap_model, "final_estimator_"):
                # Es un StackingRegressor - usar explainers de los modelos base
                # st.info("🔍 Calculando SHAP desde modelo base (optimizado)", icon=":material/analytics:")

                # Usar el primer estimador base (típicamente el más importante)
                # Esto es mucho más rápido que KernelExplainer en el stacking completo
                base_estimators = self.shap_model.estimators_

                if len(base_estimators) > 0:
                    # Usar el primer modelo base (usualmente Random Forest o similar)
                    base_model = base_estimators[0]

                    try:
                        # Intentar TreeExplainer con el modelo base
                        explainer = shap.TreeExplainer(base_model)
                        shap_explanation = explainer(feat_df)
                    except Exception:
                        # Si falla, usar aproximación simple con LinearExplainer
                        # Crear un explainer lineal simple basado en feature importance
                        st.warning(
                            "⚠️ Usando aproximación simplificada de SHAP", icon=":material/info:"
                        )

                        # Calcular importancias de features si están disponibles
                        if hasattr(base_model, "feature_importances_"):
                            importances = base_model.feature_importances_
                            prediction = base_model.predict(feat_df)[0]
                            base_value = np.mean(base_model.predict(feat_df))

                            # Aproximar SHAP values usando feature importance
                            shap_values = np.array(
                                [importances * (feat_df.values[0] - feat_df.values[0].mean())]
                            )

                            shap_explanation = shap.Explanation(
                                values=shap_values,
                                base_values=base_value,
                                data=feat_df.values,
                                feature_names=feat_df.columns.tolist(),
                            )
                        else:
                            raise ValueError("No se puede calcular SHAP para este modelo")
                else:
                    raise ValueError("No hay estimadores base disponibles")
            else:
                # Modelo directo (no stacking) - usar TreeExplainer
                try:
                    explainer = shap.TreeExplainer(self.shap_model)
                    shap_explanation = explainer(feat_df)
                except Exception:
                    # Fallback a KernelExplainer
                    background = shap.sample(feat_df, min(10, len(feat_df)))
                    explainer = shap.KernelExplainer(self.shap_model.predict, background)
                    shap_values = explainer.shap_values(feat_df)
                    shap_explanation = shap.Explanation(
                        values=shap_values,
                        base_values=explainer.expected_value,
                        data=feat_df.values,
                        feature_names=feat_df.columns.tolist(),
                    )
        except Exception as e:
            st.error(f"❌ Error en SHAP: {type(e).__name__}: {str(e)}")
            raise
        return shap_explanation

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

    def check_api_health(self) -> tuple[bool, Optional[Dict]]:
        """Verifica si la API está disponible y saludable.

        Retorna:
            tuple: (is_healthy, health_data_dict)
        """
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.api_url}/health")
                response.raise_for_status()
                data = response.json()
                is_healthy = data.get("status") == "healthy"
                return is_healthy, data
        except Exception:
            return False, None

    def get_categories(self) -> Dict:
        """Obtiene el mapa de categorías desde la API.

        Retorna:
            Dict con {category_id: category_name}
        """
        if self._categories is not None:
            return self._categories

        try:
            with httpx.Client(timeout=5.0) as client:
                # Asumiendo que la API tiene un endpoint /categories
                # Si no existe, lo implementaremos
                response = client.get(f"{self.api_url}/categories")
                response.raise_for_status()
                self._categories = response.json()
                return self._categories
        except httpx.HTTPStatusError:
            # Fallback: cargar desde archivo local (temporal)
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            categories_path = os.path.join(base_dir, "data", "item_categories.csv")

            if os.path.exists(categories_path):
                df = pd.read_csv(categories_path)
                self._categories = dict(zip(df["item_category_id"], df["item_category_name"]))
                return self._categories

            st.warning("No se pudieron cargar las categorías")
            return {}
        except Exception as e:
            st.warning(f"Error al cargar categorías: {e}")
            return {}

    def get_category_prices(self) -> Dict:
        """Obtiene los precios promedio por categoría desde la API.

        Retorna:
            Dict con {category_id: average_price}
        """
        if self._category_prices is not None:
            return self._category_prices

        try:
            with httpx.Client(timeout=5.0) as client:
                # Obtener todos los precios de categorías
                response = client.get(f"{self.api_url}/prices")
                response.raise_for_status()
                self._category_prices = response.json()
                return self._category_prices
        except httpx.HTTPStatusError:
            # Fallback: cargar desde modelo local
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            prices_path = os.path.join(base_dir, "models", "category_prices.pkl")

            if os.path.exists(prices_path):
                self._category_prices = joblib.load(prices_path)
                return self._category_prices

            st.warning("No se pudieron cargar los precios")
            return {}
        except Exception as e:
            st.warning(f"Error al cargar precios: {e}")
            return {}

    def _load_shap_model_if_needed(self) -> None:
        """Carga el modelo SHAP localmente si no está disponible."""
        if self.shap_model is not None:
            return

        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            model_path = os.path.join(base_dir, "models", "stacking_model.pkl")

            if os.path.exists(model_path):
                self.shap_model = joblib.load(model_path)
            else:
                st.warning("Modelo SHAP no disponible. Las explicaciones estarán limitadas.")
        except Exception as e:
            st.warning(f"No se pudo cargar el modelo SHAP: {e}")

    def regenerate_datasets(self) -> tuple[bool, str]:
        """Solicita a la API regenerar los datasets desde KaggleHub.

        Returns:
            tuple: (success, message)
        """
        try:
            with httpx.Client(timeout=300.0) as client:  # 5 minutos timeout
                response = client.post(f"{self.api_url}/regenerate-datasets")
                response.raise_for_status()
                result = response.json()
                return True, result.get("message", "Datasets regenerados exitosamente")
        except httpx.HTTPError as e:
            return False, f"Error al regenerar datasets vía API: {str(e)}"
        except Exception as e:
            return False, f"Error inesperado: {str(e)}"

    def retrain_model(self, rolling_windows: list, use_balancing: bool = False) -> tuple[bool, str]:
        """Solicita a la API reentrenar el modelo con nuevas configuraciones.

        Args:
            rolling_windows: Lista de ventanas rolling (ej: [3, 6])
            use_balancing: Si se debe aplicar SMOTE para balanceo de clases

        Returns:
            tuple: (success, message)
        """
        try:
            payload = {
                "rolling_windows": rolling_windows,
                "use_balancing": use_balancing,
            }

            with httpx.Client(timeout=600.0) as client:  # 10 minutos timeout
                response = client.post(f"{self.api_url}/retrain", json=payload)
                response.raise_for_status()
                result = response.json()
                return True, result.get("message", "Modelo reentrenado exitosamente")
        except httpx.HTTPError as e:
            return False, f"Error al reentrenar vía API: {str(e)}"
        except Exception as e:
            return False, f"Error inesperado: {str(e)}"

    def get_metrics(self) -> Optional[Dict]:
        """Obtiene las métricas de rendimiento desde la API.

        Returns:
            Dict con métricas de todos los modelos o None si hay error
        """
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.api_url}/metrics")
                response.raise_for_status()
                return response.json()
        except Exception:
            return None
