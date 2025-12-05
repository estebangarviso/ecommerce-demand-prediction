"""Servicio para predicciones de demanda."""

from typing import Dict, Tuple, Any
import pandas as pd
import shap


class PredictionService:
    """Servicio para predicciones de demanda."""

    def __init__(self, model: Any, shap_model: Any):
        """
        Inicializa el servicio de predicciones.

        Args:
            model: Modelo de predicción
            shap_model: Modelo para análisis SHAP
        """
        self.model = model
        self.shap_model = shap_model

    def predict(self, input_data: Dict) -> float:
        """
        Realiza una predicción de demanda.

        Args:
            input_data: Datos de entrada para la predicción

        Returns:
            Valor predicho
        """
        from src.inference import predict_demand

        return predict_demand(self.model, input_data)

    def calculate_shap_values(self, input_data: Dict) -> Tuple[Any, pd.DataFrame, Any]:
        """
        Calcula los valores SHAP para la explicabilidad.

        Args:
            input_data: Datos de entrada

        Returns:
            Tupla con (shap_values, feature_dataframe, expected_value)
        """
        explainer = shap.TreeExplainer(self.shap_model)
        feat_df = pd.DataFrame([input_data])
        shap_values = explainer.shap_values(feat_df)
        return shap_values, feat_df, explainer.expected_value
