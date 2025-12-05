"""Constructor de DataFrames para visualización."""

from typing import Any
import pandas as pd


class DataFrameBuilder:
    """Constructor de DataFrames para visualización."""

    @staticmethod
    def create_trend_dataframe(lag_3: int, lag_2: int, lag_1: int) -> pd.DataFrame:
        """Crea un DataFrame para la tendencia de entrada."""
        return pd.DataFrame({"Mes": ["t-3", "t-2", "t-1"], "Ventas": [lag_3, lag_2, lag_1]})

    @staticmethod
    def create_temporal_dataframe(
        lag_3: int, lag_2: int, lag_1: int, prediction: float
    ) -> pd.DataFrame:
        """Crea un DataFrame para la proyección temporal."""
        return pd.DataFrame(
            {
                "Periodo": ["Mes t-3", "Mes t-2", "Mes t-1", "Predicción (t)"],
                "Ventas": [lag_3, lag_2, lag_1, prediction],
                "Tipo": ["Histórico", "Histórico", "Histórico", "Predicción"],
            }
        )

    @staticmethod
    def create_monitoring_dataframe(dates: pd.DatetimeIndex, residuals: Any) -> pd.DataFrame:
        """Crea un DataFrame para el monitoreo de errores."""
        return pd.DataFrame({"Fecha": dates, "Error Residual": residuals})
