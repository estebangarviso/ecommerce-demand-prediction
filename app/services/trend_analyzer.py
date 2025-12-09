"""Analizador de tendencias de ventas."""

from typing import Tuple


class TrendAnalyzer:
    """Analizador de tendencias de ventas."""

    @staticmethod
    def calculate_delta(prediction: float, last_value: float) -> float:
        """Calcula el delta entre la predicción y el último valor."""
        return prediction - last_value

    @staticmethod
    def get_trend_status(delta: float) -> Tuple[str, str, str]:
        """Determina el estado de la tendencia según el delta (mensaje, icono, modo)."""
        if delta > 0:
            return (
                f"Tendencia al alza (+{delta:.1f})",
                ":material/trending_up:",
                "normal",
            )
        elif delta < 0:
            return (
                f"Baja proyectada ({delta:.1f})",
                ":material/trending_down:",
                "normal",
            )
        else:
            return ("Demanda estable", ":material/trending_flat:", "off")

    @staticmethod
    def get_chart_colors(
        prediction: float,
        last_value: float,
        historical_color: str,
        positive_color: str,
        negative_color: str,
    ) -> list:
        """Retorna colores para el gráfico de tendencia basados en la predicción."""
        colors = [historical_color] * 3
        colors.append(positive_color if prediction >= last_value else negative_color)
        return colors
