"""Constructor de gráficos siguiendo el patrón Builder."""

from typing import Any
import pandas as pd
import plotly.express as px


class ChartBuilder:
    """Constructor de gráficos siguiendo el patrón Builder."""

    @staticmethod
    def create_bar_chart(
        df: pd.DataFrame, x: str, y: str, height: int = 150, show_legend: bool = False
    ) -> Any:
        """Crea un gráfico de barras básico."""
        fig = px.bar(df, x=x, y=y, height=height)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=show_legend)
        return fig

    @staticmethod
    def create_temporal_chart(
        lags_df: pd.DataFrame, colors: list, title: str = "Serie Temporal"
    ) -> Any:
        """Crea un gráfico de serie temporal con colores personalizados."""
        fig = px.bar(lags_df, x="Periodo", y="Ventas", title=title, text_auto=True)
        fig.update_traces(marker_color=colors, texttemplate="%{y:.1f}")
        return fig

    @staticmethod
    def create_scatter_chart(df: pd.DataFrame, x: str, y: str, trendline: str = "lowess") -> Any:
        """Crea un gráfico de dispersión con línea de tendencia."""
        fig = px.scatter(df, x=x, y=y, trendline=trendline)
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        return fig

    @staticmethod
    def create_histogram(data: Any, nbins: int = 10, label: str = "Error") -> Any:
        """Crea un histograma."""
        fig = px.histogram(data, nbins=nbins, labels={"value": label})
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        return fig
