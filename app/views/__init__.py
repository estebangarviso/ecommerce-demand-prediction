"""Módulo de vistas de la aplicación."""

from .prediction_view import PredictionView
from .monitoring_view import MonitoringView
from .about_view import AboutView
from .technical_analysis_view import TechnicalAnalysisView

__all__ = [
    "PredictionView",
    "MonitoringView",
    "AboutView",
    "TechnicalAnalysisView",
]
