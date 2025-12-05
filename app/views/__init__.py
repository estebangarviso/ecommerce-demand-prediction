"""Módulo de vistas de la aplicación."""

from .prediction_view import PredictionView
from .monitoring_view import MonitoringView
from .about_view import AboutView

__all__ = [
    "PredictionView",
    "MonitoringView",
    "AboutView",
]
