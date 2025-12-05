"""Módulo de vistas de la aplicación."""

from .prediction_view import PredictionView
from .monitoring_view import MonitoringView
from .architecture_view import ArchitectureView

__all__ = [
    "PredictionView",
    "MonitoringView",
    "ArchitectureView",
]
