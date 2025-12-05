"""
Módulo de procesamiento de datos y entrenamiento del modelo.

Este módulo contiene:
- data_processing: Pipeline de ETL y feature engineering
- train: Entrenamiento del modelo ensemble
- inference: Carga del modelo y predicciones
"""

from src import data_processing
from src import inference
from src import train

__all__ = [
    "data_processing",
    "inference",
    "train",
]
