"""
API REST para predicción de demanda.

Este módulo implementa una API REST con FastAPI que expone los modelos
de Machine Learning para realizar predicciones de demanda.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, create_model
from typing import Dict, Optional, List
import joblib
import numpy as np
import pandas as pd
import os
import json
from contextlib import asynccontextmanager

# Configuración de directorios
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


class ModelState:
    """Almacena el estado de los modelos cargados."""

    model = None
    features: Optional[List[str]] = None
    scaler = None
    category_prices: Optional[Dict] = None
    metrics: Optional[List[Dict]] = None
    rolling_windows: Optional[List[int]] = None
    PredictionInputDynamic: Optional[type] = None  # Schema dinámico


def create_prediction_input_schema(rolling_windows: List[int]) -> type[BaseModel]:
    """Crea un schema de Pydantic dinámico basado en las ventanas rolling configuradas.

    Args:
        rolling_windows: Lista de ventanas temporales (ej: [3, 6] o [2, 4, 8])

    Returns:
        Clase BaseModel de Pydantic con campos dinámicos
    """
    # Campos base (siempre presentes)
    base_fields = {
        "shop_cluster": (int, Field(..., ge=0, le=2, description="Cluster de tienda (0-2)")),
        "item_category_id": (int, Field(..., ge=0, description="ID de categoría del producto")),
        "item_price": (float, Field(..., gt=0, description="Precio del producto")),
        "item_cnt_lag_1": (float, Field(..., ge=0, description="Ventas del mes anterior")),
        "item_cnt_lag_2": (float, Field(..., ge=0, description="Ventas de hace 2 meses")),
        "item_cnt_lag_3": (float, Field(..., ge=0, description="Ventas de hace 3 meses")),
    }

    # Generar campos dinámicos para cada ventana rolling
    for window in rolling_windows:
        base_fields[f"rolling_mean_{window}"] = (
            Optional[float],
            Field(None, ge=0, description=f"Media móvil {window} meses"),
        )
        base_fields[f"rolling_std_{window}"] = (
            Optional[float],
            Field(None, ge=0, description=f"Desv. estándar {window} meses"),
        )

    # Crear ejemplo dinámico
    example_dict = {
        "shop_cluster": 1,
        "item_category_id": 40,
        "item_price": 1500.0,
        "item_cnt_lag_1": 5.0,
        "item_cnt_lag_2": 4.5,
        "item_cnt_lag_3": 6.0,
    }

    # Agregar ejemplos de rolling features
    for window in rolling_windows:
        example_dict[f"rolling_mean_{window}"] = 5.0
        example_dict[f"rolling_std_{window}"] = 0.8

    # Crear el modelo dinámicamente
    DynamicModel = create_model("PredictionInput", **base_fields)

    # Agregar ejemplo al schema
    DynamicModel.model_config = {"json_schema_extra": {"example": example_dict}}

    return DynamicModel


class PredictionOutput(BaseModel):
    """Schema de salida para predicciones."""

    prediction: float = Field(..., description="Demanda predicha")
    prediction_log: float = Field(..., description="Predicción en escala logarítmica")
    input_features: Dict = Field(..., description="Features usadas en la predicción")
    model_info: Dict = Field(..., description="Información del modelo")


class RetrainRequest(BaseModel):
    """Schema de entrada para reentrenamiento."""

    rolling_windows: List[int] = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Lista con EXACTAMENTE 2 ventanas rolling (ej: [3, 6])",
        example=[3, 6],
    )
    use_balancing: bool = Field(
        default=False, description="Si se debe aplicar balanceo de clases con SMOTE"
    )

    @classmethod
    def model_validate(cls, value):
        """Validación personalizada para rolling_windows."""
        instance = super().model_validate(value)

        # Validar que sean exactamente 2 elementos
        if len(instance.rolling_windows) != 2:
            raise ValueError(
                f"rolling_windows debe tener EXACTAMENTE 2 elementos. "
                f"Recibido: {len(instance.rolling_windows)}"
            )

        # Validar que sean enteros positivos
        if not all(isinstance(w, int) and w > 0 for w in instance.rolling_windows):
            raise ValueError("Todos los elementos deben ser enteros positivos")

        # Validar orden correcto
        if instance.rolling_windows[0] >= instance.rolling_windows[1]:
            raise ValueError(
                f"La primera ventana ({instance.rolling_windows[0]}) debe ser menor que "
                f"la segunda ({instance.rolling_windows[1]})"
            )

        return instance


class RetrainResponse(BaseModel):
    """Schema de respuesta para reentrenamiento."""

    status: str
    message: str
    rolling_windows: List[int]
    metrics: Optional[List[Dict]] = None


class HealthResponse(BaseModel):
    """Schema de respuesta para health check."""

    status: str
    models_loaded: bool
    available_endpoints: List[str]
    model_metrics: Optional[Dict] = None


@asynccontextmanager
async def lifespan(_app: FastAPI):  # pylint: disable=unused-argument
    """Maneja el ciclo de vida de la aplicación (startup/shutdown)."""
    # Startup: cargar modelos
    print("🚀 Iniciando API...")
    load_models()
    yield
    # Shutdown: liberar recursos
    print("👋 Cerrando API...")


app = FastAPI(
    title="API de Predicción de Demanda",
    description="API REST para predicción de demanda usando Machine Learning y Deep Learning",
    version="1.0.0",
    lifespan=lifespan,
)

# Configurar CORS para permitir requests del frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios exactos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_models():
    """Carga todos los modelos y metadatos al iniciar la API."""
    try:
        print(f"📂 Cargando modelos desde: {MODELS_DIR}")

        # Cargar modelo principal
        model_path = os.path.join(MODELS_DIR, "stacking_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

        ModelState.model = joblib.load(model_path)
        print("✅ Modelo principal cargado (Stacking Ensemble)")

        # Cargar features
        ModelState.features = joblib.load(os.path.join(MODELS_DIR, "features.pkl"))
        print(f"✅ Features cargadas: {len(ModelState.features) if ModelState.features else 0}")

        # Cargar configuración de rolling windows
        rolling_path = os.path.join(MODELS_DIR, "rolling_windows.pkl")
        if os.path.exists(rolling_path):
            ModelState.rolling_windows = joblib.load(rolling_path)
            print(f"✅ Rolling windows cargadas: {ModelState.rolling_windows}")
        else:
            # Fallback a ventanas por defecto
            ModelState.rolling_windows = [3, 6]
            print("⚠️ Usando rolling windows por defecto: [3, 6]")

        # Crear schema dinámico de PredictionInput basado en rolling_windows
        ModelState.PredictionInputDynamic = create_prediction_input_schema(
            ModelState.rolling_windows or [3, 6]
        )
        print(
            f"✅ Schema de entrada creado dinámicamente para ventanas: {ModelState.rolling_windows}"
        )

        # Cargar scaler (opcional, puede no existir en versiones antiguas)
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        if os.path.exists(scaler_path):
            ModelState.scaler = joblib.load(scaler_path)
            print("✅ Scaler cargado")

        # Cargar precios por categoría
        ModelState.category_prices = joblib.load(os.path.join(MODELS_DIR, "category_prices.pkl"))
        print(
            f"✅ Precios de categorías cargados: {len(ModelState.category_prices) if ModelState.category_prices else 0}"
        )

        # Cargar métricas
        metrics_path = os.path.join(MODELS_DIR, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                ModelState.metrics = json.load(f)
            print("✅ Métricas cargadas")

        print("🎉 Todos los modelos cargados exitosamente\n")

    except Exception as e:
        print(f"❌ Error al cargar modelos: {e}")
        raise


def calculate_rolling_features(
    input_data: BaseModel, custom_windows: Optional[List[int]] = None
) -> Dict:
    """Calcula features de rolling window si no están presentes.

    Usa la configuración de rolling_windows del request o la cargada del modelo.

    Args:
        input_data: Instancia del schema dinámico PredictionInput
        custom_windows: Ventanas rolling personalizadas (opcional)

    Returns:
        Diccionario con todos los rolling features necesarios
    """
    # Obtener ventanas configuradas (priorizar custom_windows si se proporciona)
    windows = (
        custom_windows if custom_windows is not None else (ModelState.rolling_windows or [3, 6])
    )

    # Lags disponibles - usar getattr con valores por defecto
    lags = [
        getattr(input_data, "item_cnt_lag_1", 0),
        getattr(input_data, "item_cnt_lag_2", 0),
        getattr(input_data, "item_cnt_lag_3", 0),
    ]

    # Calcular features para cada ventana
    rolling_features = {}
    for window in windows:
        # Verificar si ya está presente en input_data
        mean_key = f"rolling_mean_{window}"
        std_key = f"rolling_std_{window}"

        # Si ya están presentes, usarlos
        if hasattr(input_data, mean_key) and getattr(input_data, mean_key) is not None:
            rolling_features[mean_key] = getattr(input_data, mean_key)
            rolling_features[std_key] = getattr(input_data, std_key, 0.0)
        else:
            # Calcular aproximaciones basadas en lags disponibles
            rolling_features[mean_key] = np.mean(lags)
            rolling_features[std_key] = np.std(lags)

    return rolling_features


def calculate_pricing_features(
    item_price: float,
    item_category_id: int,
    item_price_lag_1: float = None,
    item_cnt_lag_1: float = 0.0,
    item_cnt_lag_2: float = 0.0,
    item_cnt_lag_3: float = 0.0,
) -> Dict:
    """Calcula features de pricing dinámicamente (backward compatible).

    Args:
        item_price: Precio actual del producto
        item_category_id: ID de categoría
        item_price_lag_1: Precio del mes anterior (opcional)
        item_cnt_lag_1: Ventas del mes anterior
        item_cnt_lag_2: Ventas de hace 2 meses
        item_cnt_lag_3: Ventas de hace 3 meses

    Returns:
        Diccionario con features (básicas o pricing según el modelo)
    """
    # Verificar si el modelo actual necesita pricing features
    model_features = ModelState.features or []
    has_pricing = "price_rel_category" in model_features or "item_price_log" in model_features

    if not has_pricing:
        # Modelo antiguo - retornar solo features básicas
        return {
            "item_price": item_price,
            "item_cnt_lag_1": item_cnt_lag_1,
            "item_cnt_lag_2": item_cnt_lag_2,
            "item_cnt_lag_3": item_cnt_lag_3,
        }

    # Modelo nuevo con pricing features
    pricing_features = {}

    # Precio promedio de la categoría
    category_avg = ModelState.category_prices.get(item_category_id, item_price)

    # Features de pricing
    pricing_features["price_rel_category"] = item_price / (category_avg + 1e-5)
    price_max = item_price_lag_1 if item_price_lag_1 else item_price
    pricing_features["price_discount"] = (item_price / (price_max + 1e-5)) - 1
    pricing_features["is_new_price"] = (
        1 if item_price_lag_1 and item_price != item_price_lag_1 else 0
    )
    pricing_features["price_change_pct"] = (
        (item_price - item_price_lag_1) / (item_price_lag_1 + 1e-6) if item_price_lag_1 else 0.0
    )
    pricing_features["price_change_2m_pct"] = pricing_features["price_change_pct"]
    pricing_features["revenue_potential"] = item_cnt_lag_1 * item_price
    pricing_features["price_demand_elasticity"] = 0.0

    # Transformaciones logarítmicas
    pricing_features["item_price_log"] = np.log1p(item_price)
    pricing_features["price_rel_category_log"] = np.log1p(pricing_features["price_rel_category"])
    pricing_features["revenue_potential_log"] = np.log1p(pricing_features["revenue_potential"])
    pricing_features["item_cnt_lag_1_log"] = np.log1p(item_cnt_lag_1)
    pricing_features["item_cnt_lag_2_log"] = np.log1p(item_cnt_lag_2)
    pricing_features["item_cnt_lag_3_log"] = np.log1p(item_cnt_lag_3)

    return pricing_features


@app.get("/", response_model=Dict)
async def root():
    """Endpoint raíz con información de la API."""
    return {
        "message": "API de Predicción de Demanda",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST - schema dinámico según rolling windows)",
            "schema": "/schema (GET - obtener schema de entrada dinámico)",
            "metrics": "/metrics",
            "categories": "/categories (GET - obtener todas las categorías)",
            "prices": "/prices (GET - obtener todos los precios por categoría)",
            "price_by_category": "/prices/{category_id} (GET - precio de una categoría)",
            "regenerate_datasets": "/regenerate-datasets (POST - regenerar datasets desde KaggleHub)",
            "retrain": "/retrain (POST - reentrenar modelo con nuevas configuraciones)",
            "docs": "/docs",
        },
        "rolling_windows": ModelState.rolling_windows,
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint para verificar el estado de la API."""
    models_loaded = ModelState.model is not None

    metrics_info = None
    if ModelState.metrics and len(ModelState.metrics) > 0:
        metrics_info = ModelState.metrics[0]  # type: ignore

    # Agregar info de rolling windows al metrics
    if metrics_info and ModelState.rolling_windows:
        metrics_info = {**metrics_info, "rolling_windows": ModelState.rolling_windows}

    return HealthResponse(
        status="healthy" if models_loaded else "unhealthy",
        models_loaded=models_loaded,
        available_endpoints=["/", "/health", "/predict", "/metrics", "/schema", "/docs"],
        model_metrics=metrics_info,
    )


@app.get("/schema", response_model=Dict)
async def get_prediction_schema():
    """Retorna el schema dinámico de PredictionInput basado en el modelo cargado.

    Este endpoint es útil para que los clientes sepan qué campos enviar
    según la configuración de rolling windows del modelo actual.
    """
    if ModelState.PredictionInputDynamic is None:
        raise HTTPException(status_code=503, detail="Schema no disponible. Modelos no cargados.")

    # Obtener el schema JSON de Pydantic
    schema = ModelState.PredictionInputDynamic.model_json_schema()

    rolling_windows = ModelState.rolling_windows or []

    return {
        "schema": schema,
        "rolling_windows": rolling_windows,
        "example": schema.get("example", {}),
        "required_fields": list(schema.get("required", [])),
        "optional_fields": [f"rolling_mean_{w}" for w in rolling_windows]
        + [f"rolling_std_{w}" for w in rolling_windows],
    }


@app.get("/metrics", response_model=Dict)
async def get_metrics():
    """Retorna las métricas de todos los modelos entrenados."""
    if ModelState.metrics is None:
        raise HTTPException(status_code=404, detail="Métricas no disponibles")

    return {
        "models": ModelState.metrics,
        "best_model": max(ModelState.metrics, key=lambda x: x["r2"]),
        "production_model": "Stacking Ensemble",
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(request: Request):
    """
    Realiza una predicción de demanda basada en los features de entrada.

    Retorna la predicción en escala real y logarítmica, junto con
    información sobre los features usados.

    Nota: El schema de entrada es dinámico y depende de las ventanas rolling
    configuradas en el modelo entrenado.
    """
    if ModelState.model is None:
        raise HTTPException(
            status_code=503, detail="Modelos no cargados. Ejecuta el entrenamiento primero."
        )

    if ModelState.PredictionInputDynamic is None:
        raise HTTPException(status_code=503, detail="Schema de entrada no inicializado.")

    try:
        # Leer el JSON del request
        input_data = await request.json()

        # Extraer rolling_windows personalizado si existe
        custom_rolling_windows = input_data.pop("rolling_windows", None)

        # Validar input con el schema dinámico
        validated_input = ModelState.PredictionInputDynamic(**input_data)  # type: ignore

        # Calcular rolling features (usar custom_rolling_windows si se proporciona)
        rolling_features = calculate_rolling_features(validated_input, custom_rolling_windows)

        # Calcular features de pricing
        pricing_features = calculate_pricing_features(
            item_price=validated_input.item_price,  # type: ignore
            item_category_id=validated_input.item_category_id,  # type: ignore
            item_price_lag_1=getattr(validated_input, "item_price_lag_1", validated_input.item_price),  # type: ignore
            item_cnt_lag_1=validated_input.item_cnt_lag_1,  # type: ignore
            item_cnt_lag_2=validated_input.item_cnt_lag_2,  # type: ignore
            item_cnt_lag_3=validated_input.item_cnt_lag_3,  # type: ignore
        )

        # Construir diccionario de features (orden importante para el modelo)
        features_dict = {
            "shop_cluster": validated_input.shop_cluster,  # type: ignore
            "item_category_id": validated_input.item_category_id,  # type: ignore
            **pricing_features,  # Incluir todas las features de pricing
            **rolling_features,  # Incluir rolling features
        }

        # Convertir a DataFrame con el orden correcto de features
        df = pd.DataFrame([features_dict])[ModelState.features]  # type: ignore

        # Realizar predicción (en escala logarítmica)
        pred_log = ModelState.model.predict(df)[0]

        # Invertir transformación logarítmica
        pred_real = np.expm1(pred_log)

        # Asegurar que no sea negativa
        pred_real = max(0.0, pred_real)

        return PredictionOutput(
            prediction=float(pred_real),
            prediction_log=float(pred_log),
            input_features=features_dict,
            model_info={
                "model_type": "Stacking Ensemble (Random Forest + XGBoost)",
                "features_used": len(ModelState.features) if ModelState.features else 0,
                "feature_names": ModelState.features or [],
                "rolling_windows": custom_rolling_windows or ModelState.rolling_windows,
            },
        )

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=f"Error de validación: {str(ve)}") from ve
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}") from e


@app.get("/categories")
async def get_categories():
    """Retorna el mapa completo de categorías (id -> nombre)."""
    try:
        import pandas as pd

        categories_path = os.path.join(BASE_DIR, "data", "item_categories.csv")
        if not os.path.exists(categories_path):
            raise HTTPException(status_code=404, detail="Archivo de categorías no encontrado")

        df = pd.read_csv(categories_path)
        categories_dict = dict(zip(df["item_category_id"].astype(int), df["item_category_name"]))

        return categories_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al cargar categorías: {str(e)}")


@app.get("/prices")
async def get_all_prices():
    """Retorna todos los precios promedio por categoría."""
    if ModelState.category_prices is None:
        raise HTTPException(status_code=404, detail="Precios no disponibles")

    # Convertir a dict con valores float para serialización JSON
    return {int(k): float(v) for k, v in ModelState.category_prices.items()}  # type: ignore


@app.get("/prices/{category_id}")
async def get_category_price(category_id: int):
    """Retorna el precio promedio para una categoría específica."""
    if ModelState.category_prices is None:
        raise HTTPException(status_code=404, detail="Precios no disponibles")

    if category_id not in ModelState.category_prices:  # type: ignore
        raise HTTPException(status_code=404, detail=f"Categoría {category_id} no encontrada")

    return {
        "category_id": category_id,
        "average_price": float(ModelState.category_prices[category_id]),  # type: ignore
    }


@app.post("/regenerate-datasets")
async def regenerate_datasets():
    """
    Regenera los datasets descargándolos desde KaggleHub.

    Este endpoint fuerza la descarga fresca de los datos desde KaggleHub
    y actualiza la carpeta `data/`.
    """
    try:
        from src.data_processing import force_download_datasets

        print("\n🔄 Regenerando datasets desde KaggleHub...")

        success = force_download_datasets()

        if success:
            return {"status": "success", "message": "Datasets regenerados exitosamente en `data/`"}
        else:
            raise HTTPException(
                status_code=500, detail="Error al regenerar datasets. Revisa los logs del servidor."
            )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error durante la regeneración de datasets: {str(e)}"
        )


@app.post("/retrain", response_model=RetrainResponse)
async def retrain_model(request: RetrainRequest):
    """
    Reentrena el modelo con nuevas configuraciones de rolling windows.

    Este endpoint ejecuta el proceso de entrenamiento completo con las ventanas
    rolling especificadas. El proceso puede tomar varios minutos.

    **Nota**: Esto bloqueará la API hasta que termine el entrenamiento.
    """
    try:
        # Importar train_models aquí para evitar dependencias circulares
        from src.train import train_models

        print(f"\n🔄 Iniciando reentrenamiento con rolling_windows={request.rolling_windows}...")

        # Ejecutar entrenamiento
        train_models(use_balancing=request.use_balancing, rolling_windows=request.rolling_windows)

        # Recargar modelos
        print("\n📥 Recargando modelos...")
        load_models()

        return RetrainResponse(
            status="success",
            message=f"Modelo reentrenado exitosamente con ventanas {request.rolling_windows}",
            rolling_windows=request.rolling_windows,
            metrics=ModelState.metrics,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante el reentrenamiento: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    print("🌐 Iniciando servidor API...")
    print("📖 Documentación disponible en: http://localhost:8000/docs")

    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
