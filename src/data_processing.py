import pandas as pd
import numpy as np
import kagglehub
import os
import shutil
from sklearn.cluster import KMeans
from typing import Tuple

# Configuración de directorios
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Archivos requeridos del dataset
REQUIRED_FILES = ["sales_train.csv", "items.csv", "shops.csv", "item_categories.csv"]


def get_data_path() -> str:
    """
    Obtiene la ruta de los datos con sistema de respaldo:
    1. Intenta usar archivos locales en data/ si existen y son válidos
    2. Si no, descarga desde KaggleHub y los copia a data/
    3. Si KaggleHub falla, usa data/ como último recurso
    """
    # Verificar si data/ tiene todos los archivos
    local_files_exist = all(os.path.exists(os.path.join(DATA_DIR, f)) for f in REQUIRED_FILES)

    if local_files_exist:
        print("✅ Usando archivos locales desde data/")
        return DATA_DIR

    # Intentar descargar desde KaggleHub
    try:
        print("⏳ Descargando dataset desde KaggleHub...")
        kaggle_path = kagglehub.dataset_download(
            "jaklinmalkoc/predict-future-sales-retail-dataset-en"
        )

        # Verificar que los archivos descargados no estén vacíos
        all_valid = True
        for file in REQUIRED_FILES:
            file_path = os.path.join(kaggle_path, file)
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                all_valid = False
                break

        if not all_valid:
            raise ValueError("Archivos descargados están vacíos o incompletos")

        print(f"✅ Dataset descargado en: {kaggle_path}")

        # Copiar archivos a data/ como respaldo
        print("💾 Creando copia de respaldo en data/...")
        os.makedirs(DATA_DIR, exist_ok=True)
        for file in REQUIRED_FILES:
            src = os.path.join(kaggle_path, file)
            dst = os.path.join(DATA_DIR, file)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"   ✓ {file} respaldado")

        return kaggle_path

    except Exception as e:
        print(f"⚠️  Error al descargar desde KaggleHub: {e}")

        # Verificar si hay respaldo local
        if local_files_exist:
            print("✅ Usando archivos de respaldo desde data/")
            return DATA_DIR
        else:
            raise FileNotFoundError(
                "No se pudo descargar desde KaggleHub y no hay archivos de respaldo en data/. "
                "Por favor, descarga manualmente los archivos y colócalos en la carpeta data/"
            )


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga los datasets con sistema de respaldo automático.
    Intenta KaggleHub primero, si falla usa data/ local.
    """
    path = get_data_path()

    # Cargar archivos
    print("📂 Cargando archivos CSV...")
    sales = pd.read_csv(os.path.join(path, "sales_train.csv"))
    items = pd.read_csv(os.path.join(path, "items.csv"))
    shops = pd.read_csv(os.path.join(path, "shops.csv"))
    cats = pd.read_csv(os.path.join(path, "item_categories.csv"))

    # Validar que no estén vacíos
    if any(df.empty for df in [sales, items, shops, cats]):
        raise ValueError("Uno o más datasets están vacíos")

    print(f"✅ Datos cargados exitosamente ({len(sales):,} registros de ventas)")
    return sales, items, shops, cats


def force_download_datasets() -> bool:
    """
    Fuerza la descarga de datasets desde KaggleHub y los guarda en data/.
    Elimina archivos existentes para garantizar datos frescos.

    Returns:
        True si la descarga fue exitosa, False en caso contrario
    """
    try:
        print("🔄 Eliminando archivos antiguos de data/...")
        # Eliminar archivos antiguos
        for file in REQUIRED_FILES:
            file_path = os.path.join(DATA_DIR, file)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"   ✓ {file} eliminado")

        print("⏳ Descargando dataset fresco desde KaggleHub...")
        kaggle_path = kagglehub.dataset_download(
            "jaklinmalkoc/predict-future-sales-retail-dataset-en"
        )

        # Verificar que los archivos descargados sean válidos
        all_valid = True
        for file in REQUIRED_FILES:
            file_path = os.path.join(kaggle_path, file)
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                all_valid = False
                break

        if not all_valid:
            raise ValueError("Archivos descargados están vacíos o incompletos")

        print(f"✅ Dataset descargado en: {kaggle_path}")

        # Copiar archivos a data/
        print("💾 Guardando en data/...")
        os.makedirs(DATA_DIR, exist_ok=True)
        for file in REQUIRED_FILES:
            src = os.path.join(kaggle_path, file)
            dst = os.path.join(DATA_DIR, file)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"   ✓ {file} guardado")

        print("✅ Datasets regenerados exitosamente en data/")
        return True

    except Exception as e:
        print(f"❌ Error al regenerar datasets: {e}")
        return False


def clean_data(sales: pd.DataFrame) -> pd.DataFrame:
    """Limpieza básica y tratamiento de outliers (Clipping)."""
    # Eliminar precios negativos o cero
    sales = sales[sales["item_price"] > 0]

    # Clipping: Limitar ventas extremas (Balanceo de datos) para evitar sesgos
    sales["item_cnt_day"] = sales["item_cnt_day"].clip(0, 20)
    sales["item_price"] = sales["item_price"].clip(0, 300000)

    # Convertir fecha
    if "date" in sales.columns:
        sales["date"] = pd.to_datetime(sales["date"], dayfirst=True)

    return sales


def generate_clusters(
    shops: pd.DataFrame, sales: pd.DataFrame, n_clusters: int = 3
) -> pd.DataFrame:
    """
    APRENDIZAJE NO SUPERVISADO (Clustering).
    Agrupa tiendas según su volumen de ventas para usarlo como feature.
    """
    # Agrupar ventas totales por tienda
    shop_sales = sales.groupby("shop_id")["item_cnt_day"].sum().reset_index()

    # K-Means para agrupar tiendas por volumen de venta
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    shop_sales["shop_cluster"] = kmeans.fit_predict(shop_sales[["item_cnt_day"]])

    return shop_sales[["shop_id", "shop_cluster"]]


def feature_engineering(
    sales: pd.DataFrame, items: pd.DataFrame, shops_clusters: pd.DataFrame
) -> pd.DataFrame:
    """Genera la matriz de entrenamiento con Lags (Variables temporales)."""
    # Agrupar por mes (date_block_num), tienda e item
    monthly_sales = (
        sales.groupby(["date_block_num", "shop_id", "item_id"])
        .agg({"item_cnt_day": "sum", "item_price": "mean"})  # Precio promedio del mes
        .reset_index()
    )

    # Clip de ventas mensuales (Target range 0-20)
    monthly_sales["item_cnt_day"] = monthly_sales["item_cnt_day"].clip(0, 20)

    # Unir con clusters y categorías
    data = monthly_sales.merge(shops_clusters, on="shop_id", how="left")
    data = data.merge(items[["item_id", "item_category_id"]], on="item_id", how="left")

    # Generar Lags (Rezagos: t-1, t-2, t-3)
    data_shifted = data.copy()

    for lag in [1, 2, 3]:
        shifted = data_shifted[["date_block_num", "shop_id", "item_id", "item_cnt_day"]].copy()
        shifted.columns = ["date_block_num", "shop_id", "item_id", f"item_cnt_lag_{lag}"]
        shifted["date_block_num"] += lag
        data = data.merge(shifted, on=["date_block_num", "shop_id", "item_id"], how="left")

    # Llenar NaNs generados por los lags con 0 (meses iniciales)
    data = data.fillna(0)

    return data


def prepare_full_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ejecuta todo el pipeline y retorna datos listos para Train/Val/Test."""
    sales, items, shops, cats = load_data()

    print("🧹 Limpiando datos...")
    sales = clean_data(sales)

    print("🤖 Generando Clusters (K-Means)...")
    shops_clusters = generate_clusters(shops, sales)

    print("⚙️ Ingeniería de Características (Lags)...")
    df_final = feature_engineering(sales, items, shops_clusters)

    # Transformación Logarítmica del Target (Técnica de Balanceo #2)
    # log1p calcula log(1 + x) para manejar ceros correctamente
    df_final["target_log"] = np.log1p(df_final["item_cnt_day"])

    # Separar Train/Val/Test (Simulación temporal)
    # Usamos los últimos meses para validación y prueba para respetar la serie temporal
    max_month = df_final["date_block_num"].max()

    train = df_final[df_final["date_block_num"] < max_month - 1]
    val = df_final[df_final["date_block_num"] == max_month - 1]
    test = df_final[df_final["date_block_num"] == max_month]

    print(f"📊 Dataset listo: Train ({len(train)}), Val ({len(val)}), Test ({len(test)})")
    return train, val, test
