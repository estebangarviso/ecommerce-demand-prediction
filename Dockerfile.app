FROM python:3.10-slim

# Metadatos
LABEL maintainer="Equipo 9 - ACIF104"
LABEL description="Frontend Streamlit para Sistema Predictivo de Demanda"
LABEL version="1.0.0"

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    API_URL=http://backend:8000

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de dependencias
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY app/ ./app/

# Crear directorio de configuración de Streamlit
RUN mkdir -p /root/.streamlit

# Configuración de Streamlit
RUN echo '\
    [server]\n\
    port = 8501\n\
    address = 0.0.0.0\n\
    headless = true\n\
    enableCORS = false\n\
    enableXsrfProtection = true\n\
    \n\
    [browser]\n\
    gatherUsageStats = false\n\
    \n\
    [theme]\n\
    primaryColor = "#FF4B4B"\n\
    backgroundColor = "#FFFFFF"\n\
    secondaryBackgroundColor = "#F0F2F6"\n\
    textColor = "#262730"\n\
    font = "sans serif"\n\
    ' > /root/.streamlit/config.toml

# Exponer puerto
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Comando de inicio
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]