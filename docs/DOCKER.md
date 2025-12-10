# Guía de Deployment con Docker

## 📦 Contenedores Disponibles

Este proyecto incluye 2 servicios contenedorizados:

1. **Backend API** (`backend`) - FastAPI en puerto 8000
2. **Frontend UI** (`frontend`) - Streamlit en puerto 8501

---

## 🚀 Inicio Rápido

### Opción 1: Docker Compose (Recomendado)

```bash
# 1. Construir imágenes
docker-compose build

# 2. Iniciar servicios
docker-compose up -d

# 3. Verificar logs
docker-compose logs -f

# 4. Acceder a la aplicación
# - Frontend: http://localhost:8501
# - Backend API: http://localhost:8000
# - Docs Swagger: http://localhost:8000/docs
```

### Opción 2: Docker Manual

```bash
# 1. Crear red
docker network create demand-prediction-network

# 2. Construir Backend
docker build -f Dockerfile.api -t demand-prediction-api:latest .

# 3. Construir Frontend
docker build -f Dockerfile.app -t demand-prediction-ui:latest .

# 4. Ejecutar Backend
docker run -d \
  --name demand-prediction-api \
  --network demand-prediction-network \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  demand-prediction-api:latest

# 5. Ejecutar Frontend
docker run -d \
  --name demand-prediction-ui \
  --network demand-prediction-network \
  -p 8501:8501 \
  -e API_URL=http://backend:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  demand-prediction-ui:latest
```

---

## 🛠️ Comandos Útiles

### Gestión de Servicios

```bash
# Iniciar servicios
docker-compose up -d

# Detener servicios
docker-compose down

# Reiniciar servicios
docker-compose restart

# Ver logs en tiempo real
docker-compose logs -f

# Ver logs de un servicio específico
docker-compose logs -f backend
docker-compose logs -f frontend

# Ver estado de servicios
docker-compose ps
```

### Debugging

```bash
# Ejecutar bash en contenedor backend
docker-compose exec backend /bin/bash

# Ejecutar bash en contenedor frontend
docker-compose exec frontend /bin/bash

# Ver variables de entorno
docker-compose exec backend env
docker-compose exec frontend env

# Verificar salud de servicios
docker-compose exec backend curl http://localhost:8000/health
docker-compose exec frontend curl http://localhost:8501/_stcore/health
```

### Reconstruir Imágenes

```bash
# Reconstruir sin caché
docker-compose build --no-cache

# Reconstruir solo un servicio
docker-compose build backend
docker-compose build frontend

# Reconstruir y reiniciar
docker-compose up -d --build
```

### Limpieza

```bash
# Detener y eliminar contenedores
docker-compose down

# Eliminar volúmenes también
docker-compose down -v

# Eliminar imágenes
docker rmi demand-prediction-api:latest
docker rmi demand-prediction-ui:latest

# Limpieza completa del sistema Docker
docker system prune -a --volumes
```

---

## 📊 Health Checks

Los contenedores incluyen health checks automáticos:

- **Backend**: Verifica `/health` cada 30s
- **Frontend**: Verifica `/_stcore/health` cada 30s

Ver estado de salud:

```bash
docker ps
# CONTAINER ID   IMAGE                STATUS
# abc123         backend:latest       Up (healthy)
# def456         frontend:latest      Up (healthy)
```

---

## 🔧 Configuración Avanzada

### Variables de Entorno

Crear archivo `.env` en la raíz:

```env
# Backend
MODELS_DIR=/app/models
DATA_DIR=/app/data
WORKERS=2
LOG_LEVEL=info

# Frontend
API_URL=http://backend:8000
STREAMLIT_THEME=light
```

Usar en `docker-compose.yml`:

```yaml
services:
  backend:
    env_file:
      - .env
```

### Escalado de Workers

```bash
# Aumentar workers del backend
docker-compose up -d --scale backend=3
```

### Volúmenes Personalizados

```yaml
volumes:
  - /ruta/local/models:/app/models
  - /ruta/local/data:/app/data
  - /ruta/local/logs:/app/logs
```

---

## 🚀 Deployment en Producción

### Consideraciones

1. **Seguridad:**
   - Usar HTTPS con certificados SSL
   - Configurar firewall (solo puertos 8000, 8501)
   - Deshabilitar modo debug

2. **Performance:**
   - Aumentar workers: `--workers 4`
   - Configurar recursos:
     ```yaml
     deploy:
       resources:
         limits:
           cpus: '2'
           memory: 4G
         reservations:
           cpus: '1'
           memory: 2G
     ```

3. **Monitoreo:**
   - Logs centralizados (ELK Stack)
   - Métricas con Prometheus
   - Alertas con Grafana

### Deployment en AWS ECS

```bash
# 1. Subir imágenes a ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com

docker tag demand-prediction-api:latest <account>.dkr.ecr.us-east-1.amazonaws.com/demand-api:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/demand-api:latest

# 2. Crear Task Definition y Service en ECS
```

### Deployment en Google Cloud Run

```bash
# 1. Construir y subir
gcloud builds submit --tag gcr.io/<project>/demand-api
gcloud builds submit --tag gcr.io/<project>/demand-ui

# 2. Deploy
gcloud run deploy demand-api --image gcr.io/<project>/demand-api --platform managed --region us-central1
gcloud run deploy demand-ui --image gcr.io/<project>/demand-ui --platform managed --region us-central1
```

---

## 🐛 Troubleshooting

### Backend no responde

```bash
# Verificar logs
docker-compose logs backend

# Verificar salud
docker-compose exec backend curl http://localhost:8000/health

# Reiniciar
docker-compose restart backend
```

### Frontend no conecta con Backend

```bash
# Verificar variable de entorno
docker-compose exec frontend env | grep API_URL

# Verificar red
docker network inspect demand-prediction-network

# Probar conexión desde frontend
docker-compose exec frontend curl http://backend:8000/health
```

### Modelos no cargados

```bash
# Verificar volúmenes
docker-compose exec backend ls -la /app/models

# Re-montar volumen
docker-compose down
docker-compose up -d
```

### Puerto ya en uso

```bash
# Cambiar puertos en docker-compose.yml
ports:
  - "8001:8000"  # Backend
  - "8502:8501"  # Frontend
```

---

## 📚 Referencias

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
- [FastAPI Docker Guide](https://fastapi.tiangolo.com/deployment/docker/)
- [Streamlit Docker Guide](https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker)

---

**Desarrollado por:** Equipo 9 - ACIF104  
**Universidad:** Andrés Bello  
**Año:** 2025