# MP165 - Auto-Rickshaw Fare (Jaipur)

**Group**: G221  
**Problem**: MP165  
**Faculty**: Upasna Joshi, Batch 21  

## Members
- **S24CSEU1292 VANSHBOORA**
- **S24CSEU0826 DevanshTyagi**
- **S24CSEU1305 AnshSinghal**
- **S24CSEU0920 kartiklakra**

---

## Project Lifecycle Stages

The project is structured into four sequential MLOps lifecycle stages:
1. **Stage 1 — Data Audit, Quality Checks & DVC Pipeline**: Exploratory anomaly detection (handling leaking target receipt columns, unit conversions, stuck sensors, typos, and categorical variants) with DVC data tracking.
2. **Stage 2 — Training & MLflow Experiment Tracking**: Scikit-learn preprocessing pipeline, model training, MLflow metrics logging, and model registry versioning.
3. **Stage 3 — Containerization, Serving & Kubernetes Deployment**: Production FastAPI service, schema validation, batch inference, Docker container hardening, Kubernetes deployment with HPA autoscaling, and automated CI/CD pipeline.
4. **Stage 4 — Drift Monitoring, Champion Promotion & Holdout Scoring**: Population Stability Index (PSI) feature drift detection, zonal subgroup error analysis, chronological champion promotion, and holdout predictions.

---

## Stage 3: Deployment, Containerization & Orchestration

Stage 3 provides the production serving infrastructure for the champion auto-rickshaw fare prediction model.

### 1. FastAPI Serving Architecture
The service (`app/main.py`) exposes high-performance asynchronous endpoints powered by FastAPI and Pydantic:
- **`GET /`**: Service metadata, stage identification, versioning, and endpoint directory.
- **`GET /health`**: Liveness and readiness probe endpoint returning operational status (`{"status": "ok"}`).
- **`GET /model-info`**: Inspection route exposing loaded champion model type, expected numeric/categorical features, and target variable.
- **`POST /predict`**: Real-time single ride fare prediction with Pydantic validation (ranges and enum constraints).
- **`POST /predict/batch`**: Vectorized batch inference endpoint supporting up to 500 ride estimations in a single request.
- **Request Timing Middleware**: Measures latency in milliseconds and injects `X-Process-Time-Ms` in HTTP response headers alongside structured JSON logging.

#### Example Single Prediction Request
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "distance_km": 6.5,
    "wait_min": 8.0,
    "passengers": 2,
    "hour": 15,
    "luggage_bags": 1,
    "fuel_price_rs": 96.5,
    "booking": "app",
    "zone": "Market"
  }'
```
**Response:**
```json
{
  "predicted_fare_rs": 142.50
}
```

#### Example Batch Prediction Request
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "rides": [
      {
        "distance_km": 3.0,
        "wait_min": 5.0,
        "passengers": 1,
        "hour": 10,
        "luggage_bags": 0,
        "fuel_price_rs": 95.0,
        "booking": "street",
        "zone": "Station Area"
      },
      {
        "distance_km": 11.2,
        "wait_min": 18.0,
        "passengers": 3,
        "hour": 19,
        "luggage_bags": 2,
        "fuel_price_rs": 98.0,
        "booking": "app",
        "zone": "Market"
      }
    ]
  }'
```

---

### 2. Docker Containerization & Compose

The container build (`Dockerfile`) is hardened according to production security standards:
- **Non-root Execution**: Runs as dedicated system user `appuser` (UID 10001).
- **Embedded Healthcheck**: Built-in container health verification using standard `urllib`.
- **Environment Tuning**: Configured with `PYTHONUNBUFFERED=1` and `PYTHONDONTWRITEBYTECODE=1` for optimal logging and memory usage.

#### Build and Run Locally:
```bash
# Build Docker image
docker build -t mp165-fare-api:1.0 .

# Run container
docker run -d -p 8000:8000 --name mp165-fare mp165-fare-api:1.0

# Using Docker Compose
docker compose up -d
```

---

### 3. Kubernetes Orchestration (`k8s/`)

The Kubernetes manifests provide production resilience, scalability, and self-healing:
- **`k8s/deployment.yaml`**: 
  - Manages 4 pod replicas with rolling updates.
  - Configures non-root security context (`runAsNonRoot: true`, `runAsUser: 10001`).
  - Strict resource governance (`requests: 100m CPU / 256Mi RAM`, `limits: 500m CPU / 512Mi RAM`).
  - Periodic HTTP readiness and liveness health probes.
- **`k8s/service.yaml`**: Exposes the application internally on port 80 routing to container port 8000.
- **`k8s/configmap.yaml`**: Externalizes operational parameters (`PORT`, `LOG_LEVEL`, `ENVIRONMENT`).
- **`k8s/hpa.yaml`**: HorizontalPodAutoscaler dynamically autoscaling from 2 to 8 replicas based on CPU (70%) and Memory (80%) utilization.

#### Apply Kubernetes Manifests:
```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

---

### 4. Automated Testing & CI/CD Pipeline

The test suite (`tests/test_api.py`) exercises:
- Health check and root metadata verification.
- Valid single and batch fare estimations.
- Schema validation rejections (negative distances, out-of-range fuel price, invalid booking channel enums, invalid zones, and hour boundaries).

#### Run Tests:
```bash
python -m pytest tests/ -v
```
GitHub Actions (`.github/workflows/ci.yml`) automatically triggers on pull requests and pushes to `main` to execute all tests and build the Docker container artifact.
