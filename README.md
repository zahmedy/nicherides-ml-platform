# NicheRides's Machine Learning Platoform 
Data processing, ML training, model evaluation, model registry, inference service, monitoring, retraining

For start:

1. Price Predication feature

- Raw data -> clean -> validate -> feature engineer 
            -> train model -> evaluate model -> save versioned model 
            -> serve model 
            -> log predication


Pipelines: 

    - Data Preprocessing pipeline:
        Raw data -> clean -> validate -> feature engineer
    - Training Pipeline
        Train model -> evaluate model -> save versioned model


Features storage
- 

## MLflow with a Small Lightsail VM

Do not run the MLflow UI/server on the Lightsail instance if the VM is small.
Run only a lightweight Postgres container there for MLflow metadata, then run
the MLflow UI locally on your laptop.

On the Lightsail instance:

```bash
cp .env.example .env
docker compose up -d mlflow-db
docker compose logs -f mlflow-db
```

Keep Postgres bound to `127.0.0.1` on the VM and connect through SSH:

```bash
ssh -N -L 15432:127.0.0.1:15432 ubuntu@YOUR_LIGHTSAIL_IP
```

On your laptop, install the local MLflow extras:

```bash
pip install -e ".[mlflow]"
```

Run the UI locally:

```bash
mlflow ui \
  --backend-store-uri postgresql+psycopg2://mlflow:change-me@127.0.0.1:15432/mlflow \
  --default-artifact-root file://$PWD/mlartifacts \
  --host 127.0.0.1 \
  --port 5000 \
  --workers 1
```

Then open:

```text
http://localhost:5000
```

Use a real S3-compatible artifact root when runs need to be shared across
machines:

```bash
mlflow ui \
  --backend-store-uri postgresql+psycopg2://mlflow:change-me@127.0.0.1:15432/mlflow \
  --default-artifact-root s3://YOUR_BUCKET/mlflow-artifacts \
  --host 127.0.0.1 \
  --port 5000 \
  --workers 1
```

For local training while the UI is running:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

## Inference API

Run the API container locally:

```bash
docker compose up -d inference-api
```

The API is exposed on:

```text
http://localhost:8001
```

If another Docker container calls this API through the host, use:

```text
PRICE_PREDICTION_API_URL=http://host.docker.internal:8001/v1/price/predict
VIN_SCAN_API_URL=http://host.docker.internal:8001/v1/vin/photo
```

On Linux, add this to that other container so `host.docker.internal` resolves:

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

Keep port `8001` closed in the Lightsail firewall unless this API should be
public.
