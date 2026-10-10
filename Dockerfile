FROM python:3.10-slim

# XGBoost needs the OpenMP runtime, which slim images lack
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY app ./app
COPY models/preprocessor.joblib models/xgb_tuned.joblib \
     models/feature_order.json models/deploy_config.json ./models/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]