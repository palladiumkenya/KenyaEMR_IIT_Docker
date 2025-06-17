FROM python:3.9-slim

WORKDIR /app
ENV PYTHONPATH=/app

# Copy only inference-related files
COPY pipelines /app/pipelines/
COPY models/feature_order.pkl /app/models/feature_order.pkl
COPY models/mod_latest.json /app/models/mod_latest.json
COPY models/ohe_latest.pkl /app/models/ohe_latest.pkl
COPY models/threshold_latest.pkl /app/models/threshold_latest.pkl
COPY src /app/src/
COPY data /app/data/
COPY requirements.txt /app/

# Install only the dependencies needed for inference
RUN pip install --no-cache-dir -r requirements.txt

# Set the entrypoint (adjust as needed)
CMD ["uvicorn", "src.inference.api:app", "--host", "0.0.0.0", "--port", "8000"]

