FROM python:3.9-slim
# FROM python:3.9-alpine

WORKDIR /app
ENV PYTHONPATH=/app

# Copy only inference-related files
COPY pipelines /app/pipelines/
COPY models/feature_order.pkl /app/models/feature_order.pkl
COPY models/mod_latest.json /app/models/mod_latest.json
COPY models/mod_latest.pkl /app/models/mod_latest.pkl
COPY models/ohe_latest.pkl /app/models/ohe_latest.pkl
COPY models/thresholds_latest.pkl /app/models/thresholds_latest.pkl
COPY models/site_thresholds_latest.pkl /app/models/site_thresholds_latest.pkl
COPY src /app/src/
COPY data /app/data/
COPY requirements-inference.txt /app/

# Install only the dependencies needed for inference
# RUN apk add --no-cache build-base linux-headers gfortran cmake
RUN pip install --no-cache-dir -r requirements-inference.txt

# Set the entrypoint (adjust as needed)
CMD ["uvicorn", "src.inference.api:app", "--host", "0.0.0.0", "--port", "8000"]
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

