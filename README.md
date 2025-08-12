[![Run Makefile Targets](https://github.com/JDFPalladium/KenyaEMR_IIT/actions/workflows/main.yml/badge.svg)](https://github.com/JDFPalladium/KenyaEMR_IIT/actions/workflows/main.yml)

# KenyaEMR_IIT
 
## Docker build container
1. docker build -t kenyaemr-inference .

## Data and settings
Check that these files exist:
1. /opt/ml/iit/settings.json -- facility specific settings
2. /opt/ml/iit/locational_variables_latest.csv -- facility location variables
3. /opt/ml/iit/models/thresholds_latest.pkl -- thresholds
4. /opt/ml/iit/models/site_thresholds_latest.pkl -- site thresholds
5. /opt/ml/iit/models/ohe_latest.pkl -- model
6. /opt/ml/iit/models/mod_latest.pkl -- model
7. /opt/ml/iit/models/feature_order.pkl -- features

## Docker run 
<!-- docker run -p 8000:8000 kenyaemr-inference -->
1. docker run -v /opt/ml/iit/settings.json:/data/settings.json -v /opt/ml/iit/locational_variables_latest.csv:/data/locational_variables_latest.csv -v /opt/ml/iit/models/thresholds_latest.pkl:/models/thresholds_latest.pkl -v /opt/ml/iit/models/site_thresholds_latest.pkl:/models/site_thresholds_latest.pkl -v /opt/ml/iit/models/ohe_latest.pkl:/models/ohe_latest.pkl -v /opt/ml/iit/models/mod_latest.pkl:/models/mod_latest.pkl -v /opt/ml/iit/models/feature_order.pkl:/models/feature_order.pkl --add-host=host.docker.internal:host-gateway -p 8000:8000 kenyaemr-inference

## Or Docker Compose
#### With local rebuild
docker compose up --build
#### By pulling from docker hub
docker compose up

### Clean up docker images to save space

#### Safe cleanup
1. docker container prune
2. docker image ls
3. docker rmi "IMAGE ID"
4. docker builder prune

#### Total cleanup -- Use with caution
1. docker rmi -f $(docker images -aq)
2. docker system prune -a --volumes -f

### Example Payload
curl -X POST "http://localhost:8000/inference" -H "Content-Type: application/json" -d '{"ppk": "7E14A8034F39478149EE6A4CA37A247C631D17907C746BE0336D3D7CEC68F66F", "sc": "13074", "start_date": "2021-01-01", "end_date": "2025-01-01"}'

### Development
1. python3.12 -m venv myenv
2. source myenv/bin/activate
3. pip --version
4. python --version
5. pip install --no-cache-dir -r requirements-inference.txt
6. uvicorn src.inference.api:app --host 0.0.0.0 --port 8000

