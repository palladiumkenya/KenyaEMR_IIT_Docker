[![Run Makefile Targets](https://github.com/JDFPalladium/KenyaEMR_IIT/actions/workflows/main.yml/badge.svg)](https://github.com/JDFPalladium/KenyaEMR_IIT/actions/workflows/main.yml)

# KenyaEMR_IIT
 
## Docker build container
docker build -t kenyaemr-inference .

## Data and settings
Check that these files exist:
1. /opt/ml/iit/settings.json -- facility specific settings
2. /opt/ml/iit/locational_variables_latest.csv -- facility location variables

## Docker run 
# docker run -p 8000:8000 kenyaemr-inference
docker run -v /opt/ml/iit/settings.json:/data/settings.json -v /opt/ml/iit/locational_variables_latest.csv:/data/locational_variables_latest.csv -p 8000:8000 kenyaemr-inference

### Clean up docker images to save space
docker container prune
docker rmi "IMAGE ID"
docker builder prune

### Example Payload
curl -X POST "http://localhost:8000/inference" -H "Content-Type: application/json" -d '{"ppk": "7E14A8034F39478149EE6A4CA37A247C631D17907C746BE0336D3D7CEC68F66F", "sc": "13074", "start_date": "2021-01-01", "end_date": "2025-01-01"}'
