#!/usr/bin/env bash
set -euo pipefail

echo "=== 1) DOCKER STACK STATUS ==="
docker compose -f docker-compose.target.yml --profile serving --profile gateway ps || true
echo
docker compose -f docker-compose.monitoring.yml ps || true
echo
docker compose -f docker-compose.streamlit.yml ps || true

echo
echo "=== 2) LISTENING PORTS ==="
ss -ltnp | grep -E ':3000|:8080|:8088|:8501|:9091' || true

echo
echo "=== 3) CAFA5 HEALTH ==="
curl -i http://localhost:8088/health
echo
curl -i http://localhost:8088/api/predict/health

echo
echo "=== 4) GRAFANA / PROMETHEUS / STREAMLIT HEADERS ==="
echo
echo "--- Grafana ---"
curl -I http://localhost:3000
echo
echo "--- Prometheus ---"
curl -I http://localhost:9091
echo
echo "--- Streamlit ---"
curl -I http://localhost:8501

echo
echo "=== 5) PROMETHEUS TARGETS SUMMARY ==="
curl -s http://localhost:9091/api/v1/targets | python - << 'PY'
import json, sys
data = json.load(sys.stdin)
targets = data["data"]["activeTargets"]
for t in targets:
    print(f'{t["labels"]["job"]:15s} | {t["labels"]["instance"]:55s} | {t["health"]}')
PY

echo
echo "=== 6) REAL PREDICTION TEST ==="
cat << 'JSON' > /tmp/cafa5_predict_payload.json
{
  "sequence": "MSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVGGVLTATQKQNVG"
}
JSON

curl -i \
  -H 'Content-Type: application/json' \
  -X POST \
  --data @/tmp/cafa5_predict_payload.json \
  http://localhost:8088/api/predict

echo
echo "=== 7) QUICK INTERPRETATION ==="
echo "Erwartet:"
echo "- CAFA5 /health -> HTTP/1.1 200 OK"
echo "- CAFA5 /api/predict/health -> HTTP/1.1 200 OK"
echo "- Grafana -> 302 /login ist OK"
echo "- Streamlit -> 200 OK"
echo "- Prometheus Targets -> alle 'up'"
echo "- Prediction POST -> kein 404/502/500"
