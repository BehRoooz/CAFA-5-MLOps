# Docker — Schnellstart

## Voraussetzungen
- Docker >= 24
- Docker Compose >= 2.20
- AMD GPU mit ROCm / NVIDIA GPU / oder CPU

## 1. Image bauen
```bash
docker compose --profile amd build
docker compose --profile nvidia build
docker compose --profile cpu build
```

## 2. Erste Checks vor dem Start
```bash
docker --version
rocm-smi
ls data/embeddings/esm2/
sudo chcon -Rt svirt_sandbox_file_t ./outputs
sudo chcon -Rt svirt_sandbox_file_t ./data
```

## 3. Pipeline ausführen
```bash
docker compose --profile amd up
```

## 4. Inference API starten
```bash
docker compose --profile amd build api
docker compose --profile amd up api
```

## 5. API testen
```bash
curl http://localhost:8000/health
firefox http://localhost:8000/docs
```

## 6. Outputs prüfen
```bash
ls outputs/label_matrix_top500/
ls outputs/checkpoints/
ls -lh outputs/submission.tsv
```

## Troubleshooting
| Problem | Lösung |
|---|---|
| Permission denied | sudo chcon -Rt svirt_sandbox_file_t ./outputs |
| GPU nicht erkannt | rocm-smi prüfen, HSA_OVERRIDE_GFX_VERSION=12.0.0 setzen |
| Port 8000 belegt | docker ps, anderen Container stoppen |
