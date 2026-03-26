# Docker — Quick Start

## Prerequisites
- Docker >= 24
- Docker Compose >= 2.20
- AMD GPU with ROCm / NVIDIA GPU / or CPU

## 1. Build Image
```bash
docker compose --profile amd build
docker compose --profile nvidia build
docker compose --profile cpu build
```

## 2. First Checks
```bash
docker --version
rocm-smi
ls data/embeddings/esm2/
sudo chcon -Rt svirt_sandbox_file_t ./outputs
sudo chcon -Rt svirt_sandbox_file_t ./data
```

## 3. Run Pipeline
```bash
docker compose --profile amd up
```

## 4. Start API
```bash
docker compose --profile amd build api
docker compose --profile amd up api
```

## 5. Test API
```bash
curl http://localhost:8000/health
firefox http://localhost:8000/docs
```

## 6. Check Outputs
```bash
ls outputs/label_matrix_top500/
ls outputs/checkpoints/
ls -lh outputs/submission.tsv
```

## Troubleshooting
| Problem | Solution |
|---|---|
| Permission denied | sudo chcon -Rt svirt_sandbox_file_t ./outputs |
| GPU not detected | Check rocm-smi, set HSA_OVERRIDE_GFX_VERSION=12.0.0 |
| Port 8000 in use | docker ps, stop conflicting container |
