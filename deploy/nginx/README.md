# NGINX gateway

## Zweck
- einziger externer Einstiegspunkt
- User-Routen für go-prediction-api
- Admin-Routen für embedding-api

## Routing
- /health -> go-prediction-api /health
- /predict -> go-prediction-api /predict
- /api/v1/* -> embedding-api-* (admin-only)

## Admin Auth
- Basic Auth über .htpasswd-admin

## Hinweis
- mlflow bleibt intern und wird nicht über nginx veröffentlicht
