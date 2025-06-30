# Deployment Guide

## Docker Deployment

```bash
make docker-build
make docker-run
```

## Kubernetes Deployment

```bash
kubectl apply -f k8s/
```

## Environment Variables

See .env.example for all required variables.
