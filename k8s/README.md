# Kubernetes Deployment for Mutual Fund Chatbot

## Build Docker Image

```bash
docker build -t mutual-fund-chatbot:latest .
```

## Push Docker Image to Registry

Tag and push the image to your container registry (Docker Hub, AWS ECR, GCR, etc.)

```bash
docker tag mutual-fund-chatbot:latest your-registry/mutual-fund-chatbot:latest
docker push your-registry/mutual-fund-chatbot:latest
```

Update the image field in `deployment.yaml` accordingly.

## Deploy to Kubernetes

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

## Verify Deployment

```bash
kubectl get pods
kubectl get svc
```

## Access the Service

If using a cloud provider, get the external IP from:

```bash
kubectl get svc mutual-fund-chatbot-service
```

Then access the chatbot API at `http://<EXTERNAL-IP>/ask`

## Notes

- Adjust resource limits and replicas in `deployment.yaml` as needed.
- Add secrets or config maps for environment variables if required.
- Ensure your Kubernetes cluster has access to the container registry.
