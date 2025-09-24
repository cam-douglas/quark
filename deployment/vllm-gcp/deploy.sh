#!/bin/bash
# Deploy vLLM to Google Cloud Platform

set -e

PROJECT_ID="quark-469604"
REGION="us-central1"
SERVICE_NAME="quark-vllm-brain"

echo "🚀 Deploying vLLM to Google Cloud..."

# Build and push Docker image
echo "📦 Building Docker image..."
docker build -t gcr.io/${PROJECT_ID}/${SERVICE_NAME} .
docker push gcr.io/${PROJECT_ID}/${SERVICE_NAME}

# Deploy to Cloud Run
echo "☁️ Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image gcr.io/${PROJECT_ID}/${SERVICE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 8Gi \
    --cpu 4 \
    --timeout 3600 \
    --set-env-vars HF_TOKEN=${HF_TOKEN}

echo "✅ Deployment complete!"
echo "🔗 Service URL: $(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format 'value(status.url)')"
