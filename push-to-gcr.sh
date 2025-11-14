#!/bin/bash
# Script to build and push Clinic-AI backend to Google Container Registry

set -e  # Exit on any error

# GCP Configuration
PROJECT_ID="clinic-ai-472907"
IMAGE_NAME="clinic-ai-backend"
TAG="${1:-latest}"  # Use first argument as tag, default to 'latest'
GCR_IMAGE="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}"

echo "🚀 Clinic-AI Backend - GCR Push Script"
echo "========================================"
echo "Project ID: ${PROJECT_ID}"
echo "Image Name: ${IMAGE_NAME}"
echo "Tag: ${TAG}"
echo "Full Image: ${GCR_IMAGE}"
echo ""

# Step 1: Check if gcloud is installed
echo "Step 1: Checking gcloud CLI..."
if ! command -v gcloud &> /dev/null; then
    echo "❌ Error: gcloud CLI is not installed"
    echo "   Install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi
echo "✅ gcloud CLI found: $(gcloud --version | head -n1)"
echo ""

# Step 2: Check if docker is installed
echo "Step 2: Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed"
    echo "   Install it from: https://docs.docker.com/get-docker/"
    exit 1
fi
echo "✅ Docker found: $(docker --version)"
echo ""

# Step 3: Set GCP project
echo "Step 3: Setting GCP project..."
gcloud config set project ${PROJECT_ID}
echo "✅ Project set to: ${PROJECT_ID}"
echo ""

# Step 4: Authenticate with GCR
echo "Step 4: Authenticating with Google Container Registry..."
gcloud auth configure-docker --quiet
echo "✅ Authentication configured"
echo ""

# Step 5: Build Docker image
echo "Step 5: Building Docker image..."
echo "   Building from: $(pwd)"
echo "   Dockerfile: docker/Dockerfile"
docker build -f docker/Dockerfile -t ${IMAGE_NAME}:${TAG} .
echo "✅ Docker image built: ${IMAGE_NAME}:${TAG}"
echo ""

# Step 6: Tag image for GCR
echo "Step 6: Tagging image for GCR..."
docker tag ${IMAGE_NAME}:${TAG} ${GCR_IMAGE}
echo "✅ Image tagged: ${GCR_IMAGE}"
echo ""

# Step 7: Push to GCR
echo "Step 7: Pushing image to GCR..."
echo "   This may take a few minutes..."
docker push ${GCR_IMAGE}
echo ""

# Success message
echo "========================================"
echo "✅ SUCCESS! Image pushed to GCR"
echo "========================================"
echo "Image URL: ${GCR_IMAGE}"
echo ""
echo "To pull this image, use:"
echo "  docker pull ${GCR_IMAGE}"
echo ""
echo "To deploy to Cloud Run, use:"
echo "  gcloud run deploy clinic-ai-backend \\"
echo "    --image ${GCR_IMAGE} \\"
echo "    --platform managed \\"
echo "    --region us-central1"
echo ""

