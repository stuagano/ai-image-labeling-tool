#!/bin/bash

# AI Image Labeler - Cloud Run Deployment Script
# This script helps deploy the AI image labeling tool to Google Cloud Run

set -e

# Configuration
PROJECT_ID=${PROJECT_ID:-$(gcloud config get-value project)}
REGION=${REGION:-us-central1}
SERVICE_NAME=${SERVICE_NAME:-ai-image-labeler}
IMAGE_NAME=${IMAGE_NAME:-gcr.io/$PROJECT_ID/$SERVICE_NAME}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if ! command_exists gcloud; then
        print_error "gcloud CLI is not installed. Please install it first."
        exit 1
    fi
    
    if ! command_exists docker; then
        print_error "Docker is not installed. Please install it first."
        exit 1
    fi
    
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        print_error "Not authenticated with gcloud. Please run 'gcloud auth login' first."
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Function to build and push Docker image
build_and_push() {
    local tag=$1
    print_status "Building Docker image with tag: $tag"
    
    docker build -t $IMAGE_NAME:$tag .
    docker push $IMAGE_NAME:$tag
    
    print_success "Image built and pushed successfully"
}

# Function to deploy to Cloud Run
deploy_to_cloud_run() {
    local image_tag=$1
    local memory=${2:-2Gi}
    local cpu=${3:-2}
    local max_instances=${4:-10}
    
    print_status "Deploying to Cloud Run..."
    print_status "Service: $SERVICE_NAME"
    print_status "Region: $REGION"
    print_status "Memory: $memory"
    print_status "CPU: $cpu"
    print_status "Max instances: $max_instances"
    
    gcloud run deploy $SERVICE_NAME \
        --image $IMAGE_NAME:$image_tag \
        --region $REGION \
        --platform managed \
        --allow-unauthenticated \
        --memory $memory \
        --cpu $cpu \
        --max-instances $max_instances \
        --timeout 300 \
        --concurrency 80 \
        --set-env-vars STREAMLIT_SERVER_PORT=8080,STREAMLIT_SERVER_ADDRESS=0.0.0.0,STREAMLIT_SERVER_HEADLESS=true
    
    print_success "Deployment completed successfully"
    
    # Get the service URL
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")
    print_success "Service URL: $SERVICE_URL"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  build [TAG]           Build and push Docker image (default tag: latest)"
    echo "  deploy [TAG]          Deploy to Cloud Run (default tag: latest)"
    echo "  build-and-deploy      Build and deploy in one step"
    echo "  setup                 Initial setup (enable APIs, set up permissions)"
    echo "  logs                  Show Cloud Run logs"
    echo "  delete                Delete Cloud Run service"
    echo "  help                  Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  PROJECT_ID            Google Cloud project ID"
    echo "  REGION                Cloud Run region (default: us-central1)"
    echo "  SERVICE_NAME          Cloud Run service name (default: ai-image-labeler)"
    echo "  GOOGLE_API_KEY        Google API key for Gemini (optional)"
    echo ""
    echo "Examples:"
    echo "  $0 build-and-deploy"
    echo "  $0 deploy v1.0.0"
    echo "  PROJECT_ID=my-project $0 setup"
}

# Function to setup initial configuration
setup() {
    print_status "Setting up initial configuration..."
    
    # Enable required APIs
    print_status "Enabling required APIs..."
    gcloud services enable cloudbuild.googleapis.com
    gcloud services enable run.googleapis.com
    gcloud services enable containerregistry.googleapis.com
    
    # Set up permissions
    print_status "Setting up permissions..."
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:$PROJECT_ID@appspot.gserviceaccount.com" \
        --role="roles/run.admin"
    
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:$PROJECT_ID@appspot.gserviceaccount.com" \
        --role="roles/storage.admin"
    
    print_success "Setup completed successfully"
}

# Function to show logs
show_logs() {
    print_status "Showing Cloud Run logs..."
    gcloud logs tail --service=$SERVICE_NAME --region=$REGION
}

# Function to delete service
delete_service() {
    print_warning "This will delete the Cloud Run service: $SERVICE_NAME"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        gcloud run services delete $SERVICE_NAME --region=$REGION --quiet
        print_success "Service deleted successfully"
    else
        print_status "Deletion cancelled"
    fi
}

# Main script logic
main() {
    local command=${1:-help}
    local tag=${2:-latest}
    
    case $command in
        "build")
            check_prerequisites
            build_and_push $tag
            ;;
        "deploy")
            check_prerequisites
            deploy_to_cloud_run $tag
            ;;
        "build-and-deploy")
            check_prerequisites
            build_and_push $tag
            deploy_to_cloud_run $tag
            ;;
        "setup")
            check_prerequisites
            setup
            ;;
        "logs")
            show_logs
            ;;
        "delete")
            delete_service
            ;;
        "help"|*)
            show_usage
            ;;
    esac
}

# Run main function with all arguments
main "$@" 