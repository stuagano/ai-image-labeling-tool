# ☁️ Cloud Deployment Guide

Complete guide for deploying the AI Image Labeling Tool to Google Cloud Run with Cloud Storage persistence.

## 🚀 Quick Start

### 1. Prerequisites

- Google Cloud Platform account
- Google Cloud CLI (`gcloud`) installed and configured
- Docker installed
- Git repository with the code

### 2. Initial Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd streamlit-img-label

# Make deployment script executable
chmod +x deploy.sh

# Run initial setup
./deploy.sh setup
```

### 3. Create Cloud Storage Bucket

```bash
# Create a bucket for storing annotations
gsutil mb gs://your-project-ai-annotations

# Make bucket publicly readable (optional, for public access)
gsutil iam ch allUsers:objectViewer gs://your-project-ai-annotations
```

### 4. Deploy to Cloud Run

```bash
# Deploy with default settings
./deploy.sh build-and-deploy

# Or deploy with custom tag
./deploy.sh build-and-deploy v1.0.0
```

## 📋 Detailed Setup

### Environment Variables

Set these environment variables in your Cloud Run service:

```bash
# Required
GCS_BUCKET_NAME=your-project-ai-annotations
GOOGLE_CLOUD_PROJECT=your-project-id

# Optional (for Gemini AI)
GOOGLE_API_KEY=your-gemini-api-key
```

### Manual Deployment Steps

If you prefer manual deployment:

```bash
# 1. Build Docker image
docker build -t gcr.io/YOUR_PROJECT_ID/ai-image-labeler:latest .

# 2. Push to Container Registry
docker push gcr.io/YOUR_PROJECT_ID/ai-image-labeler:latest

# 3. Deploy to Cloud Run
gcloud run deploy ai-image-labeler \
  --image gcr.io/YOUR_PROJECT_ID/ai-image-labeler:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 10 \
  --set-env-vars GCS_BUCKET_NAME=your-bucket-name
```

## 🔧 Configuration Options

### Cloud Run Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| Memory | 2Gi | Memory allocation per instance |
| CPU | 2 | CPU allocation per instance |
| Max Instances | 10 | Maximum concurrent instances |
| Timeout | 300s | Request timeout |
| Concurrency | 80 | Requests per instance |

### Custom Configuration

```bash
# Deploy with custom resources
gcloud run deploy ai-image-labeler \
  --image gcr.io/YOUR_PROJECT_ID/ai-image-labeler:latest \
  --memory 4Gi \
  --cpu 4 \
  --max-instances 20
```

## 📁 Cloud Storage Structure

The application uses the following Cloud Storage structure:

```
gs://your-bucket/
├── annotations/           # Individual annotation files
│   ├── image1.json
│   ├── image2.json
│   └── ...
├── exports/              # Dataset exports
│   ├── dataset_coco_20231201_143022.json
│   └── dataset_json_20231201_143022.json
└── uploads/              # Uploaded images (optional)
    ├── image1.jpg
    └── image2.png
```

### JSON Annotation Format

Each annotation file contains:

```json
{
  "rects": [
    {
      "left": 100,
      "top": 150,
      "width": 200,
      "height": 150,
      "label": "person"
    }
  ],
  "image_name": "image1.jpg",
  "image_width": 1920,
  "image_height": 1080,
  "labels": ["person", "car", "dog"],
  "format": "json",
  "metadata": {
    "image_name": "image1.jpg",
    "saved_at": "2023-12-01T14:30:22",
    "version": "1.0"
  }
}
```

## 🔐 Security & Permissions

### IAM Roles Required

```bash
# Grant Cloud Run service account access to Cloud Storage
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:YOUR_PROJECT_ID@appspot.gserviceaccount.com" \
  --role="roles/storage.objectViewer"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:YOUR_PROJECT_ID@appspot.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"
```

### Service Account Setup

```bash
# Create service account for Cloud Run
gcloud iam service-accounts create ai-labeler-sa \
  --display-name="AI Image Labeler Service Account"

# Grant necessary permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:ai-labeler-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

# Use service account in deployment
gcloud run deploy ai-image-labeler \
  --service-account=ai-labeler-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

## 🔄 CI/CD Pipeline

### GitHub Actions

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Cloud Run

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Cloud SDK
      uses: google-github-actions/setup-gcloud@v0
      with:
        project_id: ${{ secrets.GCP_PROJECT_ID }}
        service_account_key: ${{ secrets.GCP_SA_KEY }}
    
    - name: Build and push image
      run: |
        gcloud builds submit --tag gcr.io/${{ secrets.GCP_PROJECT_ID }}/ai-image-labeler:${{ github.sha }}
    
    - name: Deploy to Cloud Run
      run: |
        gcloud run deploy ai-image-labeler \
          --image gcr.io/${{ secrets.GCP_PROJECT_ID }}/ai-image-labeler:${{ github.sha }} \
          --region us-central1 \
          --platform managed \
          --allow-unauthenticated
```

### Cloud Build

Use the provided `cloudbuild.yaml`:

```bash
# Trigger build
gcloud builds submit --config cloudbuild.yaml
```

## 📊 Monitoring & Logging

### View Logs

```bash
# View real-time logs
./deploy.sh logs

# Or use gcloud directly
gcloud logs tail --service=ai-image-labeler --region=us-central1
```

### Set up Monitoring

```bash
# Enable monitoring
gcloud services enable monitoring.googleapis.com

# Create alerting policy
gcloud alpha monitoring policies create \
  --policy-from-file=monitoring-policy.yaml
```

### Health Checks

The application includes health checks:

```bash
# Check service health
curl https://your-service-url/_stcore/health
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Cloud Storage Access Denied

```bash
# Check service account permissions
gcloud projects get-iam-policy YOUR_PROJECT_ID \
  --flatten="bindings[].members" \
  --format="table(bindings.role)" \
  --filter="bindings.members:ai-labeler-sa"
```

#### 2. Memory Issues

```bash
# Increase memory allocation
gcloud run services update ai-image-labeler \
  --memory 4Gi \
  --region us-central1
```

#### 3. Cold Start Issues

```bash
# Set minimum instances
gcloud run services update ai-image-labeler \
  --min-instances 1 \
  --region us-central1
```

#### 4. AI Model Loading Failures

```bash
# Check environment variables
gcloud run services describe ai-image-labeler \
  --region us-central1 \
  --format="value(spec.template.spec.containers[0].env[].name,spec.template.spec.containers[0].env[].value)"
```

### Debug Commands

```bash
# Check service status
gcloud run services describe ai-image-labeler --region=us-central1

# View recent revisions
gcloud run revisions list --service=ai-image-labeler --region=us-central1

# Check resource usage
gcloud run services describe ai-image-labeler --region=us-central1 \
  --format="value(status.conditions[].message)"
```

## 💰 Cost Optimization

### Resource Optimization

```bash
# Use smaller instances for development
gcloud run deploy ai-image-labeler \
  --memory 1Gi \
  --cpu 1 \
  --max-instances 5
```

### Auto-scaling Configuration

```bash
# Configure auto-scaling
gcloud run services update ai-image-labeler \
  --min-instances 0 \
  --max-instances 10 \
  --concurrency 80
```

### Cost Monitoring

```bash
# Set up billing alerts
gcloud billing budgets create \
  --billing-account=YOUR_BILLING_ACCOUNT \
  --display-name="AI Labeler Budget" \
  --budget-amount=100USD \
  --threshold-rule=percent=0.5 \
  --threshold-rule=percent=0.8
```

## 🔄 Updates & Maintenance

### Rolling Updates

```bash
# Deploy new version
./deploy.sh build-and-deploy v1.1.0

# Rollback if needed
gcloud run services update-traffic ai-image-labeler \
  --to-revisions=REVISION_NAME=100 \
  --region us-central1
```

### Database Migration

```bash
# Export existing data
gsutil cp gs://your-bucket/annotations/* ./backup/

# Import to new bucket
gsutil cp ./backup/* gs://new-bucket/annotations/
```

## 📈 Scaling Considerations

### Performance Tuning

- **Memory**: Increase for large images or complex AI models
- **CPU**: Increase for faster AI processing
- **Concurrency**: Adjust based on request patterns
- **Max Instances**: Set based on expected load

### Load Testing

```bash
# Simple load test
for i in {1..100}; do
  curl -s https://your-service-url > /dev/null &
done
wait
```

## 🔒 Security Best Practices

1. **Use Service Accounts**: Don't use default compute service account
2. **Principle of Least Privilege**: Grant minimal required permissions
3. **Environment Variables**: Store secrets in Secret Manager
4. **HTTPS Only**: Enforce HTTPS for all traffic
5. **Regular Updates**: Keep dependencies updated

## 📞 Support

For issues and questions:

1. Check the troubleshooting section
2. Review Cloud Run logs
3. Check Cloud Storage permissions
4. Verify environment variables
5. Create an issue on GitHub

---

**Happy cloud deployment! ☁️🚀** 