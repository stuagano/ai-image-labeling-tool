# 🤖 Vertex AI Supervised Tuning

Transform your annotated images into production-ready classification models with Google Vertex AI. This feature allows you to create supervised tuning jobs that automatically train image classification models from your annotations and deploy them to scalable endpoints.

## 🚀 Features

### **End-to-End ML Pipeline**
- **Dataset Preparation**: Automatically convert annotations to classification datasets
- **Model Training**: Support for both AutoML and custom training jobs
- **Model Deployment**: Deploy trained models to Vertex AI endpoints
- **Production Ready**: Scalable, managed endpoints for real-time predictions

### **Training Options**
- **AutoML**: Google's automated ML training (recommended for most users)
- **Custom Training**: Full control over model architecture and hyperparameters
- **Multiple Architectures**: EfficientNet, ResNet, MobileNet, and custom CNNs

### **Integration**
- **Seamless Workflow**: From annotation to deployment in one tool
- **Cloud Storage**: Automatic dataset and model storage in Google Cloud
- **API Access**: REST API endpoints for programmatic access
- **Monitoring**: Track training progress and model performance

## 🛠️ Setup

### Prerequisites

1. **Google Cloud Project** with Vertex AI enabled
2. **Service Account** with Vertex AI permissions
3. **Cloud Storage Bucket** for datasets and models
4. **Required Dependencies**:

```bash
pip install google-cloud-aiplatform pandas tensorflow pillow
```

### Environment Variables

```bash
# Required for Vertex AI
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GCS_BUCKET_NAME="your-bucket-name"

# Optional for additional features
export GOOGLE_API_KEY="your-api-key"
```

### IAM Permissions

Your service account needs these roles:
- `roles/aiplatform.admin`
- `roles/storage.admin`
- `roles/iam.serviceAccountUser`

## 📖 Usage

### 1. **Start the Application**

**Option A: Direct Streamlit App**
```bash
streamlit run app_vertex_tuning.py
```

**Option B: Local Development with FastAPI**
```bash
python start_local.py
# Then navigate to the Vertex Tuning tab
```

### 2. **Connect to Vertex AI**

1. Enter your Google Cloud Project ID
2. Select your preferred region
3. Provide your Cloud Storage bucket name
4. Click "Connect to Vertex AI"

### 3. **Load Annotations**

The system will automatically load your existing annotations from:
- Google Cloud Storage (if configured)
- Local JSON files (fallback)

### 4. **Prepare Classification Dataset**

Configure your dataset:
- **Dataset Name**: Unique identifier for your dataset
- **Training Split**: Percentage for training (default: 80%)
- **Minimum Images per Class**: Filter out classes with too few samples
- **Maximum Classes**: Limit the number of classes for training

### 5. **Create Training Job**

Choose your training approach:

#### **AutoML Training** (Recommended)
- **Budget**: Set training time limit (1-24 hours)
- **Automatic**: Google handles model selection and optimization
- **Best for**: Most use cases, especially with limited ML expertise

#### **Custom Training**
- **Model Architecture**: EfficientNet, ResNet, MobileNet, or custom CNN
- **Hyperparameters**: Learning rate, epochs, batch size
- **Full Control**: Customize every aspect of training

### 6. **Monitor Training**

Track your training jobs:
- **Status**: Running, completed, failed
- **Progress**: Real-time updates
- **Metrics**: Accuracy, loss, training time

### 7. **Deploy Models**

Once training completes:
- **Select Model**: Choose from your trained models
- **Configure Endpoint**: Set machine type and scaling
- **Deploy**: Create production-ready endpoint

### 8. **Test Predictions**

Upload test images to:
- **Verify Model**: Test classification accuracy
- **Get Predictions**: See confidence scores
- **Validate Deployment**: Ensure endpoint is working

## 🔧 API Reference

### Dataset Preparation

```python
# Prepare classification dataset
POST /vertex/prepare-dataset
{
    "dataset_name": "my_classification_dataset",
    "train_split": 0.8,
    "min_images_per_class": 10,
    "max_classes": 50
}
```

### Training Job Creation

```python
# Create AutoML training job
POST /vertex/create-training-job
{
    "job_name": "my_training_job",
    "training_type": "AutoML",
    "budget_hours": 8
}

# Create custom training job
POST /vertex/create-training-job
{
    "job_name": "my_custom_job",
    "training_type": "Custom",
    "model_type": "efficientnet",
    "epochs": 50,
    "learning_rate": 0.001
}
```

### Model Deployment

```python
# Deploy model to endpoint
POST /vertex/deploy-model
{
    "model_name": "my_trained_model",
    "endpoint_name": "my_production_endpoint",
    "machine_type": "n1-standard-2"
}
```

### Resource Management

```python
# List training jobs
GET /vertex/training-jobs

# List trained models
GET /vertex/models

# List deployed endpoints
GET /vertex/endpoints
```

## 📊 Workflow Examples

### **Example 1: Quick AutoML Classification**

```python
from vertex_tuning_manager import VertexTuningManager

# Initialize
manager = VertexTuningManager("your-project-id")
manager.set_storage_bucket("your-bucket")

# Prepare dataset
dataset_paths = manager.prepare_classification_dataset(annotations)
gcs_paths = manager.upload_dataset_to_gcs(dataset_paths)

# Create AutoML job
job_info = manager.create_automl_training_job(
    gcs_paths, 
    "quick_classifier",
    budget_milli_node_hours=8000  # 8 hours
)

# Deploy when ready
endpoint_info = manager.deploy_model_to_endpoint(
    job_info['model'],
    "production_endpoint"
)
```

### **Example 2: Custom Model Training**

```python
# Create custom training job
hyperparameters = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'image_size': 224,
    'dropout_rate': 0.2
}

job_info = manager.create_custom_training_job(
    gcs_paths,
    "custom_classifier",
    model_type="efficientnet",
    hyperparameters=hyperparameters
)
```

### **Example 3: Batch Processing**

```python
# Process multiple datasets
datasets = ["dataset1", "dataset2", "dataset3"]

for dataset_name in datasets:
    # Prepare dataset
    dataset_paths = manager.prepare_classification_dataset(
        annotations, 
        output_dir=f"datasets/{dataset_name}"
    )
    
    # Upload to GCS
    gcs_paths = manager.upload_dataset_to_gcs(dataset_paths)
    
    # Create training job
    job_info = manager.create_automl_training_job(
        gcs_paths,
        f"{dataset_name}_classifier"
    )
```

## 🏗️ Architecture

### **Data Flow**

```
Annotations → Dataset Preparation → GCS Upload → Training Job → Model → Endpoint
     ↓              ↓                    ↓            ↓         ↓        ↓
  JSON Files    CSV Files          Cloud Storage   Vertex AI  Model    Production
```

### **Components**

1. **VertexTuningManager**: Core orchestration class
2. **Dataset Preparation**: Converts annotations to classification format
3. **Training Scripts**: Custom training code for Vertex AI
4. **API Endpoints**: REST interface for all operations
5. **Streamlit UI**: User-friendly interface

### **Storage Structure**

```
gs://your-bucket/
├── datasets/
│   ├── dataset1/
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   └── dataset_info.json
│   └── dataset2/
├── models/
│   └── trained_models/
└── exports/
    └── model_exports/
```

## 📈 Best Practices

### **Dataset Preparation**

1. **Class Balance**: Ensure each class has sufficient samples (10+ minimum)
2. **Image Quality**: Use high-quality, consistent images
3. **Annotation Quality**: Accurate, consistent labels
4. **Data Augmentation**: Consider augmenting small classes

### **Training Configuration**

1. **AutoML**: Start with 8-hour budget for most datasets
2. **Custom Training**: Use transfer learning with pre-trained models
3. **Hyperparameters**: Start with defaults, tune based on results
4. **Validation**: Always use validation split to monitor overfitting

### **Deployment**

1. **Machine Type**: Start with n1-standard-2, scale as needed
2. **Scaling**: Configure auto-scaling for production workloads
3. **Monitoring**: Set up alerts for endpoint health
4. **Versioning**: Keep multiple model versions for rollback

## 🔍 Monitoring & Debugging

### **Training Monitoring**

- **Vertex AI Console**: Real-time training metrics
- **TensorBoard**: Detailed training logs (custom training)
- **API Endpoints**: Programmatic status checking

### **Common Issues**

1. **Insufficient Data**: Add more images per class
2. **Training Timeout**: Increase budget or reduce dataset size
3. **Poor Accuracy**: Check annotation quality and class balance
4. **Deployment Failures**: Verify model format and permissions

### **Debugging Commands**

```bash
# Check Vertex AI status
gcloud ai operations list --region=us-central1

# View training logs
gcloud ai custom-jobs describe JOB_ID --region=us-central1

# Test endpoint
curl -X POST ENDPOINT_URL \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  -d '{"instances": [{"image": "base64_encoded_image"}]}'
```

## 💰 Cost Optimization

### **Training Costs**

- **AutoML**: ~$20-50 per hour depending on region
- **Custom Training**: ~$2-10 per hour for compute
- **Storage**: ~$0.02 per GB per month

### **Deployment Costs**

- **Endpoints**: ~$0.50-2.00 per hour depending on machine type
- **Predictions**: ~$0.10-0.50 per 1000 predictions

### **Cost Reduction Tips**

1. **Use AutoML**: Often more cost-effective than custom training
2. **Optimize Dataset**: Remove unnecessary images and classes
3. **Right-size Endpoints**: Start small, scale as needed
4. **Clean Up**: Delete unused models and endpoints

## 🔐 Security

### **Authentication**

- **Service Accounts**: Use dedicated service accounts
- **IAM Roles**: Follow principle of least privilege
- **API Keys**: Secure storage of credentials

### **Data Protection**

- **Encryption**: All data encrypted at rest and in transit
- **Access Control**: IAM-based permissions
- **Audit Logging**: Track all operations

## 🚀 Production Deployment

### **CI/CD Pipeline**

```yaml
# Example GitHub Actions workflow
name: Vertex AI Training Pipeline
on:
  push:
    branches: [main]

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run training
        run: python train_classifier.py
        env:
          GOOGLE_CLOUD_PROJECT: ${{ secrets.GCP_PROJECT }}
          GCS_BUCKET_NAME: ${{ secrets.GCS_BUCKET }}
```

### **Monitoring Setup**

1. **Cloud Monitoring**: Set up dashboards for endpoints
2. **Alerting**: Configure alerts for errors and performance
3. **Logging**: Centralized logging with Cloud Logging
4. **Metrics**: Track prediction latency and accuracy

## 📚 Additional Resources

- [Vertex AI Documentation](https://cloud.google.com/vertex-ai)
- [AutoML Image Classification](https://cloud.google.com/vertex-ai/docs/image-data/classification)
- [Custom Training Jobs](https://cloud.google.com/vertex-ai/docs/training/create-custom-job)
- [Model Deployment](https://cloud.google.com/vertex-ai/docs/predictions/deploy-model)

## 🤝 Support

For issues and questions:
- Check the troubleshooting section
- Review Vertex AI documentation
- Create an issue on GitHub
- Contact Google Cloud support

---

**Transform your annotations into production ML models with Vertex AI! 🚀** 