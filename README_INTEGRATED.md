# 🎯 Integrated Image Labeling & Fine-Tuning Tool

A comprehensive tool that combines image annotation with AI-powered object detection and Vertex AI supervised fine-tuning. Create training data and deploy custom models in one seamless workflow.

## 🚀 Key Features

### **📝 Image Labeling**
- **Interactive Bounding Box Drawing**: Click and drag to create annotations
- **Multiple Format Support**: JSON, COCO, XML (Pascal VOC)
- **Batch Processing**: Process entire directories efficiently
- **Real-time Preview**: See annotations as you create them
- **Validation Tools**: Quality checks and statistics

### **🤖 AI-Powered Detection**
- **Multiple AI Models**: YOLO, Transformers (DETR), Google Gemini
- **Automatic Object Detection**: AI suggests bounding boxes and labels
- **Confidence Scoring**: Filter detections by confidence threshold
- **Smart Duplicate Detection**: Avoids overlapping annotations
- **Batch AI Processing**: Process entire directories automatically

### **🎯 Supervised Fine-Tuning**
- **GCP Integration**: Seamless authentication with Google Cloud
- **Gemini Prompt Generation**: AI-powered training prompt creation
- **Vertex AI Jobs**: Create AutoML and custom training jobs
- **Model Deployment**: Deploy trained models to Vertex AI endpoints
- **Dataset Management**: Prepare and upload datasets to GCS

### **🧠 Gemini Integration**
- **Automatic Prompt Generation**: Create training prompts from your labels
- **Task Description**: Describe your use case and get tailored prompts
- **Training Data Examples**: Generate example training data
- **Validation Prompts**: Create validation sets automatically

## 🛠️ Installation

### **Prerequisites**
- Python 3.8+
- Google Cloud Project with Vertex AI enabled
- Google API key for Gemini (optional)

### **Quick Setup**
```bash
# Clone and setup
git clone <your-fork-url>
cd streamlit-img-label

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up API keys (optional)
export GOOGLE_API_KEY="your-gemini-api-key"
```

## 🎯 Usage

### **Quick Start**
```bash
# Run the integrated app
python run_integrated.py --port 8502

# Or use the generic runner
python run_streamlit.py app_integrated.py --port 8502
```

### **Step-by-Step Workflow**

#### **1. Image Labeling**
1. **Upload Images**: Add your images to the `img_dir` folder
2. **Configure Labels**: Set your custom labels in the sidebar
3. **Choose AI Model**: Select YOLO, Transformers, or Gemini for assistance
4. **Annotate**: Draw bounding boxes and assign labels
5. **Validate**: Check annotation quality and statistics

#### **2. GCP Authentication**
1. **Project Setup**: Enter your GCP Project ID
2. **Choose Region**: Select your preferred GCP region
3. **Authentication**: Upload service account key or use default credentials
4. **Verify**: Confirm successful authentication

#### **3. Fine-Tuning Setup**
1. **Switch to Fine-Tuning Tab**: Click the "🎯 Fine-Tuning" tab
2. **Configure Task**: Select task type and training method
3. **Generate Prompts**: Use Gemini to create training prompts
4. **Review & Edit**: Customize prompts as needed

#### **4. Dataset Preparation**
1. **Prepare Dataset**: Convert annotations to training format
2. **Upload to GCS**: Store dataset in Google Cloud Storage
3. **Create Training Job**: Start AutoML or custom training
4. **Monitor Progress**: Track training job status

#### **5. Model Deployment**
1. **Deploy Model**: Create Vertex AI endpoint
2. **Test Endpoint**: Make predictions with your model
3. **Monitor Performance**: Track endpoint metrics

## 📋 Detailed Features

### **AI Model Configuration**

**YOLO Model**:
- Fast local processing
- Good accuracy for common objects
- No API key required

**Transformers (DETR)**:
- High accuracy with detailed detection
- Local processing
- Requires more computational resources

**Gemini Model**:
- Cloud-based processing
- Best accuracy
- Requires Google API key

### **GCP Authentication Methods**

**Service Account Key**:
```bash
# Upload JSON key file in the sidebar
# Automatically sets GOOGLE_APPLICATION_CREDENTIALS
```

**Application Default Credentials**:
```bash
# Use gcloud auth application-default login
gcloud auth application-default login
```

**OAuth 2.0**:
- Coming soon
- For interactive authentication

### **Fine-Tuning Options**

**AutoML Training**:
- Fully managed training
- Automatic hyperparameter tuning
- Best for most use cases

**Custom Training**:
- Full control over training process
- Custom model architectures
- Advanced hyperparameter tuning

### **Prompt Generation with Gemini**

The tool uses Gemini to automatically generate:
- **System Prompts**: Instructions for the model
- **Training Prompts**: Examples for each label
- **Validation Prompts**: Test cases for evaluation
- **Example Data**: Sample training data

**Example Gemini Prompt**:
```
I'm creating a supervised fine-tuning dataset for image classification. 
I have these labels: dog, cat, bird
Task description: Classify images based on the annotated objects

Please generate:
1. A system prompt for the model
2. Training prompts for each label
3. Validation prompts for each label
4. A few example training data entries
```

## 🔧 Configuration

### **Environment Variables**
```bash
# Required for GCP
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"

# Optional for AI features
export GOOGLE_API_KEY="your-gemini-api-key"
export CUDA_VISIBLE_DEVICES="0"  # GPU selection
```

### **Streamlit Configuration**
```toml
# .streamlit/config.toml
[server]
port = 8502
address = "localhost"
headless = false
enableCORS = true

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

### **Vertex AI Configuration**
```python
# Default settings
project_id = "your-project-id"
region = "us-central1"
bucket_name = "your-project-datasets"
```

## 📊 Workflow Examples

### **Example 1: Pet Classification**
```bash
# 1. Label pet images
python run_integrated.py --port 8502

# 2. In the app:
# - Upload pet images to img_dir
# - Set labels: "dog", "cat", "bird"
# - Use AI assistance for detection
# - Generate prompts with Gemini
# - Create AutoML training job
# - Deploy model to endpoint
```

### **Example 2: Product Detection**
```bash
# 1. Label product images
python run_integrated.py --port 8503

# 2. In the app:
# - Upload product images
# - Set labels: "phone", "laptop", "tablet"
# - Use custom training for specific requirements
# - Deploy to production endpoint
```

### **Example 3: Medical Image Analysis**
```bash
# 1. Label medical images
python run_integrated.py --port 8504

# 2. In the app:
# - Upload medical images
# - Set labels: "normal", "abnormal", "tumor"
# - Use high-confidence AI detection
# - Create custom training with specific parameters
```

## 🔍 Troubleshooting

### **Common Issues**

**GCP Authentication Failed**:
```bash
# Check service account permissions
# Verify project ID is correct
# Ensure Vertex AI API is enabled
```

**AI Model Not Loading**:
```bash
# Install dependencies
pip install torch torchvision ultralytics transformers

# Check GPU support
python -c "import torch; print(torch.cuda.is_available())"
```

**Gemini API Errors**:
```bash
# Verify API key
export GOOGLE_API_KEY="your-valid-api-key"
echo $GOOGLE_API_KEY
```

**Port Already in Use**:
```bash
# Use different port
python run_integrated.py --port 8503

# Or kill existing process
lsof -ti:8502 | xargs kill -9
```

### **Performance Optimization**

**For Large Datasets**:
- Use YOLO for speed-critical applications
- Process in smaller batches
- Enable GPU acceleration

**For High Accuracy**:
- Use Transformers or Gemini
- Lower confidence thresholds
- Manual review of AI suggestions

**Memory Optimization**:
- Process smaller batches
- Use CPU-only mode if needed
- Clear model cache between sessions

## 📈 Advanced Usage

### **Custom Training Scripts**
```python
# Custom training configuration
from vertex_tuning_manager import VertexTuningManager

manager = VertexTuningManager("your-project-id")
manager.create_custom_training_job(
    dataset_gcs_paths=dataset_paths,
    job_name="custom-model",
    model_type="efficientnet",
    hyperparameters={
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100
    }
)
```

### **Batch Processing**
```bash
# Process entire directory
python batch_ai_processing.py img_dir --model yolo --confidence 0.7

# Export to different formats
python batch_ai_processing.py img_dir --format coco --output-dir exports/
```

### **API Integration**
```python
# Use the API server for programmatic access
python api_server.py

# Make API calls
import requests
response = requests.post("http://localhost:8000/prepare_dataset", json={
    "img_dir": "img_dir",
    "labels": ["dog", "cat"],
    "task_type": "classification"
})
```

## 🚀 Deployment

### **Local Development**
```bash
# Quick start
python run_integrated.py --port 8502

# With custom config
python run_integrated.py --port 8502 --config .streamlit/custom_config.toml
```

### **Cloud Deployment**
```bash
# Deploy to Cloud Run
./deploy.sh build-and-deploy

# Or use Docker
docker build -t integrated-app .
docker run -p 8502:8502 integrated-app
```

### **Production Setup**
```bash
# Set production environment
export ENVIRONMENT=production
export GCS_BUCKET_NAME=your-production-bucket

# Run with production config
python run_integrated.py --port 8080 --address 0.0.0.0
```

## 📚 API Reference

### **VertexTuningManager Methods**
- `prepare_classification_dataset()`: Convert annotations to training format
- `upload_dataset_to_gcs()`: Upload dataset to Cloud Storage
- `create_automl_training_job()`: Create AutoML training job
- `create_custom_training_job()`: Create custom training job
- `deploy_model_to_endpoint()`: Deploy model to Vertex AI endpoint
- `list_training_jobs()`: List all training jobs
- `list_models()`: List trained models
- `list_endpoints()`: List deployed endpoints

### **AI Assistant Methods**
- `suggest_annotations()`: Get AI suggestions for image
- `validate_annotations()`: Validate existing annotations
- `batch_process()`: Process multiple images

### **Gemini Integration**
- `generate_training_prompts()`: Generate prompts from labels
- `create_example_data()`: Create training examples
- `validate_prompts()`: Validate generated prompts

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your enhancements
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original streamlit-img-label tool
- Google Vertex AI team
- Google Gemini team
- Streamlit community
- Open source AI model contributors

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the examples and documentation

## 📚 Additional Resources

- **[Cloud Deployment Guide](README_CLOUD.md)**: Deploy to Google Cloud Run
- **[AI Features Guide](README_AI.md)**: Detailed AI capabilities
- **[Port Configuration](README_PORT_CONFIG.md)**: Custom port setup
- **[Vertex AI Documentation](https://cloud.google.com/vertex-ai)**: Official Vertex AI docs
- **[Gemini API Documentation](https://ai.google.dev/)**: Official Gemini docs

## 🚀 Quick Commands

```bash
# Start integrated app
python run_integrated.py --port 8502

# Start with custom port
python run_integrated.py --port 8503

# Start with network access
python run_integrated.py --port 8502 --address 0.0.0.0

# Use generic runner
python run_streamlit.py app_integrated.py --port 8502

# Deploy to cloud
./deploy.sh build-and-deploy

# View logs
./deploy.sh logs
```

---

**Happy integrated labeling and fine-tuning! 🎉** 