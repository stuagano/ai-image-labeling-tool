# 🤖 AI-Enhanced Image Labeling Tool

A modern, AI-powered image annotation tool using Streamlit with automatic object detection and labeling assistance. This tool combines the power of multiple AI models with an intuitive interface for efficient image annotation workflows.

## 🚀 Key Features

### AI-Powered Capabilities
- **Multiple AI Models**: YOLO, Transformers (DETR), and Google Gemini
- **Automatic Object Detection**: AI suggests bounding boxes and labels
- **Confidence Scoring**: Filter detections by confidence threshold
- **Smart Duplicate Detection**: Avoids overlapping annotations
- **Batch Processing**: Process entire directories automatically

### Traditional Features
- **Multiple Format Support**: JSON (custom), COCO JSON, YOLO, CSV, and XML (Pascal VOC)
- **Interactive Bounding Box Drawing**: Click and drag to create bounding boxes
- **Custom Labels**: Define your own label categories
- **Format Conversion**: Convert between different annotation formats
- **Batch Export**: Export entire datasets to various formats
- **Validation**: Quality checks for annotations
- **Real-time Preview**: See your annotations as you create them

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for faster AI processing)

### Local Setup

1. **Clone the repository**:
```bash
git clone <your-fork-url>
cd streamlit-img-label
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up API keys** (optional, for Gemini):
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

### 🏠 Local Development

For local development with cloud storage via FastAPI:

1. **Quick Start**:
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GCS_BUCKET_NAME="your-project-ai-annotations"

# Start both services
python start_local.py
```

2. **Manual Start**:
```bash
# Terminal 1: Start FastAPI server
python api_server.py

# Terminal 2: Start Streamlit app
streamlit run app_local.py
```

📖 **See [Local Development Guide](README_LOCAL.md) for detailed instructions**

### ☁️ Cloud Deployment

For production deployment to Google Cloud Run with Cloud Storage persistence:

1. **Quick Deploy**:
```bash
# Make deployment script executable
chmod +x deploy.sh

# Run initial setup
./deploy.sh setup

# Deploy to Cloud Run
./deploy.sh build-and-deploy
```

2. **Manual Setup**:
```bash
# Create Cloud Storage bucket
gsutil mb gs://your-project-ai-annotations

# Build and deploy
docker build -t gcr.io/YOUR_PROJECT_ID/ai-image-labeler:latest .
docker push gcr.io/YOUR_PROJECT_ID/ai-image-labeler:latest
gcloud run deploy ai-image-labeler --image gcr.io/YOUR_PROJECT_ID/ai-image-labeler:latest
```

📖 **See [Cloud Deployment Guide](README_CLOUD.md) for detailed instructions**

## 🎯 Usage

### Interactive AI-Enhanced Annotation

**Local Development (with FastAPI)**:
```bash
python start_local.py
```

**Local Development (manual)**:
```bash
streamlit run app_ai.py
```

**Cloud Deployment**:
```bash
streamlit run app_cloud.py
```

**Production (Cloud Run)**:
The application automatically runs the cloud-optimized version when deployed to Cloud Run.

### 📦 Bulk Upload Workflow

**1. Prepare Images**:
```bash
# Create ZIP from folder
python create_image_zip.py /path/to/your/images

# Custom output name
python create_image_zip.py /path/to/your/images -o my_dataset.zip

# Limit files for testing
python create_image_zip.py /path/to/your/images -m 50
```

**2. Configure Cloud Storage**:
- Set bucket URI in sidebar: `gs://your-project-ai-annotations`
- Or use environment variable: `export GCS_BUCKET_NAME="your-bucket"`

**3. Upload and Process**:
- Choose "ZIP Folder" upload option
- Upload your ZIP file
- Use "AI Process All" for automatic annotation
- Navigate through images with Previous/Next buttons

📖 **See [Bulk Upload Guide](README_BULK_UPLOAD.md) for detailed instructions**

### AI Model Configuration

1. **Select AI Model**:
   - **YOLO**: Fast, local processing with good accuracy
   - **Transformers**: High accuracy with DETR model
   - **Gemini**: Cloud-based, requires API key

2. **Configure Settings**:
   - Set confidence threshold (0.0-1.0)
   - Enter API key for cloud services
   - Choose annotation format

3. **AI Workflow**:
   - Click "Initialize AI Model" to load the selected model
   - Use "Get AI Suggestions" to analyze current image
   - Review suggestions and confidence scores
   - Apply AI suggestions with one click
   - Manually refine annotations as needed

### Batch AI Processing

Process entire directories automatically:

```bash
# Basic batch annotation
python batch_ai_processing.py img_dir --model yolo --confidence 0.7

# With custom output directory
python batch_ai_processing.py img_dir --model transformers --output-dir ai_annotations/

# Validate existing annotations
python batch_ai_processing.py img_dir --validate --model yolo

# Use Gemini with API key
python batch_ai_processing.py img_dir --model gemini --api-key YOUR_KEY

# Save results to file
python batch_ai_processing.py img_dir --model yolo --output-file results.json
```

### Command Line Options

```bash
python batch_ai_processing.py --help
```

**Options**:
- `--model`: AI model type (yolo, transformers, gemini)
- `--api-key`: API key for cloud services
- `--confidence`: Confidence threshold (0.0-1.0)
- `--format`: Annotation format (json, coco, xml)
- `--output-dir`: Output directory
- `--overwrite`: Overwrite existing annotations
- `--validate`: Validate existing annotations
- `--output-file`: Save results to file

## 🤖 AI Models Comparison

| Model | Speed | Accuracy | Local/Cloud | Best For |
|-------|-------|----------|-------------|----------|
| YOLO | ⚡⚡⚡ | ⭐⭐⭐⭐ | Local | Fast processing, good accuracy |
| Transformers | ⚡⚡ | ⭐⭐⭐⭐⭐ | Local | High accuracy, detailed detection |
| Gemini | ⚡ | ⭐⭐⭐⭐⭐ | Cloud | Best accuracy, requires API |

## 📊 AI Features in Detail

### Automatic Object Detection
- AI models analyze images and suggest bounding boxes
- Confidence scores help filter reliable detections
- Smart duplicate detection prevents overlapping annotations

### Quality Assurance
- Confidence threshold filtering
- Overlap detection and prevention
- Boundary validation
- Statistical analysis of annotations

### Batch Processing
- Process thousands of images automatically
- Progress tracking with detailed statistics
- Error handling and reporting
- Multiple output format support

### Validation Tools
- Compare AI detections with manual annotations
- Identify potential missed objects
- Analyze annotation quality
- Generate detailed reports

## ☁️ Cloud Features

### Cloud Storage Integration
- **JSON Persistence**: All annotations saved to Google Cloud Storage
- **Scalable Storage**: No local storage limitations
- **Data Durability**: Automatic backups and redundancy
- **Access Control**: IAM-based security

### Cloud Run Optimization
- **Serverless**: Auto-scaling based on demand
- **Cost Effective**: Pay only for actual usage
- **Global Access**: Deploy to multiple regions
- **Health Monitoring**: Built-in health checks

### Production Ready
- **CI/CD Pipeline**: Automated deployment with GitHub Actions
- **Environment Management**: Separate dev/staging/prod environments
- **Monitoring & Logging**: Cloud-native observability
- **Security**: Service accounts and IAM policies

## 🔧 Advanced Configuration

### Environment Variables

**Local Development**:
```bash
export GOOGLE_API_KEY="your-gemini-api-key"
export CUDA_VISIBLE_DEVICES="0"  # GPU selection
```

**Cloud Deployment**:
```bash
# Required for Cloud Storage
export GCS_BUCKET_NAME="your-project-ai-annotations"
export GOOGLE_CLOUD_PROJECT="your-project-id"

# Optional for AI features
export GOOGLE_API_KEY="your-gemini-api-key"
```

### Model Configuration
```python
# Custom confidence thresholds per model
YOLO_CONFIDENCE = 0.5
TRANSFORMERS_CONFIDENCE = 0.7
GEMINI_CONFIDENCE = 0.8
```

### Batch Processing Scripts
```python
# Custom batch processing
from batch_ai_processing import batch_ai_annotation

results = batch_ai_annotation(
    img_dir="path/to/images",
    model_type="yolo",
    confidence_threshold=0.6,
    annotation_format="coco"
)
```

## 📈 Performance Tips

### For Large Datasets
1. **Use YOLO** for speed-critical applications
2. **Batch process** during off-peak hours
3. **Use GPU acceleration** when available
4. **Set appropriate confidence thresholds** to reduce false positives

### For High Accuracy
1. **Use Transformers or Gemini** for critical applications
2. **Lower confidence thresholds** for more detections
3. **Manual review** of AI suggestions
4. **Combine multiple models** for validation

### Memory Optimization
1. **Process in smaller batches** for large datasets
2. **Use CPU-only mode** if GPU memory is limited
3. **Clear model cache** between processing sessions

## 🔍 Troubleshooting

### Common Issues

**AI Model Not Loading**:
```bash
# Check dependencies
pip install torch torchvision ultralytics transformers

# Verify GPU support
python -c "import torch; print(torch.cuda.is_available())"
```

**Gemini API Errors**:
```bash
# Verify API key
export GOOGLE_API_KEY="your-valid-api-key"
echo $GOOGLE_API_KEY
```

**Memory Issues**:
```bash
# Use CPU-only mode
export CUDA_VISIBLE_DEVICES=""

# Reduce batch size
python batch_ai_processing.py img_dir --model yolo
```

### Performance Issues
- **Slow processing**: Use YOLO model or enable GPU
- **High memory usage**: Process smaller batches
- **Low accuracy**: Adjust confidence threshold or use different model

## 📝 Examples

### Basic AI Annotation Workflow
```python
from ai_utils import create_ai_detector, create_ai_assistant
from ai_image_manager import AIImageManager

# Initialize AI
detector = create_ai_detector("yolo")
assistant = create_ai_assistant(detector)

# Process image
im = AIImageManager("image.jpg", "json")
suggestions = assistant.suggest_annotations("image.jpg")
im.set_ai_annotations(suggestions['detections'])
im.save_annotation()
```

### Custom AI Integration
```python
# Custom confidence filtering
detections = detector.detect_objects("image.jpg", confidence_threshold=0.8)

# Custom validation
validation = assistant.validate_annotations("image.jpg", existing_annotations)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add AI enhancements
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original streamlit-img-label tool
- YOLO and Ultralytics team
- Hugging Face Transformers
- Google Gemini API
- Streamlit community

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the examples and documentation

## 📚 Additional Resources

- **[Cloud Deployment Guide](README_CLOUD.md)**: Complete guide for deploying to Google Cloud Run
- **[Local Development](README.md)**: Original documentation for local development
- **[AI Features](README_AI.md)**: Detailed AI capabilities and usage

## 🚀 Quick Start Commands

```bash
# Local development (with FastAPI)
python start_local.py

# Local development (manual)
streamlit run app_ai.py

# Cloud deployment
./deploy.sh build-and-deploy

# Bulk upload preparation
python create_image_zip.py /path/to/images

# View logs
./deploy.sh logs

# Delete service
./deploy.sh delete
```

---

**Happy AI-powered annotating! 🎉** 