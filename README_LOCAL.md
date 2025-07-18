# 🏠 Local Development Guide

Complete guide for running the AI Image Labeling Tool locally with FastAPI cloud integration.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables
```bash
export GCS_BUCKET_NAME="your-project-ai-annotations"
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_API_KEY="your-gemini-api-key"  # Optional
```

### 3. Start Both Services
```bash
python start_local.py
```

**Or start manually:**
```bash
# Terminal 1: Start FastAPI server
python api_server.py

# Terminal 2: Start Streamlit app
streamlit run app_local.py
```

## 📋 Services Overview

### FastAPI Server (`api_server.py`)
- **Port**: 8000
- **Purpose**: Cloud storage operations
- **Features**: REST API for annotations, descriptions, exports
- **Docs**: http://localhost:8000/docs

### Streamlit App (`app_local.py`)
- **Port**: 8501
- **Purpose**: User interface
- **Features**: Image upload, annotation, AI processing
- **URL**: http://localhost:8501

## 🔧 Architecture

```
┌─────────────────┐    HTTP Requests    ┌─────────────────┐
│   Streamlit     │ ──────────────────► │   FastAPI       │
│   (Local UI)    │                     │   (Cloud API)   │
│                 │ ◄────────────────── │                 │
└─────────────────┘    JSON Responses   └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Google Cloud    │
                    │ Storage         │
                    └─────────────────┘
```

## 🛠️ API Endpoints

### Health & Status
- `GET /` - Basic health check
- `GET /health` - Detailed health status
- `GET /statistics` - Storage statistics

### Annotations
- `POST /save-annotation` - Save annotation data
- `GET /load-annotation/{image_path}` - Load annotation
- `GET /list-annotations` - List all annotations
- `DELETE /delete-annotation/{image_path}` - Delete annotation

### Descriptions
- `POST /save-description` - Save Gemini description
- `GET /list-descriptions` - List all descriptions

### Exports
- `POST /export-dataset` - Export dataset (COCO/JSON)

### Batch Operations
- `POST /batch-process` - Process multiple images

## 🎯 Usage Workflow

### 1. Start Services
```bash
python start_local.py
```

### 2. Open Streamlit App
- Navigate to http://localhost:8501
- Verify FastAPI connection (green checkmark)

### 3. Configure Settings
- **API URL**: http://localhost:8000 (default)
- **AI Model**: Select YOLO, Transformers, or Gemini
- **Labels**: Enter custom labels
- **Format**: Choose JSON or COCO

### 4. Upload Images
- **Single Image**: Upload one image at a time
- **ZIP Folder**: Upload entire folder as ZIP
- **Batch Processing**: Navigate through multiple images

### 5. Process with AI
- **Individual**: Click "Get AI Suggestions" for current image
- **Batch**: Click "AI Process All" for all images
- **Gemini**: Click "Get Gemini Descriptions" for detailed analysis

### 6. Save & Export
- **Save**: Annotations saved to cloud storage
- **Export**: Download datasets in COCO or JSON format

## 🔍 API Documentation

### Interactive Docs
Visit http://localhost:8000/docs for interactive API documentation.

### Example API Calls

#### Save Annotation
```bash
curl -X POST "http://localhost:8000/save-annotation" \
  -H "Content-Type: application/json" \
  -d '{
    "rects": [
      {
        "left": 100,
        "top": 200,
        "width": 150,
        "height": 200,
        "label": "person"
      }
    ],
    "image_path": "image_001.jpg",
    "image_width": 1920,
    "image_height": 1080,
    "labels": ["person", "car"],
    "format": "json"
  }'
```

#### Load Annotation
```bash
curl "http://localhost:8000/load-annotation/image_001.jpg"
```

#### Export Dataset
```bash
curl -X POST "http://localhost:8000/export-dataset?format_type=coco"
```

#### Get Statistics
```bash
curl "http://localhost:8000/statistics"
```

## 🚨 Troubleshooting

### Common Issues

#### FastAPI Server Not Starting
```bash
# Check if port 8000 is available
lsof -i :8000

# Kill process if needed
kill -9 <PID>

# Start server manually
python api_server.py
```

#### Streamlit App Not Connecting
```bash
# Check API health
curl http://localhost:8000/health

# Verify environment variables
echo $GCS_BUCKET_NAME
echo $GOOGLE_CLOUD_PROJECT
```

#### Cloud Storage Connection Failed
```bash
# Check authentication
gcloud auth list

# Verify bucket exists
gsutil ls gs://your-bucket-name

# Test permissions
gsutil ls gs://your-bucket-name/annotations/
```

#### AI Models Not Loading
```bash
# Check dependencies
pip list | grep torch
pip list | grep transformers

# Verify API keys
echo $GOOGLE_API_KEY
```

### Error Messages

#### "FastAPI server not connected"
- Start the API server: `python api_server.py`
- Check if port 8000 is available
- Verify network connectivity

#### "Cloud storage not initialized"
- Set `GCS_BUCKET_NAME` environment variable
- Verify Google Cloud authentication
- Check bucket permissions

#### "AI model failed to load"
- Install required packages: `pip install torch transformers`
- Set API keys for cloud models
- Check internet connection

## 🔧 Development

### Adding New API Endpoints

1. **Add endpoint to `api_server.py`**:
```python
@app.post("/new-endpoint")
async def new_endpoint(data: YourModel):
    # Implementation
    return {"status": "success"}
```

2. **Add method to `APIClient` in `app_local.py`**:
```python
def new_endpoint(self, data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        response = requests.post(f"{self.base_url}/new-endpoint", json=data)
        return response.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}
```

3. **Use in Streamlit app**:
```python
result = st.session_state.api_client.new_endpoint(data)
```

### Environment Variables

#### Required
- `GCS_BUCKET_NAME`: Google Cloud Storage bucket name
- `GOOGLE_CLOUD_PROJECT`: Google Cloud project ID

#### Optional
- `GOOGLE_API_KEY`: Gemini API key
- `API_BASE_URL`: FastAPI server URL (default: http://localhost:8000)

### Configuration Files

#### `start_local.py`
- Dependency checking
- Environment validation
- Service startup coordination

#### `api_server.py`
- FastAPI server configuration
- Cloud storage integration
- REST endpoint definitions

#### `app_local.py`
- Streamlit UI configuration
- API client implementation
- User interaction logic

## 📊 Monitoring

### Health Checks
```bash
# API health
curl http://localhost:8000/health

# Streamlit status
curl http://localhost:8501/_stcore/health
```

### Logs
```bash
# FastAPI logs (in terminal)
python api_server.py

# Streamlit logs (in terminal)
streamlit run app_local.py
```

### Statistics
- Visit http://localhost:8000/statistics for storage stats
- Check sidebar in Streamlit app for real-time stats

## 🔒 Security

### Local Development
- Services run on localhost only
- No external access by default
- Environment variables for sensitive data

### Cloud Storage
- IAM-based access control
- Service account authentication
- Bucket-level permissions

### API Security
- CORS configured for localhost
- Input validation with Pydantic
- Error handling without data exposure

## 🚀 Deployment

### Production Considerations
- Use HTTPS for API communication
- Implement authentication
- Add rate limiting
- Configure logging
- Set up monitoring

### Scaling
- FastAPI supports async operations
- Streamlit can handle multiple users
- Cloud storage scales automatically

---

**Happy local development! 🎉** 