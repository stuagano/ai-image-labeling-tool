# Stateless Cloud Run Workflow

This document explains how the AI Image Labeling Tool works in a stateless Cloud Run environment.

## 🚀 Stateless Architecture

Since Cloud Run is stateless, the service doesn't maintain any local state between requests. Each request is handled independently, and the service can be scaled to zero when not in use.

## 📁 Image Identification Strategy

### Problem
In a stateless environment, we can't rely on:
- Local file systems
- Session state persistence
- File paths that exist on the server

### Solution
We use a **content-based identification** strategy:

```python
# Generate unique identifier from image content
file_identifier = f"{uploaded_file.name}_{len(uploaded_file.getvalue())}"
```

### How It Works

1. **Upload**: User uploads an image through Streamlit
2. **Identification**: Service creates a unique ID based on filename + file size
3. **Storage**: Annotation is saved with this ID as the filename
4. **Retrieval**: Same ID is used to retrieve annotations later

## 🔄 Complete Workflow

### 1. Image Upload
```python
# User uploads image
uploaded_file = st.file_uploader("Choose an image file")

# Service generates identifier
file_identifier = f"{uploaded_file.name}_{len(uploaded_file.getvalue())}"
# Example: "street_scene_001.jpg_245760"
```

### 2. Hash Generation
```python
# Convert identifier to MD5 hash for consistent storage
import hashlib
hash_object = hashlib.md5(file_identifier.encode())
image_id = hash_object.hexdigest()
# Example: "a1b2c3d4e5f678901234567890123456"
```

### 3. Annotation Storage
```python
# Save annotation with hash as filename
annotation_data = {
    "rects": [...],
    "image_path": "street_scene_001.jpg",
    "metadata": {
        "image_path": "street_scene_001.jpg",
        "image_id": "a1b2c3d4e5f678901234567890123456",
        "saved_at": "2023-12-01T14:30:22.123456"
    }
}

# File stored as: annotations/a1b2c3d4e5f678901234567890123456.json
```

### 4. Annotation Retrieval
```python
# When same image is uploaded again
file_identifier = f"{uploaded_file.name}_{len(uploaded_file.getvalue())}"
image_id = generate_hash(file_identifier)
annotation = load_annotation(image_id)
```

## 📊 Cloud Storage Structure (Updated)

```
gs://your-bucket/
├── annotations/
│   ├── a1b2c3d4e5f678901234567890123456.json  # street_scene_001.jpg
│   ├── b2c3d4e5f67890123456789012345678.json  # park_scene_002.jpg
│   └── ...
├── exports/
│   └── dataset_coco_20231201_180000.json
└── uploads/  # Optional: store original images
    ├── street_scene_001.jpg
    └── park_scene_002.jpg
```

## 🔍 Example JSON Structure

### Individual Annotation File
```json
{
  "rects": [
    {
      "left": 150.5,
      "top": 200.3,
      "width": 120.8,
      "height": 180.2,
      "label": "person"
    }
  ],
  "image_path": "street_scene_001.jpg",
  "image_width": 1920,
  "image_height": 1080,
  "metadata": {
    "image_path": "street_scene_001.jpg",
    "image_id": "a1b2c3d4e5f678901234567890123456",
    "saved_at": "2023-12-01T14:30:22.123456",
    "version": "1.0"
  }
}
```

## ⚡ Benefits of This Approach

### 1. Stateless Operation
- No local file system dependencies
- Service can scale to zero
- Multiple instances can handle same image

### 2. Consistent Identification
- Same image always gets same ID
- Deterministic hash generation
- No conflicts between different uploads

### 3. Cloud-Native
- Works with Cloud Storage
- Compatible with CDN
- Supports global distribution

### 4. Scalable
- No database required
- Horizontal scaling support
- Cost-effective storage

## 🔧 Implementation Details

### Hash Generation Function
```python
def _generate_image_id(self, image_path: str) -> str:
    """Generate unique ID for image path"""
    import hashlib
    hash_object = hashlib.md5(image_path.encode())
    return hash_object.hexdigest()
```

### File Identifier Creation
```python
# In the Streamlit app
file_identifier = f"{uploaded_file.name}_{len(uploaded_file.getvalue())}"
```

### Storage Path
```python
# Cloud Storage path
file_path = f"annotations/{image_id}.json"
```

## 🚨 Limitations and Considerations

### 1. File Size Dependency
- ID changes if file size changes
- Same image with different compression = different ID
- Consider using content hash instead of size

### 2. Collision Handling
- MD5 has very low collision probability
- Consider SHA-256 for higher security
- Monitor for potential collisions

### 3. Image Storage
- Original images not stored by default
- Consider storing images if needed for training
- Use separate bucket for image storage

## 🔄 Alternative Approaches

### 1. Content-Based Hashing
```python
# Use image content instead of filename + size
import hashlib
image_bytes = uploaded_file.getvalue()
hash_object = hashlib.md5(image_bytes)
image_id = hash_object.hexdigest()
```

### 2. UUID Generation
```python
# Generate random UUID for each upload
import uuid
image_id = str(uuid.uuid4())
```

### 3. Database Integration
```python
# Use Cloud Firestore or Cloud SQL
# Store image metadata and annotation references
# More complex but more flexible
```

## 📈 Performance Considerations

### 1. Hash Computation
- MD5 is fast and sufficient for this use case
- Consider caching hash results
- Batch operations for multiple images

### 2. Cloud Storage Operations
- Use batch operations when possible
- Implement retry logic for failed operations
- Monitor storage costs and performance

### 3. Memory Usage
- Stream large files instead of loading entirely
- Use temporary files for processing
- Clean up resources after each request

## 🔒 Security Considerations

### 1. Access Control
- Use IAM roles for Cloud Storage access
- Implement proper authentication
- Audit access patterns

### 2. Data Privacy
- Consider encrypting sensitive annotations
- Implement data retention policies
- Comply with privacy regulations

### 3. Input Validation
- Validate uploaded file types
- Check file size limits
- Sanitize file names and paths

---

This stateless approach ensures the application works reliably in Cloud Run's serverless environment while maintaining data consistency and scalability. 