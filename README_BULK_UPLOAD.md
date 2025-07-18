# 📦 Bulk Upload Guide

Complete guide for uploading folders of images to the AI Image Labeling Tool with bucket URI configuration.

## 🚀 Quick Start

### 1. Prepare Your Images

**Option A: Use the ZIP Creation Script**
```bash
# Create ZIP from folder
python create_image_zip.py /path/to/your/images

# Create ZIP with custom name
python create_image_zip.py /path/to/your/images -o my_dataset.zip

# Limit number of files
python create_image_zip.py /path/to/your/images -m 100
```

**Option B: Manual ZIP Creation**
1. Select all images in your folder
2. Right-click → "Compress" (Mac) or "Send to → Compressed folder" (Windows)
3. Rename to `images_for_labeling.zip`

### 2. Configure Cloud Storage

**Option A: Environment Variable**
```bash
export GCS_BUCKET_NAME="your-project-ai-annotations"
```

**Option B: UI Configuration**
1. Open the Cloud Labeling Tool
2. In sidebar, enter your bucket URI: `gs://your-project-ai-annotations`
3. Click "Update Bucket"

### 3. Upload and Process

1. **Select Upload Option**: Choose "ZIP Folder"
2. **Upload ZIP**: Drag and drop your ZIP file
3. **Configure AI**: Set up AI model and confidence threshold
4. **Process**: Use "AI Process All" for automatic annotation

## 📁 Supported Image Formats

The tool supports these image formats:
- **JPEG**: `.jpg`, `.jpeg`
- **PNG**: `.png`
- **GIF**: `.gif`
- **BMP**: `.bmp`
- **TIFF**: `.tiff`
- **WebP**: `.webp`

## 🔧 ZIP Creation Script Usage

### Basic Usage
```bash
# Create ZIP from folder
python create_image_zip.py /path/to/images

# List images without creating ZIP
python create_image_zip.py /path/to/images --list-only
```

### Advanced Options
```bash
# Custom output name
python create_image_zip.py /path/to/images -o dataset_v1.zip

# Limit number of files
python create_image_zip.py /path/to/images -m 50

# Combine options
python create_image_zip.py /path/to/images -o small_dataset.zip -m 25
```

### Example Output
```bash
📁 Found 150 image files in /path/to/images
Found 50 image files
Limited to 50 files
Added: image_001.jpg
Added: image_002.jpg
...
Added: image_050.jpg

✅ Successfully created small_dataset.zip
📦 Total files: 50
📏 File size: 15.2 MB

🚀 Ready to upload to Cloud Labeling Tool!
📤 Upload 'small_dataset.zip' using the 'ZIP Folder' option
```

## ☁️ Cloud Storage Configuration

### Bucket URI Format
```
gs://your-project-ai-annotations
```

### Setting Up Bucket
```bash
# Create bucket
gsutil mb gs://your-project-ai-annotations

# Make bucket publicly readable (optional)
gsutil iam ch allUsers:objectViewer gs://your-project-ai-annotations
```

### Environment Variables
```bash
# Required
export GCS_BUCKET_NAME="your-project-ai-annotations"
export GOOGLE_CLOUD_PROJECT="your-project-id"

# Optional (for Gemini AI)
export GOOGLE_API_KEY="your-gemini-api-key"
```

## 🔄 Workflow Options

### 1. Single Image Processing
- Upload one image at a time
- Manual annotation with AI assistance
- Good for small datasets or detailed work

### 2. ZIP Folder Upload
- Upload entire folder as ZIP
- Navigate through images with Previous/Next
- Process individual images or entire batch

### 3. Batch Processing
- AI-process all images automatically
- Save all annotations at once
- Perfect for large datasets

## 🤖 AI Processing Options

### Individual AI Processing
1. Upload ZIP file
2. Navigate to specific image
3. Click "🤖 Get AI Suggestions"
4. Review and refine annotations
5. Save individual image

### Batch AI Processing
1. Upload ZIP file
2. Configure AI model and confidence
3. Click "🤖 AI Process All"
4. Monitor progress bar
5. All annotations saved automatically

### Gemini Descriptions
1. Initialize Gemini AI model
2. Upload ZIP file
3. Click "🤖 Get Gemini Descriptions"
4. View detailed descriptions for all images
5. Descriptions saved to cloud storage

### AI Models Available
- **YOLO**: Fast, good accuracy, local processing
- **Transformers**: High accuracy, detailed detection
- **Gemini**: Best accuracy, requires API key

## 📊 Batch Processing Features

### Navigation Controls
- **Previous/Next**: Navigate through images
- **Progress Indicator**: Shows current position
- **Image Counter**: "Image 5 of 150"

### Batch Actions
- **AI Process All**: Automatically annotate all images
- **Save All Annotations**: Save current annotations for all images
- **Get Gemini Descriptions**: Generate detailed descriptions for all images
- **View All Descriptions**: Display all Gemini descriptions in expandable sections
- **Progress Tracking**: Real-time progress bars

### Statistics Display
- Total images in batch
- Current image position
- Processing status

## 💾 Storage and Organization

### Cloud Storage Structure
```
gs://your-bucket/
├── annotations/
│   ├── a1b2c3d4e5f678901234567890123456.json  # image_001.jpg
│   ├── b2c3d4e5f67890123456789012345678.json  # image_002.jpg
│   └── ...
├── descriptions/
│   ├── image_001_1234.json  # Gemini descriptions
│   ├── image_002_2345.json
│   └── ...
├── exports/
│   └── dataset_coco_20231201_180000.json
└── uploads/  # Optional: store original images
```

### Annotation File Format
```json
{
  "rects": [
    {
      "left": 150.5,
      "top": 200.3,
      "width": 120.8,
      "height": 180.2,
      "label": "person",
      "confidence": 0.87,
      "ai_detected": true
    }
  ],
  "image_path": "image_001.jpg",
  "metadata": {
    "image_path": "image_001.jpg",
    "image_id": "a1b2c3d4e5f678901234567890123456",
    "saved_at": "2023-12-01T14:30:22.123456"
  }
}
```

## 🚨 Best Practices

### ZIP File Preparation
1. **Organize images**: Use descriptive folder structure
2. **Limit file size**: Keep ZIP under 100MB for best performance
3. **Use consistent naming**: `image_001.jpg`, `image_002.jpg`, etc.
4. **Check formats**: Ensure all images are supported formats

### AI Processing
1. **Start with YOLO**: Fast and reliable for most use cases
2. **Adjust confidence**: Lower threshold = more detections
3. **Review results**: Always check AI suggestions
4. **Manual refinement**: Fine-tune bounding boxes as needed

### Cloud Storage
1. **Use descriptive bucket names**: `project-name-annotations`
2. **Set up permissions**: Grant appropriate IAM roles
3. **Monitor costs**: Check storage usage regularly
4. **Backup data**: Export datasets periodically

## 🔧 Troubleshooting

### Common Issues

#### ZIP File Too Large
```bash
# Limit number of files
python create_image_zip.py /path/to/images -m 50

# Check file sizes
python create_image_zip.py /path/to/images --list-only
```

#### Bucket Connection Failed
```bash
# Check bucket exists
gsutil ls gs://your-bucket-name

# Verify permissions
gcloud auth list
```

#### AI Processing Slow
- Use YOLO model for faster processing
- Reduce confidence threshold
- Process smaller batches

#### Memory Issues
- Reduce batch size
- Use smaller images
- Process in smaller chunks

### Error Messages

#### "No images found in ZIP file"
- Check file extensions are supported
- Ensure images are not in subdirectories
- Verify ZIP file is not corrupted

#### "Failed to connect to bucket"
- Verify bucket name is correct
- Check IAM permissions
- Ensure project is set correctly

#### "AI model failed to load"
- Check internet connection
- Verify API key (for Gemini)
- Try different AI model

## 📈 Performance Tips

### For Large Datasets
1. **Process in chunks**: Upload 50-100 images at a time
2. **Use batch AI**: Let AI process all images automatically
3. **Monitor progress**: Watch progress bars for completion
4. **Export regularly**: Save datasets periodically

### For High Accuracy
1. **Use Transformers/Gemini**: Higher accuracy models
2. **Lower confidence**: Catch more potential objects
3. **Manual review**: Check each annotation
4. **Refine boxes**: Adjust bounding boxes precisely

### For Speed
1. **Use YOLO**: Fastest processing
2. **Higher confidence**: Fewer false positives
3. **Smaller images**: Faster processing
4. **Batch processing**: Process all at once

## 🔄 Export Options

### COCO Format
- Standard format for ML training
- Compatible with most frameworks
- Includes all metadata

### JSON Format
- Custom format with full details
- Includes AI confidence scores
- Preserves workflow information

### Gemini Descriptions
- Detailed scene descriptions for all images
- Structured analysis of objects, actions, and context
- Useful for dataset documentation and training

### Export Process
1. Click "Export COCO Dataset" or "Export JSON Dataset"
2. Wait for processing
3. Dataset saved to `exports/` folder
4. Download from Cloud Storage

## 📞 Support

### Getting Help
1. Check this guide first
2. Review error messages carefully
3. Try with smaller dataset
4. Check Cloud Storage permissions

### Common Solutions
- **Slow processing**: Use YOLO model
- **Connection issues**: Verify bucket name
- **Memory errors**: Reduce batch size
- **AI failures**: Check API keys and internet

---

**Happy bulk labeling! 🎉** 