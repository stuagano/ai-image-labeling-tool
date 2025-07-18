# Cloud Storage Bucket Structure

This document shows the complete structure of the Cloud Storage bucket used for storing AI image annotations.

## Bucket Structure

```
gs://your-project-ai-annotations/
├── annotations/                    # Individual annotation files (by hash)
│   ├── a1b2c3d4e5f678901234567890123456.json  # street_scene_001.jpg
│   ├── b2c3d4e5f67890123456789012345678.json  # park_scene_002.jpg
│   ├── c3d4e5f6789012345678901234567890.json  # office_building_003.jpg
│   ├── d4e5f678901234567890123456789012.json  # city_street_004.jpg
│   └── ...
├── descriptions/                  # Gemini AI descriptions
│   ├── street_scene_001_1234.json  # Detailed scene descriptions
│   ├── park_scene_002_2345.json
│   └── ...
├── exports/                       # Dataset exports
│   ├── dataset_coco_20231201_180000.json
│   ├── dataset_json_20231201_180000.json
│   └── ...
├── uploads/                       # Uploaded images (optional)
│   ├── street_scene_001.jpg
│   ├── park_scene_002.jpg
│   └── ...
└── metadata/                      # Bucket metadata
    ├── labels.json               # Available labels
    ├── statistics.json           # Annotation statistics
    └── config.json              # Application configuration
```

## File Naming Conventions

### Annotation Files
- **Format**: `{image_hash}.json`
- **Location**: `annotations/` directory
- **Example**: `a1b2c3d4e5f678901234567890123456.json`
- **Hash Generation**: MD5 hash of image path + file size for stateless service

### Description Files
- **Format**: `{image_name}_{description_length}.json`
- **Location**: `descriptions/` directory
- **Example**: `street_scene_001_1234.json`
- **Content**: Detailed Gemini AI descriptions of image content

### Export Files
- **Format**: `dataset_{format}_{timestamp}.json`
- **Location**: `exports/` directory
- **Example**: `dataset_coco_20231201_180000.json`

### Upload Files
- **Format**: `{image_name}.{extension}`
- **Location**: `uploads/` directory
- **Example**: `street_scene_001.jpg`

## File Paths in Cloud Storage

### Individual Annotations
```
gs://your-bucket/annotations/a1b2c3d4e5f678901234567890123456.json  # street_scene_001.jpg
gs://your-bucket/annotations/b2c3d4e5f67890123456789012345678.json  # park_scene_002.jpg
gs://your-bucket/annotations/c3d4e5f6789012345678901234567890.json  # office_building_003.jpg
gs://your-bucket/annotations/d4e5f678901234567890123456789012.json  # city_street_004.jpg
```

### Dataset Exports
```
gs://your-bucket/exports/dataset_coco_20231201_180000.json
gs://your-bucket/exports/dataset_json_20231201_180000.json
gs://your-bucket/exports/dataset_yolo_20231201_180000.json
```

### Uploaded Images
```
gs://your-bucket/uploads/street_scene_001.jpg
gs://your-bucket/uploads/park_scene_002.jpg
gs://your-bucket/uploads/office_building_003.jpg
gs://your-bucket/uploads/city_street_004.jpg
```

## Access Patterns

### Reading Annotations
```python
# Get annotation for specific image (using image path)
annotation = cloud_manager.load_annotation(image_path)

# List all annotations
annotations = storage_manager.list_files(prefix="annotations/")
```

### Writing Annotations
```python
# Save annotation for image (using image path)
cloud_manager.save_annotation(image_path, annotation_data)
```

### Exporting Datasets
```python
# Export all annotations to COCO format
dataset = cloud_manager.export_dataset("coco")
cloud_manager.save_dataset_export(dataset, "coco")
```

## File Size Estimates

### Individual Annotation Files
- **Small image** (1-3 objects): ~2-5 KB
- **Medium image** (5-10 objects): ~5-15 KB
- **Large image** (10+ objects): ~15-50 KB

### Dataset Export Files
- **100 images**: ~50-100 KB
- **1,000 images**: ~500 KB - 1 MB
- **10,000 images**: ~5-10 MB

### Uploaded Images
- **JPEG** (1920x1080): ~200-500 KB
- **PNG** (1920x1080): ~1-3 MB
- **High-res** (4K): ~2-8 MB

## Storage Costs (Estimated)

### Google Cloud Storage Pricing (US)
- **Standard Storage**: $0.020 per GB per month
- **Nearline Storage**: $0.010 per GB per month (30+ days)
- **Coldline Storage**: $0.004 per GB per month (90+ days)

### Cost Examples
- **1,000 annotations**: ~5 MB = $0.0001/month
- **10,000 annotations**: ~50 MB = $0.001/month
- **100,000 annotations**: ~500 MB = $0.01/month

## Backup and Versioning

### Object Versioning
```bash
# Enable object versioning
gsutil versioning set on gs://your-bucket

# List versions
gsutil ls -a gs://your-bucket/annotations/
```

### Lifecycle Management
```bash
# Move old files to cheaper storage
gsutil lifecycle set lifecycle.json gs://your-bucket
```

Example `lifecycle.json`:
```json
{
  "rule": [
    {
      "action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
      "condition": {
        "age": 30,
        "matchesStorageClass": ["STANDARD"]
      }
    }
  ]
}
```

## Security and Access Control

### IAM Roles
- **Storage Object Viewer**: Read access to annotations
- **Storage Object Admin**: Full access to bucket
- **Storage Admin**: Manage bucket settings

### Service Account Permissions
```bash
# Grant Cloud Run service account access
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:PROJECT_ID@appspot.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"
```

## Monitoring and Analytics

### Cloud Monitoring Metrics
- **Object count**: Number of annotation files
- **Storage usage**: Total bucket size
- **Access patterns**: Read/write frequency
- **Error rates**: Failed operations

### Logging
```bash
# View bucket access logs
gsutil logging get gs://your-bucket
```

## Best Practices

### File Organization
1. **Use consistent naming**: `{image_name}.json`
2. **Organize by date**: `annotations/2023/12/01/`
3. **Version control**: Include version in filename
4. **Metadata**: Store configuration separately

### Performance
1. **Batch operations**: Use batch uploads for efficiency
2. **Caching**: Cache frequently accessed annotations
3. **Compression**: Compress large export files
4. **CDN**: Use Cloud CDN for global access

### Security
1. **Encryption**: Enable default encryption
2. **Access logs**: Enable access logging
3. **Audit trails**: Monitor all operations
4. **Backup**: Regular backup to different region 