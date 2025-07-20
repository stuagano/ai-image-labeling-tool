#!/usr/bin/env python3
"""
Example: Vertex AI Supervised Tuning

This script demonstrates how to use the Vertex AI supervised tuning functionality
to create classification models from annotated images.
"""

import os
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from vertex_tuning_manager import VertexTuningManager
    from local_storage_manager import LocalStorageManager
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please install required dependencies:")
    print("pip install google-cloud-aiplatform pandas tensorflow pillow")
    sys.exit(1)

def load_sample_annotations():
    """Load sample annotations for demonstration."""
    sample_annotations = [
        {
            "image_path": "img_dir/example1.jpg",
            "annotations": [
                {"label": "cat", "x": 100, "y": 100, "width": 200, "height": 150},
                {"label": "cat", "x": 300, "y": 200, "width": 180, "height": 120}
            ]
        },
        {
            "image_path": "img_dir/example2.jpg", 
            "annotations": [
                {"label": "dog", "x": 150, "y": 120, "width": 250, "height": 180}
            ]
        },
        {
            "image_path": "img_dir/example3.jpg",
            "annotations": [
                {"label": "bird", "x": 80, "y": 80, "width": 120, "height": 100}
            ]
        }
    ]
    return sample_annotations

def main():
    """Main demonstration function."""
    print("🤖 Vertex AI Supervised Tuning Example")
    print("=" * 50)
    
    # Check environment variables
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    bucket_name = os.getenv("GCS_BUCKET_NAME")
    
    if not project_id:
        print("❌ GOOGLE_CLOUD_PROJECT environment variable not set")
        print("Please set it with: export GOOGLE_CLOUD_PROJECT='your-project-id'")
        return
    
    if not bucket_name:
        print("❌ GCS_BUCKET_NAME environment variable not set")
        print("Please set it with: export GCS_BUCKET_NAME='your-bucket-name'")
        return
    
    print(f"✅ Using project: {project_id}")
    print(f"✅ Using bucket: {bucket_name}")
    
    try:
        # Initialize Vertex AI manager
        print("\n🔧 Initializing Vertex AI manager...")
        vertex_manager = VertexTuningManager(project_id)
        vertex_manager.set_storage_bucket(bucket_name)
        print("✅ Vertex AI manager initialized")
        
        # Load sample annotations
        print("\n📊 Loading sample annotations...")
        annotations = load_sample_annotations()
        print(f"✅ Loaded {len(annotations)} sample annotations")
        
        # Show annotation statistics
        classes = set()
        total_annotations = 0
        for ann in annotations:
            for obj in ann.get('annotations', []):
                classes.add(obj.get('label', ''))
                total_annotations += 1
        
        print(f"📈 Statistics:")
        print(f"   - Total images: {len(annotations)}")
        print(f"   - Total annotations: {total_annotations}")
        print(f"   - Unique classes: {len(classes)}")
        print(f"   - Classes: {', '.join(classes)}")
        
        # Prepare classification dataset
        print("\n🎯 Preparing classification dataset...")
        dataset_paths = vertex_manager.prepare_classification_dataset(
            annotations,
            output_dir="datasets/sample_classification"
        )
        print("✅ Dataset prepared successfully")
        
        # Show dataset info
        if 'metadata' in dataset_paths:
            metadata = dataset_paths['metadata']
            print(f"📊 Dataset info:")
            print(f"   - Classes: {metadata['classes']}")
            print(f"   - Total images: {metadata['total_images']}")
            print(f"   - Class distribution: {metadata['class_distribution']}")
        
        # Upload dataset to GCS
        print("\n☁️ Uploading dataset to Google Cloud Storage...")
        gcs_paths = vertex_manager.upload_dataset_to_gcs(dataset_paths)
        print("✅ Dataset uploaded to GCS")
        
        for key, path in gcs_paths.items():
            print(f"   - {key}: {path}")
        
        # List existing training jobs
        print("\n📋 Checking existing training jobs...")
        jobs = vertex_manager.list_training_jobs()
        print(f"✅ Found {len(jobs)} existing training jobs")
        
        for job in jobs:
            print(f"   - {job['name']} ({job['type']}): {job['state']}")
        
        # List existing models
        print("\n🤖 Checking existing models...")
        models = vertex_manager.list_models()
        print(f"✅ Found {len(models)} existing models")
        
        for model in models:
            print(f"   - {model['name']} (v{model['version_id']})")
        
        # List existing endpoints
        print("\n🌐 Checking existing endpoints...")
        endpoints = vertex_manager.list_endpoints()
        print(f"✅ Found {len(endpoints)} existing endpoints")
        
        for endpoint in endpoints:
            print(f"   - {endpoint['name']} ({endpoint['deployed_models']} models)")
        
        print("\n🎉 Example completed successfully!")
        print("\nNext steps:")
        print("1. Use the Streamlit app: streamlit run app_vertex_tuning.py")
        print("2. Or use the API endpoints: python api_server.py")
        print("3. Create training jobs and deploy models")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check your Google Cloud credentials")
        print("2. Verify Vertex AI is enabled in your project")
        print("3. Ensure you have the required IAM permissions")
        print("4. Check that your bucket exists and is accessible")

if __name__ == "__main__":
    main() 