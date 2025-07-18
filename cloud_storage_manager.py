"""
Cloud Storage Manager for AI Image Labeling Tool
Handles JSON file persistence in Google Cloud Storage
"""

import os
import json
import tempfile
from typing import Dict, List, Optional, Any
from google.cloud import storage
from google.cloud.exceptions import NotFound
import streamlit as st

class CloudStorageManager:
    """Manages file operations with Google Cloud Storage"""
    
    def __init__(self, bucket_name: str = None, project_id: str = None):
        """
        Initialize Cloud Storage Manager
        
        Args:
            bucket_name: GCS bucket name (defaults to environment variable)
            project_id: GCP project ID (defaults to environment variable)
        """
        self.bucket_name = bucket_name or os.getenv('GCS_BUCKET_NAME')
        self.project_id = project_id or os.getenv('GOOGLE_CLOUD_PROJECT')
        
        if not self.bucket_name:
            raise ValueError("GCS_BUCKET_NAME environment variable must be set")
        
        # Initialize storage client
        try:
            self.client = storage.Client(project=self.project_id)
            self.bucket = self.client.bucket(self.bucket_name)
        except Exception as e:
            st.error(f"Failed to initialize Cloud Storage: {e}")
            raise
    
    def upload_json(self, data: Dict[str, Any], file_path: str) -> bool:
        """
        Upload JSON data to Cloud Storage
        
        Args:
            data: JSON data to upload
            file_path: Path within the bucket
            
        Returns:
            True if successful, False otherwise
        """
        try:
            blob = self.bucket.blob(file_path)
            blob.upload_from_string(
                json.dumps(data, indent=2),
                content_type='application/json'
            )
            return True
        except Exception as e:
            st.error(f"Failed to upload {file_path}: {e}")
            return False
    
    def download_json(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Download JSON data from Cloud Storage
        
        Args:
            file_path: Path within the bucket
            
        Returns:
            JSON data or None if failed
        """
        try:
            blob = self.bucket.blob(file_path)
            if not blob.exists():
                return None
            
            content = blob.download_as_text()
            return json.loads(content)
        except Exception as e:
            st.error(f"Failed to download {file_path}: {e}")
            return None
    
    def list_files(self, prefix: str = "") -> List[str]:
        """
        List files in bucket with optional prefix
        
        Args:
            prefix: File prefix to filter by
            
        Returns:
            List of file paths
        """
        try:
            blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)
            return [blob.name for blob in blobs]
        except Exception as e:
            st.error(f"Failed to list files: {e}")
            return []
    
    def delete_file(self, file_path: str) -> bool:
        """
        Delete file from Cloud Storage
        
        Args:
            file_path: Path within the bucket
            
        Returns:
            True if successful, False otherwise
        """
        try:
            blob = self.bucket.blob(file_path)
            blob.delete()
            return True
        except Exception as e:
            st.error(f"Failed to delete {file_path}: {e}")
            return False
    
    def file_exists(self, file_path: str) -> bool:
        """
        Check if file exists in Cloud Storage
        
        Args:
            file_path: Path within the bucket
            
        Returns:
            True if file exists, False otherwise
        """
        try:
            blob = self.bucket.blob(file_path)
            return blob.exists()
        except Exception as e:
            st.error(f"Failed to check file existence: {e}")
            return False
    
    def upload_image(self, image_data: bytes, file_path: str, content_type: str = 'image/jpeg') -> bool:
        """
        Upload image data to Cloud Storage
        
        Args:
            image_data: Image bytes
            file_path: Path within the bucket
            content_type: MIME type of the image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            blob = self.bucket.blob(file_path)
            blob.upload_from_string(
                image_data,
                content_type=content_type
            )
            return True
        except Exception as e:
            st.error(f"Failed to upload image {file_path}: {e}")
            return False
    
    def download_image(self, file_path: str) -> Optional[bytes]:
        """
        Download image data from Cloud Storage
        
        Args:
            file_path: Path within the bucket
            
        Returns:
            Image bytes or None if failed
        """
        try:
            blob = self.bucket.blob(file_path)
            if not blob.exists():
                return None
            
            return blob.download_as_bytes()
        except Exception as e:
            st.error(f"Failed to download image {file_path}: {e}")
            return None
    
    def get_public_url(self, file_path: str) -> Optional[str]:
        """
        Get public URL for a file (if bucket is public)
        
        Args:
            file_path: Path within the bucket
            
        Returns:
            Public URL or None if not available
        """
        try:
            blob = self.bucket.blob(file_path)
            return blob.public_url
        except Exception as e:
            st.error(f"Failed to get public URL: {e}")
            return None

class CloudImageManager:
    """Enhanced image manager with cloud storage support"""
    
    def __init__(self, storage_manager: CloudStorageManager, base_path: str = "annotations"):
        """
        Initialize Cloud Image Manager
        
        Args:
            storage_manager: CloudStorageManager instance
            base_path: Base path for annotations in storage
        """
        self.storage = storage_manager
        self.base_path = base_path
    
    def save_annotation(self, image_path: str, annotation_data: Dict[str, Any]) -> bool:
        """
        Save annotation data for an image
        
        Args:
            image_path: Path or identifier for the image
            annotation_data: Annotation data to save
            
        Returns:
            True if successful, False otherwise
        """
        # Create a unique identifier from the image path
        # This could be a hash, UUID, or sanitized path
        image_id = self._generate_image_id(image_path)
        file_path = f"{self.base_path}/{image_id}.json"
        
        # Add metadata
        annotation_data['metadata'] = {
            'image_path': image_path,
            'image_id': image_id,
            'saved_at': str(datetime.datetime.now()),
            'version': '1.0'
        }
        
        return self.storage.upload_json(annotation_data, file_path)
    
    def _generate_image_id(self, image_path: str) -> str:
        """
        Generate a unique ID for an image path
        
        Args:
            image_path: Original image path or identifier
            
        Returns:
            Unique identifier for the image
        """
        import hashlib
        
        # Create a hash of the image path for consistent ID generation
        hash_object = hashlib.md5(image_path.encode())
        return hash_object.hexdigest()
    
    def load_annotation(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Load annotation data for an image
        
        Args:
            image_path: Path or identifier for the image
            
        Returns:
            Annotation data or None if not found
        """
        image_id = self._generate_image_id(image_path)
        file_path = f"{self.base_path}/{image_id}.json"
        
        return self.storage.download_json(file_path)
    
    def list_annotations(self) -> List[str]:
        """
        List all annotation files
        
        Returns:
            List of annotation file names
        """
        files = self.storage.list_files(prefix=self.base_path)
        return [os.path.basename(f) for f in files if f.endswith('.json')]
    
    def delete_annotation(self, image_path: str) -> bool:
        """
        Delete annotation for an image
        
        Args:
            image_path: Path or identifier for the image
            
        Returns:
            True if successful, False otherwise
        """
        image_id = self._generate_image_id(image_path)
        file_path = f"{self.base_path}/{image_id}.json"
        
        return self.storage.delete_file(file_path)
    
    def export_dataset(self, format_type: str = "coco") -> Optional[Dict[str, Any]]:
        """
        Export all annotations in a specific format
        
        Args:
            format_type: Export format ("coco", "yolo", "csv")
            
        Returns:
            Dataset in specified format or None if failed
        """
        annotations = self.list_annotations()
        dataset = {
            'info': {
                'description': 'AI Image Labeling Dataset',
                'version': '1.0',
                'date_created': str(datetime.datetime.now())
            },
            'images': [],
            'annotations': []
        }
        
        annotation_id = 1
        for ann_file in annotations:
            # Extract image_id from filename (remove .json extension)
            image_id = ann_file.replace('.json', '')
            ann_data = self.storage.download_json(f"{self.base_path}/{ann_file}")
            
            if ann_data:
                # Add image info
                dataset['images'].append({
                    'id': len(dataset['images']) + 1,
                    'file_name': ann_data.get('metadata', {}).get('image_path', f'image_{image_id}'),
                    'image_id': image_id,
                    'width': ann_data.get('image_width', 0),
                    'height': ann_data.get('image_height', 0)
                })
                
                # Add annotations
                for rect in ann_data.get('rects', []):
                    dataset['annotations'].append({
                        'id': annotation_id,
                        'image_id': len(dataset['images']),
                        'category_id': 1,  # Default category
                        'bbox': [rect['left'], rect['top'], rect['width'], rect['height']],
                        'area': rect['width'] * rect['height'],
                        'iscrowd': 0
                    })
                    annotation_id += 1
        
        return dataset
    
    def save_dataset_export(self, dataset: Dict[str, Any], format_type: str) -> bool:
        """
        Save dataset export to cloud storage
        
        Args:
            dataset: Dataset to save
            format_type: Format type for filename
            
        Returns:
            True if successful, False otherwise
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exports/dataset_{format_type}_{timestamp}.json"
        
        return self.storage.upload_json(dataset, filename)

# Import datetime for metadata
import datetime 