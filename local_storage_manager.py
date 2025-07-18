"""
Local Storage Manager for Annotations
Saves annotations to local JSON files instead of cloud storage
"""

import os
import json
import datetime
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path

class LocalStorageManager:
    """Manages local file storage for annotations"""
    
    def __init__(self, base_dir: str = "local_annotations"):
        """
        Initialize local storage manager
        
        Args:
            base_dir: Base directory for storing annotations
        """
        self.base_dir = Path(base_dir)
        self.annotations_dir = self.base_dir / "annotations"
        self.descriptions_dir = self.base_dir / "descriptions"
        self.exports_dir = self.base_dir / "exports"
        self.uploads_dir = self.base_dir / "uploads"
        
        # Create directories if they don't exist
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        self.descriptions_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
    
    def upload_json(self, data: Dict[str, Any], file_path: str) -> bool:
        """
        Save JSON data to local file
        
        Args:
            data: Data to save
            file_path: Relative path within storage directory
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Determine which directory to use based on file path
            if file_path.startswith("annotations/"):
                full_path = self.annotations_dir / file_path.replace("annotations/", "")
            elif file_path.startswith("descriptions/"):
                full_path = self.descriptions_dir / file_path.replace("descriptions/", "")
            elif file_path.startswith("exports/"):
                full_path = self.exports_dir / file_path.replace("exports/", "")
            elif file_path.startswith("uploads/"):
                full_path = self.uploads_dir / file_path.replace("uploads/", "")
            else:
                full_path = self.base_dir / file_path
            
            # Ensure directory exists
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save JSON file
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"Failed to save JSON to {file_path}: {e}")
            return False
    
    def download_json(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load JSON data from local file
        
        Args:
            file_path: Relative path within storage directory
            
        Returns:
            Data if successful, None otherwise
        """
        try:
            # Determine which directory to use based on file path
            if file_path.startswith("annotations/"):
                full_path = self.annotations_dir / file_path.replace("annotations/", "")
            elif file_path.startswith("descriptions/"):
                full_path = self.descriptions_dir / file_path.replace("descriptions/", "")
            elif file_path.startswith("exports/"):
                full_path = self.exports_dir / file_path.replace("exports/", "")
            elif file_path.startswith("uploads/"):
                full_path = self.uploads_dir / file_path.replace("uploads/", "")
            else:
                full_path = self.base_dir / file_path
            
            if not full_path.exists():
                return None
            
            # Load JSON file
            with open(full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            print(f"Failed to load JSON from {file_path}: {e}")
            return None
    
    def list_files(self, prefix: str = "") -> List[str]:
        """
        List files in storage directory
        
        Args:
            prefix: File prefix to filter by
            
        Returns:
            List of file paths
        """
        try:
            files = []
            
            # Determine which directory to search
            if prefix.startswith("annotations/"):
                search_dir = self.annotations_dir
                prefix = prefix.replace("annotations/", "")
            elif prefix.startswith("descriptions/"):
                search_dir = self.descriptions_dir
                prefix = prefix.replace("descriptions/", "")
            elif prefix.startswith("exports/"):
                search_dir = self.exports_dir
                prefix = prefix.replace("exports/", "")
            elif prefix.startswith("uploads/"):
                search_dir = self.uploads_dir
                prefix = prefix.replace("uploads/", "")
            else:
                search_dir = self.base_dir
            
            # List files
            for file_path in search_dir.rglob("*"):
                if file_path.is_file():
                    rel_path = str(file_path.relative_to(search_dir))
                    if not prefix or rel_path.startswith(prefix):
                        files.append(rel_path)
            
            return files
            
        except Exception as e:
            print(f"Failed to list files with prefix {prefix}: {e}")
            return []
    
    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file from storage
        
        Args:
            file_path: Relative path within storage directory
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Determine which directory to use based on file path
            if file_path.startswith("annotations/"):
                full_path = self.annotations_dir / file_path.replace("annotations/", "")
            elif file_path.startswith("descriptions/"):
                full_path = self.descriptions_dir / file_path.replace("descriptions/", "")
            elif file_path.startswith("exports/"):
                full_path = self.exports_dir / file_path.replace("exports/", "")
            elif file_path.startswith("uploads/"):
                full_path = self.uploads_dir / file_path.replace("uploads/", "")
            else:
                full_path = self.base_dir / file_path
            
            if full_path.exists():
                full_path.unlink()
                return True
            else:
                return False
                
        except Exception as e:
            print(f"Failed to delete file {file_path}: {e}")
            return False

class LocalImageManager:
    """Manages image annotations using local storage"""
    
    def __init__(self, storage: LocalStorageManager):
        """
        Initialize local image manager
        
        Args:
            storage: Local storage manager instance
        """
        self.storage = storage
        self.base_path = "annotations"
    
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
    
    def load_annotation(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Load annotation data for an image
        
        Args:
            image_path: Path or identifier for the image
            
        Returns:
            Annotation data if found, None otherwise
        """
        # Try to find annotation by image path
        image_id = self._generate_image_id(image_path)
        file_path = f"{self.base_path}/{image_id}.json"
        
        return self.storage.download_json(file_path)
    
    def delete_annotation(self, image_path: str) -> bool:
        """
        Delete annotation data for an image
        
        Args:
            image_path: Path or identifier for the image
            
        Returns:
            True if successful, False otherwise
        """
        image_id = self._generate_image_id(image_path)
        file_path = f"{self.base_path}/{image_id}.json"
        
        return self.storage.delete_file(file_path)
    
    def list_annotations(self) -> List[str]:
        """
        List all annotation files
        
        Returns:
            List of annotation file paths
        """
        return self.storage.list_files(prefix=self.base_path)
    
    def export_dataset(self, format_type: str = "coco") -> Optional[Dict[str, Any]]:
        """
        Export all annotations as a dataset
        
        Args:
            format_type: Export format (coco, json)
            
        Returns:
            Dataset data if successful, None otherwise
        """
        try:
            annotations = []
            
            # Load all annotation files
            for file_path in self.storage.list_files(prefix=self.base_path):
                data = self.storage.download_json(f"{self.base_path}/{file_path}")
                if data:
                    annotations.append(data)
            
            if not annotations:
                return None
            
            # Convert to requested format
            if format_type.lower() == "coco":
                return self._convert_to_coco(annotations)
            else:
                return {
                    "annotations": annotations,
                    "format": "json",
                    "exported_at": str(datetime.datetime.now()),
                    "total_annotations": len(annotations)
                }
                
        except Exception as e:
            print(f"Failed to export dataset: {e}")
            return None
    
    def _generate_image_id(self, image_path: str) -> str:
        """
        Generate a unique ID for an image path
        
        Args:
            image_path: Original image path or identifier
            
        Returns:
            Unique identifier for the image
        """
        # Create a hash of the image path for consistent ID generation
        hash_object = hashlib.md5(image_path.encode())
        return hash_object.hexdigest()
    
    def _convert_to_coco(self, annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Convert annotations to COCO format
        
        Args:
            annotations: List of annotation data
            
        Returns:
            COCO format dataset
        """
        coco_data = {
            "info": {
                "description": "AI Image Labeling Tool Dataset",
                "url": "",
                "version": "1.0",
                "year": datetime.datetime.now().year,
                "contributor": "",
                "date_created": str(datetime.datetime.now())
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Track categories
        categories = {}
        category_id = 1
        
        # Process each annotation
        for i, ann_data in enumerate(annotations):
            # Add image info
            image_info = {
                "id": i + 1,
                "file_name": ann_data.get('image_path', f"image_{i+1}.jpg"),
                "width": ann_data.get('image_width', 0),
                "height": ann_data.get('image_height', 0)
            }
            coco_data["images"].append(image_info)
            
            # Process bounding boxes
            for j, rect in enumerate(ann_data.get('rects', [])):
                label = rect.get('label', 'unknown')
                
                # Add category if not exists
                if label not in categories:
                    categories[label] = category_id
                    coco_data["categories"].append({
                        "id": category_id,
                        "name": label,
                        "supercategory": "object"
                    })
                    category_id += 1
                
                # Add annotation
                annotation = {
                    "id": len(coco_data["annotations"]) + 1,
                    "image_id": i + 1,
                    "category_id": categories[label],
                    "bbox": [
                        rect.get('left', 0),
                        rect.get('top', 0),
                        rect.get('width', 0),
                        rect.get('height', 0)
                    ],
                    "area": rect.get('width', 0) * rect.get('height', 0),
                    "iscrowd": 0
                }
                coco_data["annotations"].append(annotation)
        
        return coco_data 