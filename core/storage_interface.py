"""
Unified Storage Interface for Image Labeling Application
Eliminates DRY violations between CloudStorageManager and LocalStorageManager
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import json
import os
from pathlib import Path


class StorageManagerInterface(ABC):
    """Abstract base class for storage managers"""
    
    def __init__(self):
        """Initialize storage manager"""
        pass
    
    @abstractmethod
    def upload_json(self, data: Dict[str, Any], file_path: str) -> bool:
        """
        Upload JSON data to storage
        
        Args:
            data: JSON data to upload
            file_path: Path within storage
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def download_json(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Download JSON data from storage
        
        Args:
            file_path: Path within storage
            
        Returns:
            Data if successful, None otherwise
        """
        pass
    
    @abstractmethod
    def list_files(self, prefix: str = "") -> List[str]:
        """
        List files in storage with optional prefix filter
        
        Args:
            prefix: Optional prefix to filter files
            
        Returns:
            List of file paths
        """
        pass
    
    @abstractmethod
    def file_exists(self, file_path: str) -> bool:
        """
        Check if file exists in storage
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file exists, False otherwise
        """
        pass
    
    @abstractmethod
    def delete_file(self, file_path: str) -> bool:
        """
        Delete file from storage
        
        Args:
            file_path: Path to delete
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    def save_annotation(self, annotation_data: Dict[str, Any]) -> bool:
        """
        Save annotation data (common logic)
        
        Args:
            annotation_data: Annotation data to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            image_path = annotation_data.get('image_path', '')
            if not image_path:
                raise ValueError("Image path is required")
            
            # Generate annotation file path
            annotation_path = self._generate_annotation_path(image_path)
            
            # Save annotation
            return self.upload_json(annotation_data, annotation_path)
            
        except Exception as e:
            print(f"Failed to save annotation: {e}")
            return False
    
    def load_annotation(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Load annotation for an image (common logic)
        
        Args:
            image_path: Path to the image
            
        Returns:
            Annotation data if found, None otherwise
        """
        try:
            annotation_path = self._generate_annotation_path(image_path)
            return self.download_json(annotation_path)
        except Exception as e:
            print(f"Failed to load annotation: {e}")
            return None
    
    def _generate_annotation_path(self, image_path: str) -> str:
        """
        Generate annotation file path from image path
        
        Args:
            image_path: Path to the image
            
        Returns:
            Annotation file path
        """
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        return f"annotations/{base_name}.json"
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get storage statistics (common logic)
        
        Returns:
            Dictionary with storage statistics
        """
        try:
            annotation_files = self.list_files("annotations/")
            description_files = self.list_files("descriptions/")
            
            return {
                "total_annotations": len(annotation_files),
                "total_descriptions": len(description_files),
                "storage_type": self.__class__.__name__
            }
        except Exception as e:
            return {
                "error": str(e),
                "storage_type": self.__class__.__name__
            }


class BaseImageManager(ABC):
    """Abstract base class for image managers"""
    
    def __init__(self, storage_manager: StorageManagerInterface):
        """
        Initialize image manager
        
        Args:
            storage_manager: Storage manager instance
        """
        self.storage = storage_manager
    
    @abstractmethod
    def save_annotation(self, image_path: str, annotations: List[Dict[str, Any]]) -> bool:
        """
        Save annotations for an image
        
        Args:
            image_path: Path to the image
            annotations: List of annotation dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def load_annotation(self, image_path: str) -> Optional[List[Dict[str, Any]]]:
        """
        Load annotations for an image
        
        Args:
            image_path: Path to the image
            
        Returns:
            List of annotations if found, None otherwise
        """
        pass
    
    def save_description(self, image_path: str, description: str) -> bool:
        """
        Save description for an image (common logic)
        
        Args:
            image_path: Path to the image
            description: Description text
            
        Returns:
            True if successful, False otherwise
        """
        description_data = {
            "image_path": image_path,
            "description": description,
            "timestamp": self._get_timestamp()
        }
        
        description_path = self._generate_description_path(image_path)
        return self.storage.upload_json(description_data, description_path)
    
    def load_description(self, image_path: str) -> Optional[str]:
        """
        Load description for an image (common logic)
        
        Args:
            image_path: Path to the image
            
        Returns:
            Description if found, None otherwise
        """
        description_path = self._generate_description_path(image_path)
        data = self.storage.download_json(description_path)
        return data.get("description") if data else None
    
    def _generate_description_path(self, image_path: str) -> str:
        """
        Generate description file path from image path
        
        Args:
            image_path: Path to the image
            
        Returns:
            Description file path
        """
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        return f"descriptions/{base_name}.json"
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as string"""
        import datetime
        return datetime.datetime.now().isoformat()


class StorageUtilities:
    """Common utility functions for storage operations"""
    
    @staticmethod
    def validate_json_data(data: Any) -> bool:
        """
        Validate JSON serializable data
        
        Args:
            data: Data to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            json.dumps(data)
            return True
        except (TypeError, ValueError):
            return False
    
    @staticmethod
    def sanitize_file_path(file_path: str) -> str:
        """
        Sanitize file path for storage
        
        Args:
            file_path: Raw file path
            
        Returns:
            Sanitized file path
        """
        # Remove leading/trailing whitespace and slashes
        sanitized = file_path.strip().strip('/')
        
        # Replace backslashes with forward slashes
        sanitized = sanitized.replace('\\', '/')
        
        # Remove any double slashes
        while '//' in sanitized:
            sanitized = sanitized.replace('//', '/')
        
        return sanitized
    
    @staticmethod
    def generate_unique_filename(base_name: str, extension: str, existing_files: List[str]) -> str:
        """
        Generate unique filename to avoid conflicts
        
        Args:
            base_name: Base name without extension
            extension: File extension
            existing_files: List of existing file names
            
        Returns:
            Unique filename
        """
        filename = f"{base_name}.{extension}"
        counter = 1
        
        while filename in existing_files:
            filename = f"{base_name}_{counter}.{extension}"
            counter += 1
        
        return filename
    
    @staticmethod
    def batch_operation(operation_func, items: List[Any], **kwargs) -> Dict[str, Any]:
        """
        Perform batch operation with error handling
        
        Args:
            operation_func: Function to apply to each item
            items: List of items to process
            **kwargs: Additional arguments for operation_func
            
        Returns:
            Dictionary with operation results
        """
        results = {
            "successful": [],
            "failed": [],
            "total": len(items),
            "success_count": 0,
            "error_count": 0
        }
        
        for item in items:
            try:
                result = operation_func(item, **kwargs)
                results["successful"].append({"item": item, "result": result})
                results["success_count"] += 1
            except Exception as e:
                results["failed"].append({"item": item, "error": str(e)})
                results["error_count"] += 1
        
        return results