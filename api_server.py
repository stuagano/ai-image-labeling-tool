"""
FastAPI Server for Cloud Storage Operations
Provides REST endpoints for the local Streamlit app to interact with Google Cloud Storage
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import json
import tempfile
import io
from PIL import Image
import hashlib
import datetime

# Import our storage modules
from cloud_storage_manager import CloudStorageManager, CloudImageManager
from local_storage_manager import LocalStorageManager, LocalImageManager

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI"""
    # Startup
    if not initialize_storage():
        print("Warning: No storage initialized. Check configuration.")
    yield
    # Shutdown
    pass

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="AI Image Labeling API",
    description="REST API for cloud storage operations",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class AnnotationData(BaseModel):
    rects: List[Dict[str, Any]]
    image_path: str
    image_width: int
    image_height: int
    labels: List[str]
    format: str

class DescriptionData(BaseModel):
    image_name: str
    description: str
    timestamp: str
    source: str = "gemini"
    metadata: Dict[str, Any]

class BatchProcessRequest(BaseModel):
    image_paths: List[str]
    labels: List[str]
    format: str = "json"
    confidence_threshold: float = 0.5

# Global storage managers
cloud_storage = None
cloud_manager = None
local_storage = None
local_manager = None

def initialize_storage():
    """Initialize storage connection (cloud or local)"""
    global cloud_storage, cloud_manager, local_storage, local_manager
    
    # Try cloud storage first
    try:
        bucket_name = os.getenv('GCS_BUCKET_NAME')
        if bucket_name:
            cloud_storage = CloudStorageManager(bucket_name)
            cloud_manager = CloudImageManager(cloud_storage)
            print("✅ Cloud storage initialized")
            return True
    except Exception as e:
        print(f"Failed to initialize cloud storage: {e}")
    
    # Fall back to local storage
    try:
        local_storage = LocalStorageManager()
        local_manager = LocalImageManager(local_storage)
        print("✅ Local storage initialized")
        return True
    except Exception as e:
        print(f"Failed to initialize local storage: {e}")
        return False

@app.get("/")
async def root():
    """Health check endpoint"""
    storage_type = "cloud" if cloud_storage else "local" if local_storage else "none"
    return {
        "status": "healthy",
        "service": "AI Image Labeling API",
        "storage": storage_type,
        "storage_connected": bool(cloud_storage or local_storage)
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    storage_type = "cloud" if cloud_storage else "local" if local_storage else "none"
    return {
        "status": "healthy",
        "timestamp": str(datetime.datetime.now()),
        "storage_type": storage_type,
        "storage_connected": bool(cloud_storage or local_storage),
        "bucket_name": os.getenv('GCS_BUCKET_NAME', 'not_set')
    }

@app.post("/upload-image")
async def upload_image(
    file: UploadFile = File(...),
    image_path: str = Form(...)
):
    """Upload an image to cloud storage"""
    if not cloud_storage:
        raise HTTPException(status_code=500, detail="Cloud storage not initialized")
    
    try:
        # Read image data
        image_data = await file.read()
        
        # Determine content type
        content_type = file.content_type or 'image/jpeg'
        
        # Upload to cloud storage
        success = cloud_storage.upload_image(
            image_data, 
            f"uploads/{image_path}", 
            content_type
        )
        
        if success:
            return {
                "status": "success",
                "message": f"Image {image_path} uploaded successfully",
                "image_path": image_path
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to upload image")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/save-annotation")
async def save_annotation(annotation: AnnotationData):
    """Save annotation data to storage"""
    if not cloud_manager and not local_manager:
        raise HTTPException(status_code=500, detail="No storage initialized")
    
    try:
        # Generate file identifier for stateless service
        file_identifier = f"{annotation.image_path}_{annotation.image_width}_{annotation.image_height}"
        
        # Add metadata
        annotation_data = annotation.dict()
        annotation_data['metadata'] = {
            'image_path': annotation.image_path,
            'image_id': hashlib.md5(file_identifier.encode()).hexdigest(),
            'saved_at': str(datetime.datetime.now()),
            'version': '1.0'
        }
        
        # Save annotation using available storage
        if cloud_manager:
            success = cloud_manager.save_annotation(file_identifier, annotation_data)
        else:
            success = local_manager.save_annotation(file_identifier, annotation_data)
        
        if success:
            storage_type = "cloud" if cloud_manager else "local"
            return {
                "status": "success",
                "message": f"Annotation for {annotation.image_path} saved to {storage_type} storage",
                "image_path": annotation.image_path,
                "storage_type": storage_type
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to save annotation")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save failed: {str(e)}")

@app.get("/load-annotation/{image_path:path}")
async def load_annotation(image_path: str):
    """Load annotation data from storage"""
    if not cloud_manager and not local_manager:
        raise HTTPException(status_code=500, detail="No storage initialized")
    
    try:
        # Try cloud storage first, then local storage
        annotation = None
        storage_type = "none"
        
        if cloud_manager:
            annotation = cloud_manager.load_annotation(image_path)
            if annotation:
                storage_type = "cloud"
        
        if not annotation and local_manager:
            annotation = local_manager.load_annotation(image_path)
            if annotation:
                storage_type = "local"
        
        if annotation:
            return {
                "status": "success",
                "annotation": annotation,
                "image_path": image_path,
                "storage_type": storage_type
            }
        else:
            return {
                "status": "not_found",
                "message": f"No annotation found for {image_path}",
                "image_path": image_path
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Load failed: {str(e)}")

@app.post("/save-description")
async def save_description(description: DescriptionData):
    """Save Gemini description to cloud storage"""
    if not cloud_storage:
        raise HTTPException(status_code=500, detail="Cloud storage not initialized")
    
    try:
        # Create file identifier
        file_identifier = f"{description.image_name}_{len(description.description)}"
        file_path = f"descriptions/{file_identifier}.json"
        
        # Save description
        success = cloud_storage.upload_json(description.dict(), file_path)
        
        if success:
            return {
                "status": "success",
                "message": f"Description for {description.image_name} saved successfully",
                "image_name": description.image_name
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to save description")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save failed: {str(e)}")

@app.get("/list-annotations")
async def list_annotations():
    """List all annotations in storage"""
    if not cloud_manager and not local_manager:
        raise HTTPException(status_code=500, detail="No storage initialized")
    
    try:
        annotations = []
        storage_type = "none"
        
        if cloud_manager:
            annotations = cloud_manager.list_annotations()
            storage_type = "cloud"
        elif local_manager:
            annotations = local_manager.list_annotations()
            storage_type = "local"
        
        return {
            "status": "success",
            "annotations": annotations,
            "count": len(annotations),
            "storage_type": storage_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"List failed: {str(e)}")

@app.get("/list-descriptions")
async def list_descriptions():
    """List all descriptions in cloud storage"""
    if not cloud_storage:
        raise HTTPException(status_code=500, detail="Cloud storage not initialized")
    
    try:
        descriptions = cloud_storage.list_files(prefix="descriptions/")
        return {
            "status": "success",
            "descriptions": descriptions,
            "count": len(descriptions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"List failed: {str(e)}")

@app.delete("/delete-annotation/{image_path:path}")
async def delete_annotation(image_path: str):
    """Delete annotation from cloud storage"""
    if not cloud_manager:
        raise HTTPException(status_code=500, detail="Cloud storage not initialized")
    
    try:
        success = cloud_manager.delete_annotation(image_path)
        
        if success:
            return {
                "status": "success",
                "message": f"Annotation for {image_path} deleted successfully",
                "image_path": image_path
            }
        else:
            raise HTTPException(status_code=404, detail="Annotation not found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

@app.post("/export-dataset")
async def export_dataset(format_type: str = "coco"):
    """Export dataset in specified format"""
    if not cloud_manager:
        raise HTTPException(status_code=500, detail="Cloud storage not initialized")
    
    try:
        dataset = cloud_manager.export_dataset(format_type)
        
        if dataset:
            # Save export to cloud storage
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            export_filename = f"dataset_{format_type}_{timestamp}.json"
            export_path = f"exports/{export_filename}"
            
            success = cloud_storage.upload_json(dataset, export_path)
            
            if success:
                return {
                    "status": "success",
                    "message": f"Dataset exported as {format_type}",
                    "export_path": export_path,
                    "dataset": dataset
                }
            else:
                raise HTTPException(status_code=500, detail="Failed to save export")
        else:
            raise HTTPException(status_code=404, detail="No annotations to export")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@app.post("/batch-process")
async def batch_process(request: BatchProcessRequest):
    """Process multiple images in batch"""
    if not cloud_manager:
        raise HTTPException(status_code=500, detail="Cloud storage not initialized")
    
    try:
        results = []
        
        for image_path in request.image_paths:
            try:
                # Load existing annotation
                annotation = cloud_manager.load_annotation(image_path)
                
                if annotation:
                    results.append({
                        "image_path": image_path,
                        "status": "loaded",
                        "annotation_count": len(annotation.get('rects', []))
                    })
                else:
                    results.append({
                        "image_path": image_path,
                        "status": "not_found"
                    })
                    
            except Exception as e:
                results.append({
                    "image_path": image_path,
                    "status": "error",
                    "error": str(e)
                })
        
        return {
            "status": "success",
            "processed_count": len(results),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@app.get("/statistics")
async def get_statistics():
    """Get storage statistics"""
    if not cloud_storage and not local_storage:
        raise HTTPException(status_code=500, detail="No storage initialized")
    
    try:
        storage_type = "none"
        annotations = []
        descriptions = []
        exports = []
        uploads = []
        
        if cloud_storage and cloud_manager:
            storage_type = "cloud"
            annotations = cloud_manager.list_annotations()
            descriptions = cloud_storage.list_files(prefix="descriptions/")
            exports = cloud_storage.list_files(prefix="exports/")
            uploads = cloud_storage.list_files(prefix="uploads/")
        elif local_storage and local_manager:
            storage_type = "local"
            annotations = local_manager.list_annotations()
            descriptions = local_storage.list_files(prefix="descriptions/")
            exports = local_storage.list_files(prefix="exports/")
            uploads = local_storage.list_files(prefix="uploads/")
        
        return {
            "status": "success",
            "storage_type": storage_type,
            "statistics": {
                "annotations": len(annotations),
                "descriptions": len(descriptions),
                "exports": len(exports),
                "uploads": len(uploads),
                "total_files": len(annotations) + len(descriptions) + len(exports) + len(uploads)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Statistics failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Run the FastAPI server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 