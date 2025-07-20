"""
Vertex AI Supervised Tuning Manager

This module handles the creation and management of supervised tuning jobs
for image classification models that can be deployed to Google Vertex AI.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import tempfile
import zipfile
from pathlib import Path

try:
    from google.cloud import aiplatform
    from google.cloud import storage
    from google.cloud.aiplatform import Model
    from google.cloud.aiplatform import Endpoint
    from google.cloud.aiplatform import TrainingJob
    VERTEX_AVAILABLE = True
except ImportError:
    VERTEX_AVAILABLE = False
    logging.warning("Google Cloud AI Platform not available. Install with: pip install google-cloud-aiplatform")

try:
    import pandas as pd
    import numpy as np
    from PIL import Image
    import tensorflow as tf
    from tensorflow import keras
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logging.warning("Pandas/TensorFlow not available. Install with: pip install pandas tensorflow pillow")

class VertexTuningManager:
    """Manages supervised tuning jobs for Vertex AI image classification."""
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        """
        Initialize the Vertex Tuning Manager.
        
        Args:
            project_id: Google Cloud project ID
            location: GCP region for Vertex AI
        """
        self.project_id = project_id
        self.location = location
        self.bucket_name = None
        self.storage_client = None
        
        if VERTEX_AVAILABLE:
            aiplatform.init(project=project_id, location=location)
            self.storage_client = storage.Client(project=project_id)
        else:
            raise ImportError("Google Cloud AI Platform is required for Vertex tuning")
    
    def set_storage_bucket(self, bucket_name: str):
        """Set the Cloud Storage bucket for storing datasets and models."""
        self.bucket_name = bucket_name
    
    def prepare_classification_dataset(
        self, 
        annotations: List[Dict], 
        output_dir: str = "datasets"
    ) -> Dict[str, str]:
        """
        Prepare a classification dataset from annotations.
        
        Args:
            annotations: List of annotation dictionaries
            output_dir: Directory to save the dataset
            
        Returns:
            Dictionary with dataset paths and metadata
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("Pandas and TensorFlow are required for dataset preparation")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Group images by class
        class_images = {}
        for annotation in annotations:
            image_path = annotation.get('image_path', '')
            if not image_path or not os.path.exists(image_path):
                continue
                
            # Extract class from annotations
            classes = set()
            for obj in annotation.get('annotations', []):
                label = obj.get('label', '').strip()
                if label:
                    classes.add(label)
            
            # Use first class or 'unknown' if no classes
            class_name = list(classes)[0] if classes else 'unknown'
            
            if class_name not in class_images:
                class_images[class_name] = []
            class_images[class_name].append(image_path)
        
        # Create dataset structure
        dataset_info = {
            'classes': list(class_images.keys()),
            'total_images': sum(len(images) for images in class_images.values()),
            'class_distribution': {cls: len(images) for cls, images in class_images.items()},
            'created_at': datetime.now().isoformat()
        }
        
        # Save dataset info
        dataset_info_path = os.path.join(output_dir, 'dataset_info.json')
        with open(dataset_info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        # Create training/validation split
        train_data = []
        val_data = []
        
        for class_name, images in class_images.items():
            # Shuffle images
            np.random.shuffle(images)
            
            # Split 80/20
            split_idx = int(len(images) * 0.8)
            train_images = images[:split_idx]
            val_images = images[split_idx:]
            
            for img_path in train_images:
                train_data.append({
                    'image_path': img_path,
                    'class': class_name,
                    'class_id': dataset_info['classes'].index(class_name)
                })
            
            for img_path in val_images:
                val_data.append({
                    'image_path': img_path,
                    'class': class_name,
                    'class_id': dataset_info['classes'].index(class_name)
                })
        
        # Save splits
        train_df = pd.DataFrame(train_data)
        val_df = pd.DataFrame(val_data)
        
        train_path = os.path.join(output_dir, 'train.csv')
        val_path = os.path.join(output_dir, 'validation.csv')
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        
        return {
            'dataset_info': dataset_info_path,
            'train_csv': train_path,
            'validation_csv': val_path,
            'output_dir': output_dir,
            'metadata': dataset_info
        }
    
    def upload_dataset_to_gcs(self, dataset_paths: Dict[str, str]) -> Dict[str, str]:
        """
        Upload dataset to Google Cloud Storage.
        
        Args:
            dataset_paths: Dictionary with local dataset paths
            
        Returns:
            Dictionary with GCS paths
        """
        if not self.bucket_name:
            raise ValueError("Storage bucket not set. Call set_storage_bucket() first.")
        
        bucket = self.storage_client.bucket(self.bucket_name)
        gcs_paths = {}
        
        # Upload dataset files
        for key, local_path in dataset_paths.items():
            if key == 'metadata':
                continue
                
            if os.path.exists(local_path):
                blob_name = f"datasets/{os.path.basename(local_path)}"
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(local_path)
                gcs_paths[key] = f"gs://{self.bucket_name}/{blob_name}"
        
        return gcs_paths
    
    def create_custom_training_job(
        self,
        dataset_gcs_paths: Dict[str, str],
        job_name: str,
        model_type: str = "efficientnet",
        hyperparameters: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Create a custom training job for image classification.
        
        Args:
            dataset_gcs_paths: GCS paths to dataset files
            job_name: Name for the training job
            model_type: Type of model to train
            hyperparameters: Training hyperparameters
            
        Returns:
            Training job information
        """
        # Default hyperparameters
        default_hparams = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 50,
            'image_size': 224,
            'dropout_rate': 0.2
        }
        
        if hyperparameters:
            default_hparams.update(hyperparameters)
        
        # Create training job
        job = aiplatform.CustomTrainingJob(
            display_name=job_name,
            script_path="gs://your-bucket/training_scripts/train_classifier.py",
            container_uri="gcr.io/cloud-aiplatform/training/tf-cpu.2-8:latest",
            model_serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/tf2-cpu.2-8:latest",
            requirements=["tensorflow==2.8.0", "pillow", "pandas"],
            args=[
                f"--train_csv={dataset_gcs_paths['train_csv']}",
                f"--validation_csv={dataset_gcs_paths['validation_csv']}",
                f"--model_type={model_type}",
                f"--learning_rate={default_hparams['learning_rate']}",
                f"--batch_size={default_hparams['batch_size']}",
                f"--epochs={default_hparams['epochs']}",
                f"--image_size={default_hparams['image_size']}",
                f"--dropout_rate={default_hparams['dropout_rate']}"
            ]
        )
        
        # Start training
        model = job.run(
            dataset=aiplatform.TabularDataset.create(
                display_name=f"{job_name}_dataset",
                gcs_source=[dataset_gcs_paths['train_csv']]
            ),
            model_display_name=f"{job_name}_model",
            training_fraction_split=0.8,
            validation_fraction_split=0.2,
            test_fraction_split=0.0
        )
        
        return {
            'job_name': job_name,
            'model': model,
            'hyperparameters': default_hparams,
            'status': 'RUNNING'
        }
    
    def create_automl_training_job(
        self,
        dataset_gcs_paths: Dict[str, str],
        job_name: str,
        budget_milli_node_hours: int = 1000
    ) -> Dict[str, Any]:
        """
        Create an AutoML training job for image classification.
        
        Args:
            dataset_gcs_paths: GCS paths to dataset files
            job_name: Name for the training job
            budget_milli_node_hours: Training budget in milli node hours
            
        Returns:
            Training job information
        """
        # Create AutoML training job
        job = aiplatform.AutoMLImageTrainingJob(
            display_name=job_name,
            prediction_type="classification",
            model_type="CLOUD",
            base_model=None,
            budget_milli_node_hours=budget_milli_node_hours
        )
        
        # Start training
        model = job.run(
            dataset=aiplatform.ImageDataset.create(
                display_name=f"{job_name}_dataset",
                gcs_source=dataset_gcs_paths['train_csv']
            ),
            model_display_name=f"{job_name}_model",
            training_fraction_split=0.8,
            validation_fraction_split=0.2,
            test_fraction_split=0.0
        )
        
        return {
            'job_name': job_name,
            'model': model,
            'budget_milli_node_hours': budget_milli_node_hours,
            'status': 'RUNNING'
        }
    
    def deploy_model_to_endpoint(
        self,
        model: Model,
        endpoint_name: str,
        machine_type: str = "n1-standard-2"
    ) -> Dict[str, Any]:
        """
        Deploy a trained model to a Vertex AI endpoint.
        
        Args:
            model: Trained Vertex AI model
            endpoint_name: Name for the endpoint
            machine_type: Machine type for deployment
            
        Returns:
            Endpoint information
        """
        # Create endpoint
        endpoint = model.deploy(
            deployed_model_display_name=endpoint_name,
            machine_type=machine_type,
            accelerator_type=None,
            accelerator_count=None,
            starting_replica_count=1,
            max_replica_count=10,
            sync=True
        )
        
        return {
            'endpoint_name': endpoint_name,
            'endpoint': endpoint,
            'endpoint_url': endpoint.resource_name,
            'status': 'DEPLOYED'
        }
    
    def list_training_jobs(self) -> List[Dict[str, Any]]:
        """List all training jobs in the project."""
        jobs = []
        
        # List custom training jobs
        custom_jobs = aiplatform.CustomTrainingJob.list()
        for job in custom_jobs:
            jobs.append({
                'name': job.display_name,
                'type': 'custom',
                'state': job.state.name,
                'create_time': job.create_time,
                'end_time': job.end_time
            })
        
        # List AutoML jobs
        automl_jobs = aiplatform.AutoMLImageTrainingJob.list()
        for job in automl_jobs:
            jobs.append({
                'name': job.display_name,
                'type': 'automl',
                'state': job.state.name,
                'create_time': job.create_time,
                'end_time': job.end_time
            })
        
        return jobs
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all trained models in the project."""
        models = aiplatform.Model.list()
        
        return [{
            'name': model.display_name,
            'model_id': model.name,
            'create_time': model.create_time,
            'update_time': model.update_time,
            'version_id': model.version_id
        } for model in models]
    
    def list_endpoints(self) -> List[Dict[str, Any]]:
        """List all deployed endpoints in the project."""
        endpoints = aiplatform.Endpoint.list()
        
        return [{
            'name': endpoint.display_name,
            'endpoint_id': endpoint.name,
            'deployed_models': len(endpoint.list_models()),
            'create_time': endpoint.create_time
        } for endpoint in endpoints]
    
    def predict_with_endpoint(
        self,
        endpoint: Endpoint,
        image_path: str
    ) -> Dict[str, Any]:
        """
        Make a prediction using a deployed endpoint.
        
        Args:
            endpoint: Deployed Vertex AI endpoint
            image_path: Path to image file
            
        Returns:
            Prediction results
        """
        # Load and preprocess image
        image = Image.open(image_path)
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Make prediction
        prediction = endpoint.predict(instances=image_array.tolist())
        
        return {
            'predictions': prediction.predictions,
            'confidence_scores': prediction.predictions[0] if prediction.predictions else []
        }
    
    def delete_endpoint(self, endpoint: Endpoint):
        """Delete a deployed endpoint."""
        endpoint.delete()
    
    def delete_model(self, model: Model):
        """Delete a trained model."""
        model.delete() 