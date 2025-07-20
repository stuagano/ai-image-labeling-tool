"""
Streamlit App for Vertex AI Supervised Tuning

This app provides a user-friendly interface for creating and managing
supervised tuning jobs for image classification models on Google Vertex AI.
"""

import streamlit as st
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import tempfile
import zipfile
from pathlib import Path

# Import our custom modules
try:
    from vertex_tuning_manager import VertexTuningManager
    from cloud_storage_manager import CloudStorageManager
    from local_storage_manager import LocalStorageManager
    VERTEX_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    VERTEX_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Vertex AI Supervised Tuning",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .info-box {
        background-color: #e8f4fd;
        border-left: 4px solid #3498db;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'vertex_manager' not in st.session_state:
        st.session_state.vertex_manager = None
    if 'storage_manager' not in st.session_state:
        st.session_state.storage_manager = None
    if 'annotations' not in st.session_state:
        st.session_state.annotations = []
    if 'dataset_info' not in st.session_state:
        st.session_state.dataset_info = None
    if 'training_jobs' not in st.session_state:
        st.session_state.training_jobs = []
    if 'models' not in st.session_state:
        st.session_state.models = []
    if 'endpoints' not in st.session_state:
        st.session_state.endpoints = []

def setup_vertex_connection():
    """Setup Vertex AI connection."""
    st.markdown('<div class="section-header">🔧 Vertex AI Configuration</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        project_id = st.text_input(
            "Google Cloud Project ID",
            value=os.getenv("GOOGLE_CLOUD_PROJECT", ""),
            help="Your Google Cloud project ID"
        )
        
        location = st.selectbox(
            "Vertex AI Region",
            ["us-central1", "us-east1", "us-west1", "europe-west1", "asia-northeast1"],
            index=0,
            help="Region where Vertex AI resources will be created"
        )
    
    with col2:
        bucket_name = st.text_input(
            "Cloud Storage Bucket",
            value=os.getenv("GCS_BUCKET_NAME", ""),
            help="GCS bucket for storing datasets and models"
        )
        
        api_key = st.text_input(
            "Google API Key (optional)",
            value=os.getenv("GOOGLE_API_KEY", ""),
            type="password",
            help="API key for additional Google services"
        )
    
    if st.button("🔗 Connect to Vertex AI", type="primary"):
        if not project_id:
            st.error("Please enter a Google Cloud Project ID")
            return False
        
        try:
            with st.spinner("Connecting to Vertex AI..."):
                vertex_manager = VertexTuningManager(project_id, location)
                
                if bucket_name:
                    vertex_manager.set_storage_bucket(bucket_name)
                
                st.session_state.vertex_manager = vertex_manager
                
                # Test connection by listing resources
                jobs = vertex_manager.list_training_jobs()
                models = vertex_manager.list_models()
                endpoints = vertex_manager.list_endpoints()
                
                st.session_state.training_jobs = jobs
                st.session_state.models = models
                st.session_state.endpoints = endpoints
                
                st.success("✅ Successfully connected to Vertex AI!")
                st.info(f"Found {len(jobs)} training jobs, {len(models)} models, and {len(endpoints)} endpoints")
                
        except Exception as e:
            st.error(f"Failed to connect to Vertex AI: {e}")
            return False
    
    return st.session_state.vertex_manager is not None

def load_annotations():
    """Load annotations from storage."""
    st.markdown('<div class="section-header">📊 Load Annotations</div>', unsafe_allow_html=True)
    
    # Initialize storage manager
    if st.session_state.storage_manager is None:
        if os.getenv("GCS_BUCKET_NAME"):
            st.session_state.storage_manager = CloudStorageManager()
        else:
            st.session_state.storage_manager = LocalStorageManager()
    
    # Load annotations
    try:
        annotations = st.session_state.storage_manager.list_annotations()
        st.session_state.annotations = annotations
        
        if annotations:
            st.success(f"✅ Loaded {len(annotations)} annotations")
            
            # Show annotation statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Images", len(annotations))
            
            with col2:
                # Count unique classes
                classes = set()
                for ann in annotations:
                    for obj in ann.get('annotations', []):
                        classes.add(obj.get('label', ''))
                st.metric("Unique Classes", len(classes))
            
            with col3:
                # Count total annotations
                total_annotations = sum(len(ann.get('annotations', [])) for ann in annotations)
                st.metric("Total Annotations", total_annotations)
            
            # Show class distribution
            if classes:
                st.subheader("Class Distribution")
                class_counts = {}
                for ann in annotations:
                    for obj in ann.get('annotations', []):
                        label = obj.get('label', '')
                        class_counts[label] = class_counts.get(label, 0) + 1
                
                # Create bar chart
                if class_counts:
                    chart_data = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Count'])
                    st.bar_chart(chart_data.set_index('Class'))
        
        else:
            st.warning("⚠️ No annotations found. Please create some annotations first.")
            
    except Exception as e:
        st.error(f"Failed to load annotations: {e}")

def prepare_classification_dataset():
    """Prepare classification dataset from annotations."""
    st.markdown('<div class="section-header">🎯 Prepare Classification Dataset</div>', unsafe_allow_html=True)
    
    if not st.session_state.annotations:
        st.warning("⚠️ No annotations available. Please load annotations first.")
        return
    
    # Dataset configuration
    col1, col2 = st.columns(2)
    
    with col1:
        dataset_name = st.text_input(
            "Dataset Name",
            value=f"classification_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            help="Name for the classification dataset"
        )
        
        train_split = st.slider(
            "Training Split",
            min_value=0.5,
            max_value=0.9,
            value=0.8,
            step=0.05,
            help="Percentage of data for training"
        )
    
    with col2:
        min_images_per_class = st.number_input(
            "Minimum Images per Class",
            min_value=1,
            value=10,
            help="Minimum number of images required per class"
        )
        
        max_classes = st.number_input(
            "Maximum Classes",
            min_value=1,
            value=50,
            help="Maximum number of classes to include"
        )
    
    if st.button("🔧 Prepare Dataset", type="primary"):
        try:
            with st.spinner("Preparing classification dataset..."):
                # Filter annotations based on criteria
                filtered_annotations = []
                class_counts = {}
                
                for ann in st.session_state.annotations:
                    image_path = ann.get('image_path', '')
                    if not image_path or not os.path.exists(image_path):
                        continue
                    
                    # Count classes for this image
                    classes = set()
                    for obj in ann.get('annotations', []):
                        label = obj.get('label', '').strip()
                        if label:
                            classes.add(label)
                    
                    # Use first class
                    if classes:
                        class_name = list(classes)[0]
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                        filtered_annotations.append(ann)
                
                # Filter classes based on minimum count
                valid_classes = {cls: count for cls, count in class_counts.items() 
                               if count >= min_images_per_class}
                
                if len(valid_classes) > max_classes:
                    # Keep top classes by count
                    sorted_classes = sorted(valid_classes.items(), key=lambda x: x[1], reverse=True)
                    valid_classes = dict(sorted_classes[:max_classes])
                
                # Filter annotations to only include valid classes
                final_annotations = []
                for ann in filtered_annotations:
                    classes = set()
                    for obj in ann.get('annotations', []):
                        label = obj.get('label', '').strip()
                        if label in valid_classes:
                            classes.add(label)
                    
                    if classes:
                        final_annotations.append(ann)
                
                if not final_annotations:
                    st.error("No valid annotations found with the specified criteria")
                    return
                
                # Prepare dataset
                dataset_paths = st.session_state.vertex_manager.prepare_classification_dataset(
                    final_annotations,
                    output_dir=f"datasets/{dataset_name}"
                )
                
                st.session_state.dataset_info = dataset_paths
                
                st.success(f"✅ Dataset prepared successfully!")
                st.info(f"Created dataset with {len(final_annotations)} images and {len(valid_classes)} classes")
                
                # Show dataset statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Images", len(final_annotations))
                
                with col2:
                    st.metric("Classes", len(valid_classes))
                
                with col3:
                    st.metric("Training Images", int(len(final_annotations) * train_split))
                
                # Show class distribution
                st.subheader("Final Class Distribution")
                chart_data = pd.DataFrame(list(valid_classes.items()), columns=['Class', 'Count'])
                st.bar_chart(chart_data.set_index('Class'))
                
        except Exception as e:
            st.error(f"Failed to prepare dataset: {e}")

def create_training_job():
    """Create a training job on Vertex AI."""
    st.markdown('<div class="section-header">🚀 Create Training Job</div>', unsafe_allow_html=True)
    
    if not st.session_state.dataset_info:
        st.warning("⚠️ No dataset prepared. Please prepare a dataset first.")
        return
    
    # Training job configuration
    col1, col2 = st.columns(2)
    
    with col1:
        job_name = st.text_input(
            "Training Job Name",
            value=f"classification_job_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            help="Name for the training job"
        )
        
        training_type = st.selectbox(
            "Training Type",
            ["AutoML", "Custom Training"],
            help="Choose between AutoML (easier) or Custom Training (more control)"
        )
        
        model_type = st.selectbox(
            "Model Architecture",
            ["efficientnet", "resnet", "mobilenet", "custom"],
            help="Model architecture for custom training"
        )
    
    with col2:
        if training_type == "AutoML":
            budget_hours = st.slider(
                "Training Budget (hours)",
                min_value=1,
                max_value=24,
                value=8,
                help="Maximum training time in hours"
            )
        else:
            epochs = st.number_input(
                "Training Epochs",
                min_value=1,
                max_value=200,
                value=50,
                help="Number of training epochs"
            )
            
            learning_rate = st.selectbox(
                "Learning Rate",
                [0.0001, 0.0005, 0.001, 0.005, 0.01],
                index=2,
                help="Learning rate for training"
            )
    
    if st.button("🚀 Start Training Job", type="primary"):
        try:
            with st.spinner("Creating training job..."):
                # Upload dataset to GCS
                gcs_paths = st.session_state.vertex_manager.upload_dataset_to_gcs(
                    st.session_state.dataset_info
                )
                
                if training_type == "AutoML":
                    # Create AutoML job
                    job_info = st.session_state.vertex_manager.create_automl_training_job(
                        gcs_paths,
                        job_name,
                        budget_milli_node_hours=budget_hours * 1000
                    )
                else:
                    # Create custom training job
                    hyperparameters = {
                        'epochs': epochs,
                        'learning_rate': learning_rate,
                        'model_type': model_type
                    }
                    
                    job_info = st.session_state.vertex_manager.create_custom_training_job(
                        gcs_paths,
                        job_name,
                        model_type=model_type,
                        hyperparameters=hyperparameters
                    )
                
                st.success(f"✅ Training job '{job_name}' created successfully!")
                st.info(f"Job status: {job_info['status']}")
                
                # Refresh job list
                st.session_state.training_jobs = st.session_state.vertex_manager.list_training_jobs()
                
        except Exception as e:
            st.error(f"Failed to create training job: {e}")

def manage_training_jobs():
    """Manage existing training jobs."""
    st.markdown('<div class="section-header">📋 Training Jobs</div>', unsafe_allow_html=True)
    
    if not st.session_state.training_jobs:
        st.info("No training jobs found.")
        return
    
    # Refresh jobs
    if st.button("🔄 Refresh Jobs"):
        st.session_state.training_jobs = st.session_state.vertex_manager.list_training_jobs()
        st.rerun()
    
    # Display jobs
    for job in st.session_state.training_jobs:
        with st.expander(f"📊 {job['name']} ({job['type']})"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Type:** {job['type']}")
                st.write(f"**State:** {job['state']}")
            
            with col2:
                st.write(f"**Created:** {job['create_time']}")
                if job['end_time']:
                    st.write(f"**Ended:** {job['end_time']}")
            
            with col3:
                if job['state'] == 'JOB_STATE_SUCCEEDED':
                    st.success("✅ Completed")
                elif job['state'] == 'JOB_STATE_RUNNING':
                    st.info("🔄 Running")
                elif job['state'] == 'JOB_STATE_FAILED':
                    st.error("❌ Failed")
                else:
                    st.warning(f"⚠️ {job['state']}")

def manage_models():
    """Manage trained models."""
    st.markdown('<div class="section-header">🤖 Trained Models</div>', unsafe_allow_html=True)
    
    if not st.session_state.models:
        st.info("No trained models found.")
        return
    
    # Refresh models
    if st.button("🔄 Refresh Models"):
        st.session_state.models = st.session_state.vertex_manager.list_models()
        st.rerun()
    
    # Display models
    for model in st.session_state.models:
        with st.expander(f"🤖 {model['name']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Model ID:** {model['model_id']}")
                st.write(f"**Version:** {model['version_id']}")
            
            with col2:
                st.write(f"**Created:** {model['create_time']}")
                st.write(f"**Updated:** {model['update_time']}")
            
            # Deploy button
            if st.button(f"🚀 Deploy {model['name']}", key=f"deploy_{model['model_id']}"):
                try:
                    endpoint_name = f"{model['name']}_endpoint"
                    endpoint_info = st.session_state.vertex_manager.deploy_model_to_endpoint(
                        model, endpoint_name
                    )
                    st.success(f"✅ Model deployed to endpoint: {endpoint_name}")
                    
                    # Refresh endpoints
                    st.session_state.endpoints = st.session_state.vertex_manager.list_endpoints()
                    
                except Exception as e:
                    st.error(f"Failed to deploy model: {e}")

def manage_endpoints():
    """Manage deployed endpoints."""
    st.markdown('<div class="section-header">🌐 Deployed Endpoints</div>', unsafe_allow_html=True)
    
    if not st.session_state.endpoints:
        st.info("No deployed endpoints found.")
        return
    
    # Refresh endpoints
    if st.button("🔄 Refresh Endpoints"):
        st.session_state.endpoints = st.session_state.vertex_manager.list_endpoints()
        st.rerun()
    
    # Display endpoints
    for endpoint in st.session_state.endpoints:
        with st.expander(f"🌐 {endpoint['name']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Endpoint ID:** {endpoint['endpoint_id']}")
                st.write(f"**Deployed Models:** {endpoint['deployed_models']}")
            
            with col2:
                st.write(f"**Created:** {endpoint['create_time']}")
                st.write(f"**Status:** Active")
            
            # Test prediction
            st.subheader("Test Prediction")
            test_image = st.file_uploader(
                "Upload test image",
                type=['jpg', 'jpeg', 'png'],
                key=f"test_{endpoint['endpoint_id']}"
            )
            
            if test_image and st.button(f"🔮 Predict", key=f"predict_{endpoint['endpoint_id']}"):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        tmp_file.write(test_image.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Make prediction
                    prediction = st.session_state.vertex_manager.predict_with_endpoint(
                        endpoint, tmp_path
                    )
                    
                    st.success("✅ Prediction completed!")
                    st.json(prediction)
                    
                    # Clean up
                    os.unlink(tmp_path)
                    
                except Exception as e:
                    st.error(f"Failed to make prediction: {e}")

def main():
    """Main application function."""
    st.markdown('<div class="main-header">🤖 Vertex AI Supervised Tuning</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>Welcome to Vertex AI Supervised Tuning!</strong><br>
        This tool helps you create and manage supervised tuning jobs for image classification models
        that can be deployed to Google Vertex AI. Upload your annotated images and create production-ready
        classification models with just a few clicks.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Check if Vertex AI is available
    if not VERTEX_AVAILABLE:
        st.error("""
        ❌ Vertex AI dependencies not available.
        
        Please install the required packages:
        ```bash
        pip install google-cloud-aiplatform pandas tensorflow pillow
        ```
        """)
        return
    
    # Setup Vertex AI connection
    if not setup_vertex_connection():
        st.stop()
    
    # Main workflow tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Load Data", 
        "🎯 Prepare Dataset", 
        "🚀 Training Jobs", 
        "🤖 Models", 
        "🌐 Endpoints"
    ])
    
    with tab1:
        load_annotations()
    
    with tab2:
        prepare_classification_dataset()
        create_training_job()
    
    with tab3:
        manage_training_jobs()
    
    with tab4:
        manage_models()
    
    with tab5:
        manage_endpoints()

if __name__ == "__main__":
    main() 