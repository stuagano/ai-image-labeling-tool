"""
Cloud-optimized AI Image Labeling Tool
Deployed on Google Cloud Run with Cloud Storage persistence
"""

import streamlit as st
import os
import tempfile
from PIL import Image
import io
from typing import List, Dict, Any
import json
import zipfile
import glob
import datetime

# Import our modules
from streamlit_img_label import st_img_label
from ai_utils import create_ai_detector, create_ai_assistant
from cloud_storage_manager import CloudStorageManager, CloudImageManager

# Page configuration
st.set_page_config(
    page_title="AI Image Labeler - Cloud",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if "cloud_storage" not in st.session_state:
        st.session_state.cloud_storage = None
    if "cloud_manager" not in st.session_state:
        st.session_state.cloud_manager = None
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "current_image" not in st.session_state:
        st.session_state.current_image = None
    if "current_annotations" not in st.session_state:
        st.session_state.current_annotations = []
    if "ai_detector" not in st.session_state:
        st.session_state.ai_detector = None
    if "ai_assistant" not in st.session_state:
        st.session_state.ai_assistant = None
    if "batch_processing" not in st.session_state:
        st.session_state.batch_processing = False
    if "batch_images" not in st.session_state:
        st.session_state.batch_images = []
    if "current_batch_index" not in st.session_state:
        st.session_state.current_batch_index = 0

def initialize_cloud_storage():
    """Initialize cloud storage connection"""
    try:
        if st.session_state.cloud_storage is None:
            bucket_name = os.getenv('GCS_BUCKET_NAME')
            if not bucket_name:
                st.error("GCS_BUCKET_NAME environment variable not set")
                return False
            
            st.session_state.cloud_storage = CloudStorageManager(bucket_name)
            st.session_state.cloud_manager = CloudImageManager(st.session_state.cloud_storage)
            return True
    except Exception as e:
        st.error(f"Failed to initialize cloud storage: {e}")
        return False
    return True

def initialize_ai_models():
    """Initialize AI models based on user selection"""
    if "ai_detector" not in st.session_state:
        st.session_state.ai_detector = None
        st.session_state.ai_assistant = None

def process_zip_file(uploaded_zip):
    """Extract and process images from uploaded ZIP file"""
    images = []
    try:
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            # Extract to temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_ref.extractall(temp_dir)
                
                # Find all image files
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff']
                for ext in image_extensions:
                    images.extend(glob.glob(os.path.join(temp_dir, '**', ext), recursive=True))
                    images.extend(glob.glob(os.path.join(temp_dir, '**', ext.upper()), recursive=True))
                
                # Load images
                processed_images = []
                for img_path in images:
                    try:
                        with open(img_path, 'rb') as f:
                            img_bytes = f.read()
                            img = Image.open(io.BytesIO(img_bytes))
                            img_name = os.path.basename(img_path)
                            processed_images.append({
                                'name': img_name,
                                'path': img_path,
                                'image': img,
                                'bytes': img_bytes
                            })
                    except Exception as e:
                        st.warning(f"Failed to process {img_path}: {e}")
                
                return processed_images
    except Exception as e:
        st.error(f"Failed to process ZIP file: {e}")
        return []

def main():
    """Main application function"""
    st.title("🤖 AI-Enhanced Image Labeling Tool - Cloud Edition")
    st.write("Deployed on Google Cloud Run with Cloud Storage persistence")
    
    # Initialize session state
    initialize_session_state()
    initialize_ai_models()
    
    # Check cloud storage connection
    if not initialize_cloud_storage():
        st.stop()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Cloud Storage Configuration
        st.subheader("☁️ Cloud Storage")
        bucket_uri = st.text_input(
            "GCS Bucket URI (optional)",
            value=os.getenv('GCS_BUCKET_NAME', ''),
            help="gs://your-bucket-name (leave empty to use environment variable)"
        )
        
        if bucket_uri and bucket_uri != os.getenv('GCS_BUCKET_NAME', ''):
            if st.button("Update Bucket"):
                try:
                    # Extract bucket name from URI
                    if bucket_uri.startswith('gs://'):
                        bucket_name = bucket_uri[5:]
                    else:
                        bucket_name = bucket_uri
                    
                    st.session_state.cloud_storage = CloudStorageManager(bucket_name)
                    st.session_state.cloud_manager = CloudImageManager(st.session_state.cloud_storage)
                    st.success(f"Connected to bucket: {bucket_name}")
                except Exception as e:
                    st.error(f"Failed to connect to bucket: {e}")
        
        # AI Configuration
        st.subheader("🤖 AI Configuration")
        ai_model = st.selectbox(
            "AI Model",
            ["None", "YOLO", "Transformers", "Gemini"],
            help="Select AI model for automatic object detection"
        )
        
        # API Key for Gemini
        api_key = None
        if ai_model == "Gemini":
            api_key = st.text_input(
                "Google API Key",
                type="password",
                help="Enter your Google API key for Gemini access"
            )
        
        # Confidence threshold
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence for AI detections"
        )
        
        # Initialize AI models
        if ai_model != "None" and st.session_state.ai_detector is None:
            if st.button("Initialize AI Model"):
                with st.spinner("Loading AI model..."):
                    try:
                        model_type = ai_model.lower()
                        detector = create_ai_detector(model_type, api_key)
                        assistant = create_ai_assistant(detector)
                        st.session_state.ai_detector = detector
                        st.session_state.ai_assistant = assistant
                        st.success(f"AI model ({ai_model}) loaded successfully!")
                    except Exception as e:
                        st.error(f"Failed to load AI model: {e}")
        
        # Labels configuration
        st.subheader("🏷️ Labels")
        labels_input = st.text_input(
            "Custom Labels (comma-separated)",
            value="person, car, dog, cat",
            help="Enter labels separated by commas"
        )
        
        labels = [""] + [label.strip() for label in labels_input.split(",") if label.strip()]
        
        # Annotation format
        annotation_format = st.selectbox(
            "Annotation Format",
            ["json", "coco"],
            index=0,
            help="JSON: Custom format, COCO: Standard COCO format"
        )
        
        # Statistics
        st.subheader("📊 Statistics")
        if st.session_state.cloud_manager:
            annotations = st.session_state.cloud_manager.list_annotations()
            st.write(f"Total annotations: {len(annotations)}")
            st.write(f"Uploaded files: {len(st.session_state.uploaded_files)}")
            if st.session_state.batch_processing:
                st.write(f"Batch images: {len(st.session_state.batch_images)}")
                st.write(f"Current batch: {st.session_state.current_batch_index + 1}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📸 Image Upload & Annotation")
        
        # Upload options
        upload_option = st.radio(
            "Upload Option",
            ["Single Image", "ZIP Folder", "Batch Processing"],
            help="Choose how to upload images"
        )
        
        if upload_option == "Single Image":
            # Single file uploader
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
                help="Upload an image to annotate"
            )
            
            if uploaded_file is not None:
                process_single_image(uploaded_file, labels, annotation_format)
        
        elif upload_option == "ZIP Folder":
            # ZIP file uploader
            uploaded_zip = st.file_uploader(
                "Upload ZIP file with images",
                type=['zip'],
                help="Upload a ZIP file containing images"
            )
            
            if uploaded_zip is not None:
                with st.spinner("Processing ZIP file..."):
                    images = process_zip_file(uploaded_zip)
                    if images:
                        st.success(f"Found {len(images)} images in ZIP file")
                        st.session_state.batch_images = images
                        st.session_state.batch_processing = True
                        st.session_state.current_batch_index = 0
                        
                        # Show first image
                        if st.session_state.batch_images:
                            current_img = st.session_state.batch_images[0]
                            process_batch_image(current_img, labels, annotation_format)
                    else:
                        st.error("No images found in ZIP file")
        
        elif upload_option == "Batch Processing":
            # Batch processing interface
            if st.session_state.batch_processing and st.session_state.batch_images:
                st.subheader(f"Batch Processing ({st.session_state.current_batch_index + 1}/{len(st.session_state.batch_images)})")
                
                # Navigation
                col_nav1, col_nav2, col_nav3 = st.columns(3)
                with col_nav1:
                    if st.button("⏮️ Previous") and st.session_state.current_batch_index > 0:
                        st.session_state.current_batch_index -= 1
                        st.rerun()
                
                with col_nav2:
                    st.write(f"Image {st.session_state.current_batch_index + 1} of {len(st.session_state.batch_images)}")
                
                with col_nav3:
                    if st.button("⏭️ Next") and st.session_state.current_batch_index < len(st.session_state.batch_images) - 1:
                        st.session_state.current_batch_index += 1
                        st.rerun()
                
                # Process current image
                current_img = st.session_state.batch_images[st.session_state.current_batch_index]
                process_batch_image(current_img, labels, annotation_format)
                
                # Batch actions
                st.subheader("Batch Actions")
                col_batch1, col_batch2 = st.columns(2)
                
                with col_batch1:
                    if st.button("🤖 AI Process All"):
                        process_all_images_with_ai(labels, annotation_format, confidence_threshold)
                
                with col_batch2:
                    if st.button("💾 Save All Annotations"):
                        save_all_annotations(labels, annotation_format)
                
                # Add Gemini description feature
                st.subheader("📝 Gemini Descriptions")
                col_gem1, col_gem2 = st.columns(2)
                
                with col_gem1:
                    if st.button("🤖 Get Gemini Descriptions"):
                        if st.session_state.ai_detector and hasattr(st.session_state.ai_detector, 'model_type') and st.session_state.ai_detector.model_type == 'gemini':
                            process_all_images_with_gemini_descriptions()
                        else:
                            st.error("Please initialize Gemini AI model first")
                
                with col_gem2:
                    if st.button("📋 View All Descriptions"):
                        display_all_gemini_descriptions()
            else:
                st.info("Upload a ZIP file to start batch processing")
    
    with col2:
        st.header("📋 Current Annotations")
        
        if st.session_state.current_annotations:
            for i, rect in enumerate(st.session_state.current_annotations):
                with st.expander(f"Annotation {i+1}"):
                    st.write(f"Position: ({rect['left']:.1f}, {rect['top']:.1f})")
                    st.write(f"Size: {rect['width']:.1f} x {rect['height']:.1f}")
                    if 'label' in rect:
                        st.write(f"Label: {rect['label']}")
                    
                    # Label selection
                    if 'label' not in rect or not rect['label']:
                        selected_label = st.selectbox(
                            f"Label for annotation {i+1}",
                            labels,
                            key=f"label_{i}"
                        )
                        rect['label'] = selected_label
        
        # Export options
        st.header("📤 Export Options")
        
        if st.button("Export COCO Dataset"):
            if st.session_state.cloud_manager:
                with st.spinner("Exporting dataset..."):
                    dataset = st.session_state.cloud_manager.export_dataset("coco")
                    if dataset:
                        success = st.session_state.cloud_manager.save_dataset_export(dataset, "coco")
                        if success:
                            st.success("COCO dataset exported successfully!")
                        else:
                            st.error("Failed to export dataset")
                    else:
                        st.error("No annotations to export")
        
        if st.button("Export JSON Dataset"):
            if st.session_state.cloud_manager:
                with st.spinner("Exporting dataset..."):
                    dataset = st.session_state.cloud_manager.export_dataset("json")
                    if dataset:
                        success = st.session_state.cloud_manager.save_dataset_export(dataset, "json")
                        if success:
                            st.success("JSON dataset exported successfully!")
                        else:
                            st.error("Failed to export dataset")
                    else:
                        st.error("No annotations to export")
        
        # List existing annotations
        st.header("📁 Existing Annotations")
        if st.session_state.cloud_manager:
            annotations = st.session_state.cloud_manager.list_annotations()
            if annotations:
                for ann in annotations[:10]:  # Show first 10
                    st.write(f"• {ann}")
                if len(annotations) > 10:
                    st.write(f"... and {len(annotations) - 10} more")
            else:
                st.write("No annotations found")

def process_single_image(uploaded_file, labels, annotation_format):
    """Process a single uploaded image"""
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Store current image
    st.session_state.current_image = {
        'name': uploaded_file.name,
        'path': f"{uploaded_file.name}_{len(uploaded_file.getvalue())}",
        'image': image,
        'bytes': uploaded_file.getvalue()
    }
    
    # Load existing annotations
    if st.session_state.cloud_manager:
        file_identifier = f"{uploaded_file.name}_{len(uploaded_file.getvalue())}"
        existing_annotations = st.session_state.cloud_manager.load_annotation(file_identifier)
        if existing_annotations:
            st.session_state.current_annotations = existing_annotations.get('rects', [])
            st.info(f"Loaded {len(st.session_state.current_annotations)} existing annotations")
        else:
            st.session_state.current_annotations = []
    
    # AI Suggestions
    if st.session_state.ai_detector and st.session_state.ai_assistant:
        if st.button("🤖 Get AI Suggestions"):
            with st.spinner("Analyzing image with AI..."):
                try:
                    # Save image temporarily for AI processing
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                        image.save(tmp_file.name, 'JPEG')
                        suggestions = st.session_state.ai_assistant.suggest_annotations(
                            tmp_file.name, 
                            confidence_threshold=st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
                        )
                        os.unlink(tmp_file.name)
                    
                    if suggestions and suggestions.get('detections'):
                        st.session_state.current_annotations = suggestions['detections']
                        st.success(f"AI found {len(suggestions['detections'])} objects")
                    else:
                        st.warning("No objects detected by AI")
                except Exception as e:
                    st.error(f"AI analysis failed: {e}")
    
    # Image annotation interface
    if st.session_state.current_image:
        # Convert image for annotation
        img_array = st.session_state.current_image['image']
        
        # Create annotation interface
        rects = st_img_label(
            img_array, 
            box_color="red", 
            rects=st.session_state.current_annotations
        )
        
        if rects:
            st.session_state.current_annotations = rects
            
            # Save button
            if st.button("💾 Save Annotations"):
                if st.session_state.cloud_manager:
                    annotation_data = {
                        'rects': rects,
                        'image_name': uploaded_file.name,
                        'image_width': image.width,
                        'image_height': image.height,
                        'labels': labels,
                        'format': annotation_format
                    }
                    
                    file_identifier = f"{uploaded_file.name}_{len(uploaded_file.getvalue())}"
                    success = st.session_state.cloud_manager.save_annotation(
                        file_identifier, 
                        annotation_data
                    )
                    
                    if success:
                        st.success("Annotations saved to cloud storage!")
                    else:
                        st.error("Failed to save annotations")

def process_batch_image(current_img, labels, annotation_format):
    """Process an image in batch mode"""
    # Display image
    st.image(current_img['image'], caption=f"Batch Image: {current_img['name']}", use_column_width=True)
    
    # Store current image
    st.session_state.current_image = current_img
    
    # Load existing annotations
    if st.session_state.cloud_manager:
        file_identifier = f"{current_img['name']}_{len(current_img['bytes'])}"
        existing_annotations = st.session_state.cloud_manager.load_annotation(file_identifier)
        if existing_annotations:
            st.session_state.current_annotations = existing_annotations.get('rects', [])
            st.info(f"Loaded {len(st.session_state.current_annotations)} existing annotations")
        else:
            st.session_state.current_annotations = []
    
    # AI Suggestions for current image
    if st.session_state.ai_detector and st.session_state.ai_assistant:
        if st.button("🤖 Get AI Suggestions"):
            with st.spinner("Analyzing image with AI..."):
                try:
                    # Save image temporarily for AI processing
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                        current_img['image'].save(tmp_file.name, 'JPEG')
                        suggestions = st.session_state.ai_assistant.suggest_annotations(
                            tmp_file.name, 
                            confidence_threshold=st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
                        )
                        os.unlink(tmp_file.name)
                    
                    if suggestions and suggestions.get('detections'):
                        st.session_state.current_annotations = suggestions['detections']
                        st.success(f"AI found {len(suggestions['detections'])} objects")
                    else:
                        st.warning("No objects detected by AI")
                except Exception as e:
                    st.error(f"AI analysis failed: {e}")
    
    # Image annotation interface
    if st.session_state.current_image:
        # Convert image for annotation
        img_array = st.session_state.current_image['image']
        
        # Create annotation interface
        rects = st_img_label(
            img_array, 
            box_color="red", 
            rects=st.session_state.current_annotations
        )
        
        if rects:
            st.session_state.current_annotations = rects
            
            # Save button for current image
            if st.button("💾 Save Current Image"):
                if st.session_state.cloud_manager:
                    annotation_data = {
                        'rects': rects,
                        'image_name': current_img['name'],
                        'image_width': current_img['image'].width,
                        'image_height': current_img['image'].height,
                        'labels': labels,
                        'format': annotation_format
                    }
                    
                    file_identifier = f"{current_img['name']}_{len(current_img['bytes'])}"
                    success = st.session_state.cloud_manager.save_annotation(
                        file_identifier, 
                        annotation_data
                    )
                    
                    if success:
                        st.success("Annotations saved to cloud storage!")
                    else:
                        st.error("Failed to save annotations")

def process_all_images_with_ai(labels, annotation_format, confidence_threshold):
    """Process all images in batch with AI"""
    if not st.session_state.batch_images:
        st.error("No images in batch")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, img in enumerate(st.session_state.batch_images):
        status_text.text(f"Processing {img['name']} ({i+1}/{len(st.session_state.batch_images)})")
        
        try:
            # Get AI suggestions
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                img['image'].save(tmp_file.name, 'JPEG')
                suggestions = st.session_state.ai_assistant.suggest_annotations(
                    tmp_file.name, 
                    confidence_threshold=confidence_threshold
                )
                os.unlink(tmp_file.name)
            
            if suggestions and suggestions.get('detections'):
                # Save annotations
                annotation_data = {
                    'rects': suggestions['detections'],
                    'image_name': img['name'],
                    'image_width': img['image'].width,
                    'image_height': img['image'].height,
                    'labels': labels,
                    'format': annotation_format
                }
                
                file_identifier = f"{img['name']}_{len(img['bytes'])}"
                st.session_state.cloud_manager.save_annotation(file_identifier, annotation_data)
        
        except Exception as e:
            st.warning(f"Failed to process {img['name']}: {e}")
        
        progress_bar.progress((i + 1) / len(st.session_state.batch_images))
    
    status_text.text("Batch AI processing completed!")
    st.success(f"Processed {len(st.session_state.batch_images)} images with AI")

def save_all_annotations(labels, annotation_format):
    """Save annotations for all images in batch"""
    if not st.session_state.batch_images:
        st.error("No images in batch")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, img in enumerate(st.session_state.batch_images):
        status_text.text(f"Saving {img['name']} ({i+1}/{len(st.session_state.batch_images)})")
        
        try:
            # Save current annotations for this image
            if st.session_state.current_annotations:
                annotation_data = {
                    'rects': st.session_state.current_annotations,
                    'image_name': img['name'],
                    'image_width': img['image'].width,
                    'image_height': img['image'].height,
                    'labels': labels,
                    'format': annotation_format
                }
                
                file_identifier = f"{img['name']}_{len(img['bytes'])}"
                st.session_state.cloud_manager.save_annotation(file_identifier, annotation_data)
        
        except Exception as e:
            st.warning(f"Failed to save {img['name']}: {e}")
        
        progress_bar.progress((i + 1) / len(st.session_state.batch_images))
    
    status_text.text("Batch save completed!")
    st.success(f"Saved annotations for {len(st.session_state.batch_images)} images")

def process_all_images_with_gemini_descriptions():
    """Process all images with Gemini to get descriptions"""
    if not st.session_state.batch_images:
        st.error("No images in batch")
        return
    
    if not st.session_state.ai_detector or not hasattr(st.session_state.ai_detector, 'model_type') or st.session_state.ai_detector.model_type != 'gemini':
        st.error("Gemini AI model not initialized")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Initialize descriptions storage if not exists
    if "gemini_descriptions" not in st.session_state:
        st.session_state.gemini_descriptions = {}
    
    for i, img in enumerate(st.session_state.batch_images):
        status_text.text(f"Getting Gemini description for {img['name']} ({i+1}/{len(st.session_state.batch_images)})")
        
        try:
            # Get Gemini description
            description = get_gemini_image_description(img['image'], img['name'])
            
            if description:
                st.session_state.gemini_descriptions[img['name']] = {
                    'description': description,
                    'timestamp': str(datetime.datetime.now()),
                    'image_path': img['name']
                }
                
                # Save description to cloud storage
                save_gemini_description(img['name'], description)
        
        except Exception as e:
            st.warning(f"Failed to get description for {img['name']}: {e}")
        
        progress_bar.progress((i + 1) / len(st.session_state.batch_images))
    
    status_text.text("Gemini descriptions completed!")
    st.success(f"Generated descriptions for {len(st.session_state.gemini_descriptions)} images")

def get_gemini_image_description(image, image_name):
    """Get detailed description of image using Gemini"""
    try:
        import google.generativeai as genai
        
        # Configure Gemini
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            st.error("GOOGLE_API_KEY environment variable not set")
            return None
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro-vision')
        
        # Prepare prompt for detailed description
        prompt = """
        Please provide a detailed description of this image. Include:
        1. Main objects and subjects visible
        2. Actions or activities happening
        3. Background and setting
        4. Notable details or characteristics
        5. Overall scene composition
        
        Format your response as a clear, structured description that would be useful for image annotation and object detection training.
        """
        
        # Convert PIL image to bytes for Gemini
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Generate description
        response = model.generate_content([prompt, img_byte_arr])
        
        if response and response.text:
            return response.text.strip()
        else:
            return "No description generated"
            
    except Exception as e:
        st.error(f"Gemini API error: {e}")
        return None

def save_gemini_description(image_name, description):
    """Save Gemini description to cloud storage"""
    if not st.session_state.cloud_manager:
        return False
    
    try:
        description_data = {
            'image_name': image_name,
            'description': description,
            'timestamp': str(datetime.datetime.now()),
            'source': 'gemini',
            'metadata': {
                'model': 'gemini-pro-vision',
                'description_type': 'detailed_scene_analysis'
            }
        }
        
        # Save to descriptions folder
        file_identifier = f"{image_name}_{len(description)}"  # Simple identifier
        file_path = f"descriptions/{file_identifier}.json"
        
        return st.session_state.cloud_storage.upload_json(description_data, file_path)
        
    except Exception as e:
        st.error(f"Failed to save description: {e}")
        return False

def display_all_gemini_descriptions():
    """Display all Gemini descriptions in a table"""
    if not st.session_state.batch_images:
        st.error("No images in batch")
        return
    
    if "gemini_descriptions" not in st.session_state or not st.session_state.gemini_descriptions:
        st.info("No Gemini descriptions available. Run 'Get Gemini Descriptions' first.")
        return
    
    st.subheader("📋 All Gemini Descriptions")
    
    # Create a table of descriptions
    descriptions_data = []
    for img in st.session_state.batch_images:
        img_name = img['name']
        if img_name in st.session_state.gemini_descriptions:
            desc = st.session_state.gemini_descriptions[img_name]
            descriptions_data.append({
                'Image': img_name,
                'Description': desc['description'][:100] + "..." if len(desc['description']) > 100 else desc['description'],
                'Timestamp': desc['timestamp'],
                'Full Description': desc['description']
            })
        else:
            descriptions_data.append({
                'Image': img_name,
                'Description': 'No description available',
                'Timestamp': 'N/A',
                'Full Description': 'No description available'
            })
    
    # Display as expandable sections
    for i, data in enumerate(descriptions_data):
        with st.expander(f"📷 {data['Image']} - {data['Description']}"):
            st.write("**Full Description:**")
            st.write(data['Full Description'])
            st.write(f"**Generated:** {data['Timestamp']}")
            
            # Show the image
            for img in st.session_state.batch_images:
                if img['name'] == data['Image']:
                    st.image(img['image'], caption=data['Image'], width=300)
                    break

if __name__ == "__main__":
    main() 