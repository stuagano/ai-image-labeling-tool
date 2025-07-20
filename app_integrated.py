import streamlit as st
import os
import json
import tempfile
import webbrowser
from typing import List, Dict, Optional
from streamlit_img_label import st_img_label
from streamlit_img_label.manage import ImageDirManager
from streamlit_img_label.export import export_coco_dataset, export_yolo_format, export_csv, validate_annotations

# Import AI and Vertex AI components
try:
    from ai_utils import create_ai_detector, create_ai_assistant
    from ai_image_manager import AIImageManager
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    st.warning("AI components not available. Install with: pip install -r requirements.txt")

try:
    from vertex_tuning_manager import VertexTuningManager
    from google.cloud import aiplatform
    VERTEX_AVAILABLE = True
except ImportError:
    VERTEX_AVAILABLE = False
    st.warning("Vertex AI components not available. Install with: pip install google-cloud-aiplatform")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    st.warning("Gemini components not available. Install with: pip install google-generativeai")

# Google Cloud authentication uses Application Default Credentials (ADC)
# No additional imports needed - uses gcloud CLI authentication

def initialize_session_state():
    """Initialize session state variables"""
    if "gcp_authenticated" not in st.session_state:
        st.session_state.gcp_authenticated = False
    if "gcp_project_id" not in st.session_state:
        st.session_state.gcp_project_id = ""
    if "gcp_region" not in st.session_state:
        st.session_state.gcp_region = "us-central1"
    if "vertex_manager" not in st.session_state:
        st.session_state.vertex_manager = None
    if "ai_detector" not in st.session_state:
        st.session_state.ai_detector = None
    if "ai_assistant" not in st.session_state:
        st.session_state.ai_assistant = None
    if "gemini_model" not in st.session_state:
        st.session_state.gemini_model = None

def setup_gcp_authentication():
    """Setup GCP authentication using Application Default Credentials (ADC)"""
    st.sidebar.write("---")
    st.sidebar.write("**☁️ Google Cloud Setup**")
    
    # Show current authentication status
    if st.session_state.gcp_authenticated:
        st.sidebar.success(f"✅ Connected to: {st.session_state.gcp_project_id}")
        st.sidebar.info(f"🌍 Region: {st.session_state.gcp_region}")
        
        if st.sidebar.button("🔓 Disconnect"):
            st.session_state.gcp_authenticated = False
            st.session_state.gcp_project_id = None
            st.session_state.gcp_region = None
            st.session_state.vertex_manager = None
            st.rerun()
        return True
    
    # Check if user is authenticated with gcloud
    try:
        import subprocess
        result = subprocess.run(['gcloud', 'auth', 'list', '--filter=status:ACTIVE', '--format=value(account)'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            current_user = result.stdout.strip()
            st.sidebar.success(f"✅ Authenticated as: {current_user}")
        else:
            st.sidebar.warning("⚠️ Not authenticated with gcloud")
            st.sidebar.info("Run: `gcloud auth login` in terminal")
            return False
    except:
        st.sidebar.warning("⚠️ gcloud CLI not found")
        st.sidebar.info("Install Google Cloud SDK first")
        return False
    
    # Project ID input
    project_id = st.sidebar.text_input(
        "Google Cloud Project ID",
        value=st.session_state.gcp_project_id or "",
        placeholder="your-project-id",
        help="Enter your Google Cloud Project ID"
    )
    
    # Region selection
    region = st.sidebar.selectbox(
        "GCP Region",
        ["us-central1", "us-east1", "us-west1", "europe-west1", "asia-northeast1"],
        index=0,
        help="Select your preferred GCP region"
    )
    
    # Connect to Google Cloud
    if st.sidebar.button("☁️ Connect to Google Cloud", type="primary"):
        if not project_id.strip():
            st.sidebar.error("❌ Please enter your Google Cloud Project ID")
            return False
        
        try:
            with st.spinner("Connecting to Google Cloud..."):
                # Initialize Vertex AI with ADC
                if VERTEX_AVAILABLE:
                    try:
                        aiplatform.init(
                            project=project_id, 
                            location=region
                            # No credentials parameter - uses ADC automatically
                        )
                        st.session_state.vertex_manager = VertexTuningManager(project_id, region)
                        
                        # Store project settings
                        st.session_state.gcp_project_id = project_id
                        st.session_state.gcp_region = region
                        st.session_state.gcp_authenticated = True
                        
                        st.sidebar.success("✅ Google Cloud connected!")
                        st.rerun()
                    except Exception as e:
                        st.sidebar.error(f"❌ Failed to connect: {e}")
                        st.sidebar.info("Make sure you have access to this project")
                else:
                    st.sidebar.warning("⚠️ Vertex AI not available")
        except Exception as e:
            st.sidebar.error(f"❌ Connection failed: {e}")
    
    return False

def setup_ai_models():
    """Setup AI models for image detection"""
    if not AI_AVAILABLE:
        return
    
    st.sidebar.write("---")
    st.sidebar.write("**🤖 AI Image Detection**")
    
    # AI Model Selection
    ai_model = st.sidebar.selectbox(
        "AI Model",
        ["None", "YOLO", "Transformers", "Gemini"],
        help="Select AI model for automatic object detection"
    )
    
    # API Key for Gemini
    api_key = None
    if ai_model == "Gemini":
        api_key = st.sidebar.text_input(
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
        if st.sidebar.button("Initialize AI Model"):
            with st.spinner("Loading AI model..."):
                try:
                    model_type = ai_model.lower()
                    detector = create_ai_detector(model_type, api_key)
                    assistant = create_ai_assistant(detector)
                    st.session_state.ai_detector = detector
                    st.session_state.ai_assistant = assistant
                    st.sidebar.success(f"AI model ({ai_model}) loaded successfully!")
                except Exception as e:
                    st.sidebar.error(f"Failed to load AI model: {e}")

def setup_gemini_for_prompts():
    """Setup Gemini for prompt generation"""
    if not GEMINI_AVAILABLE:
        return None
    
    st.sidebar.write("---")
    st.sidebar.write("**🧠 Gemini Prompt Generation**")
    
    gemini_api_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        help="Enter your Google API key for Gemini access"
    )
    
    if gemini_api_key and st.session_state.gemini_model is None:
        if st.sidebar.button("Initialize Gemini"):
            try:
                genai.configure(api_key=gemini_api_key)
                model = genai.GenerativeModel('gemini-pro')
                st.session_state.gemini_model = model
                st.sidebar.success("✅ Gemini initialized successfully!")
            except Exception as e:
                st.sidebar.error(f"❌ Gemini initialization failed: {e}")
    
    return st.session_state.gemini_model

def generate_training_prompts(gemini_model, labels: List[str], task_description: str) -> Dict:
    """Generate training prompts using Gemini"""
    if not gemini_model:
        return {}
    
    try:
        prompt = f"""
        I'm creating a supervised fine-tuning dataset for image classification. 
        I have these labels: {', '.join(labels)}
        
        Task description: {task_description}
        
        Please generate:
        1. A system prompt for the model
        2. Training prompts for each label
        3. Validation prompts for each label
        4. A few example training data entries
        
        Return the response as a JSON object with these keys:
        - system_prompt: string
        - training_prompts: dict with label as key and list of prompts as value
        - validation_prompts: dict with label as key and list of prompts as value
        - example_data: list of dicts with 'prompt' and 'response' keys
        """
        
        response = gemini_model.generate_content(prompt)
        return json.loads(response.text)
    except Exception as e:
        st.error(f"Failed to generate prompts: {e}")
        return {}

def create_fine_tuning_tab():
    """Create the fine-tuning tab"""
    st.write("## 🎯 Supervised Fine-Tuning")
    
    if not st.session_state.gcp_authenticated:
        st.info("🔐 **Sign in to Google Cloud** in the sidebar to access fine-tuning features")
        st.write("### What you can do with Google Cloud:")
        st.write("• 🏗️ Create custom training jobs")
        st.write("• 🤖 Deploy models to Vertex AI endpoints")
        st.write("• 📊 Monitor training progress")
        st.write("• 🔄 Scale models automatically")
        return
    
    if not VERTEX_AVAILABLE:
        st.error("❌ Vertex AI components not available. Install with: pip install google-cloud-aiplatform")
        return
    
    if not st.session_state.vertex_manager:
        st.warning("⚠️ Vertex AI setup incomplete. You can still use image labeling features.")
        st.write("### Current Status:")
        st.write(f"• ✅ Signed in as: {st.session_state.user_email}")
        st.write(f"• 📁 Project: {st.session_state.gcp_project_id}")
        st.write(f"• 🌍 Region: {st.session_state.gcp_region}")
        return
    
    # Setup Gemini for prompt generation
    gemini_model = setup_gemini_for_prompts()
    
    # Fine-tuning configuration
    st.write("### Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        task_type = st.selectbox(
            "Task Type",
            ["classification", "object_detection", "segmentation"],
            help="Type of machine learning task"
        )
        
        training_type = st.selectbox(
            "Training Type",
            ["AutoML", "Custom Training"],
            help="Choose between AutoML or custom training"
        )
    
    with col2:
        model_name = st.text_input(
            "Model Name",
            value="my-custom-model",
            help="Name for your trained model"
        )
        
        dataset_name = st.text_input(
            "Dataset Name",
            value="my-dataset",
            help="Name for your dataset"
        )
    
    # Prompt generation section
    st.write("### 📝 Prompt Generation")
    
    task_description = st.text_area(
        "Task Description",
        value="Classify images based on the annotated objects",
        help="Describe what your model should do"
    )
    
    # Get labels from current annotations
    labels = []
    if "annotation_files" in st.session_state:
        # Extract labels from existing annotations
        for file in st.session_state.get("annotation_files", []):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and 'annotations' in data:
                        for ann in data['annotations']:
                            if 'label' in ann and ann['label'] not in labels:
                                labels.append(ann['label'])
            except:
                continue
    
    # Add manual labels input
    manual_labels = st.text_input(
        "Labels (comma-separated)",
        value=", ".join(labels),
        help="Enter or update labels for your model"
    )
    
    if manual_labels.strip():
        labels = [label.strip() for label in manual_labels.split(",") if label.strip()]
    
    # Generate prompts with Gemini
    if gemini_model and st.button("Generate Prompts with Gemini"):
        with st.spinner("Generating prompts..."):
            prompts = generate_training_prompts(gemini_model, labels, task_description)
            
            if prompts:
                st.session_state.generated_prompts = prompts
                st.success("✅ Prompts generated successfully!")
                
                # Display generated prompts
                st.write("#### Generated Prompts:")
                
                if 'system_prompt' in prompts:
                    st.text_area("System Prompt", prompts['system_prompt'], height=100)
                
                if 'training_prompts' in prompts:
                    st.write("**Training Prompts:**")
                    for label, label_prompts in prompts['training_prompts'].items():
                        with st.expander(f"Training prompts for '{label}'"):
                            for i, prompt in enumerate(label_prompts):
                                st.write(f"{i+1}. {prompt}")
                
                if 'example_data' in prompts:
                    st.write("**Example Training Data:**")
                    for i, example in enumerate(prompts['example_data'][:5]):
                        st.write(f"**Example {i+1}:**")
                        st.write(f"Prompt: {example.get('prompt', '')}")
                        st.write(f"Response: {example.get('response', '')}")
                        st.write("---")
    
    # Manual prompt editing
    st.write("### ✏️ Manual Prompt Editing")
    
    system_prompt = st.text_area(
        "System Prompt",
        value=st.session_state.get('generated_prompts', {}).get('system_prompt', ''),
        height=100,
        help="System prompt for the model"
    )
    
    # Training prompts for each label
    st.write("**Training Prompts by Label:**")
    training_prompts = {}
    for label in labels:
        with st.expander(f"Training prompts for '{label}'"):
            prompts_text = st.text_area(
                f"Prompts for {label}",
                value="\n".join(st.session_state.get('generated_prompts', {}).get('training_prompts', {}).get(label, [])),
                height=150,
                help=f"Enter training prompts for {label} (one per line)"
            )
            training_prompts[label] = [p.strip() for p in prompts_text.split('\n') if p.strip()]
    
    # Dataset preparation
    st.write("### 📊 Dataset Preparation")
    
    if st.button("Prepare Dataset for Fine-Tuning"):
        if not labels:
            st.error("❌ Please add some labels first")
            return
        
        try:
            with st.spinner("Preparing dataset..."):
                # Create dataset from current annotations
                dataset_path = st.session_state.vertex_manager.prepare_dataset_for_fine_tuning(
                    img_dir="img_dir",  # Current image directory
                    labels=labels,
                    system_prompt=system_prompt,
                    training_prompts=training_prompts,
                    task_type=task_type
                )
                
                st.session_state.dataset_path = dataset_path
                st.success(f"✅ Dataset prepared: {dataset_path}")
                
        except Exception as e:
            st.error(f"❌ Dataset preparation failed: {e}")
    
    # Upload to GCS
    if 'dataset_path' in st.session_state and st.button("Upload Dataset to GCS"):
        try:
            with st.spinner("Uploading to GCS..."):
                gcs_uri = st.session_state.vertex_manager.upload_dataset_to_gcs(
                    dataset_path=st.session_state.dataset_path,
                    bucket_name=f"{st.session_state.gcp_project_id}-datasets"
                )
                
                st.session_state.gcs_dataset_uri = gcs_uri
                st.success(f"✅ Dataset uploaded to: {gcs_uri}")
                
        except Exception as e:
            st.error(f"❌ Upload failed: {e}")
    
    # Create training job
    if 'gcs_dataset_uri' in st.session_state and st.button("Create Training Job"):
        try:
            with st.spinner("Creating training job..."):
                if training_type == "AutoML":
                    job = st.session_state.vertex_manager.create_automl_training_job(
                        dataset_uri=st.session_state.gcs_dataset_uri,
                        model_name=model_name,
                        task_type=task_type
                    )
                else:
                    job = st.session_state.vertex_manager.create_custom_training_job(
                        dataset_uri=st.session_state.gcs_dataset_uri,
                        model_name=model_name,
                        task_type=task_type
                    )
                
                st.session_state.training_job = job
                st.success(f"✅ Training job created: {job.name}")
                
        except Exception as e:
            st.error(f"❌ Training job creation failed: {e}")
    
    # Monitor training jobs
    st.write("### 📈 Training Jobs")
    
    if st.button("List Training Jobs"):
        try:
            jobs = st.session_state.vertex_manager.list_training_jobs()
            
            if jobs:
                st.write("**Active Training Jobs:**")
                for job in jobs:
                    with st.expander(f"Job: {job.display_name}"):
                        st.write(f"**Status:** {job.state}")
                        st.write(f"**Created:** {job.create_time}")
                        st.write(f"**Model:** {job.model_display_name if hasattr(job, 'model_display_name') else 'N/A'}")
                        
                        if job.state == "JOB_STATE_SUCCEEDED":
                            if st.button(f"Deploy Model {job.display_name}"):
                                try:
                                    endpoint = st.session_state.vertex_manager.deploy_model(
                                        model_name=job.model_display_name or model_name
                                    )
                                    st.success(f"✅ Model deployed to endpoint: {endpoint.name}")
                                except Exception as e:
                                    st.error(f"❌ Deployment failed: {e}")
            else:
                st.info("No training jobs found")
                
        except Exception as e:
            st.error(f"❌ Failed to list training jobs: {e}")

def run_integrated_app(img_dir, labels, annotation_format="json"):
    """Main integrated app function"""
    idm = ImageDirManager(img_dir, annotation_format)

    if "files" not in st.session_state:
        st.session_state["files"] = idm.get_all_files()
        st.session_state["annotation_files"] = idm.get_exist_annotation_files()
        st.session_state["image_index"] = 0
    else:
        idm.set_all_files(st.session_state["files"])
        idm.set_annotation_files(st.session_state["annotation_files"])
    
    def refresh():
        st.session_state["files"] = idm.get_all_files()
        st.session_state["annotation_files"] = idm.get_exist_annotation_files()
        st.session_state["image_index"] = 0

    def next_image():
        image_index = st.session_state["image_index"]
        if image_index < len(st.session_state["files"]) - 1:
            st.session_state["image_index"] += 1
        else:
            st.warning('This is the last image.')

    def previous_image():
        image_index = st.session_state["image_index"]
        if image_index > 0:
            st.session_state["image_index"] -= 1
        else:
            st.warning('This is the first image.')

    def next_annotate_file():
        image_index = st.session_state["image_index"]
        next_image_index = idm.get_next_annotation_image(image_index)
        if next_image_index:
            st.session_state["image_index"] = idm.get_next_annotation_image(image_index)
        else:
            st.warning("All images are annotated.")
            next_image()

    def go_to_image():
        file_index = st.session_state["files"].index(st.session_state["file"])
        st.session_state["image_index"] = file_index

    # Sidebar: show status
    n_files = len(st.session_state["files"])
    n_annotate_files = len(st.session_state["annotation_files"])
    st.sidebar.write("Total files:", n_files)
    st.sidebar.write("Total annotate files:", n_annotate_files)
    st.sidebar.write("Remaining files:", n_files - n_annotate_files)
    
    # Show current annotation format
    st.sidebar.write(f"Annotation format: {annotation_format.upper()}")

    st.sidebar.selectbox(
        "Files",
        st.session_state["files"],
        index=st.session_state["image_index"],
        on_change=go_to_image,
        key="file",
    )
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.button(label="Previous image", on_click=previous_image)
    with col2:
        st.button(label="Next image", on_click=next_image)
    st.sidebar.button(label="Next need annotate", on_click=next_annotate_file)
    st.sidebar.button(label="Refresh", on_click=refresh)

    # Export section in sidebar
    st.sidebar.write("---")
    st.sidebar.write("**Export Options**")
    
    if st.sidebar.button("Export COCO Dataset"):
        output_file = os.path.join(img_dir, "annotations_coco.json")
        try:
            export_coco_dataset(img_dir, output_file, annotation_format)
            st.sidebar.success(f"Exported to {output_file}")
        except Exception as e:
            st.sidebar.error(f"Export failed: {e}")
    
    if st.sidebar.button("Export YOLO Format"):
        output_dir = os.path.join(img_dir, "yolo_export")
        try:
            export_yolo_format(img_dir, output_dir, annotation_format)
            st.sidebar.success(f"Exported to {output_dir}")
        except Exception as e:
            st.sidebar.error(f"Export failed: {e}")
    
    if st.sidebar.button("Export CSV"):
        output_file = os.path.join(img_dir, "annotations.csv")
        try:
            export_csv(img_dir, output_file, annotation_format)
            st.sidebar.success(f"Exported to {output_file}")
        except Exception as e:
            st.sidebar.error(f"Export failed: {e}")
    
    # Validation section
    st.sidebar.write("---")
    st.sidebar.write("**Validation**")
    
    if st.sidebar.button("Validate Annotations"):
        try:
            stats = validate_annotations(img_dir, annotation_format)
            
            # Display validation results
            st.sidebar.write(f"**Validation Results:**")
            st.sidebar.write(f"Total images: {stats['total_images']}")
            st.sidebar.write(f"Annotated images: {stats['annotated_images']}")
            st.sidebar.write(f"Total annotations: {stats['total_annotations']}")
            st.sidebar.write(f"Empty labels: {stats['empty_labels']}")
            st.sidebar.write(f"Overlapping boxes: {stats['overlapping_boxes']}")
            
            if stats['issues']:
                st.sidebar.write("**Issues found:**")
                for issue in stats['issues'][:5]:  # Show first 5 issues
                    st.sidebar.write(f"• {issue}")
                if len(stats['issues']) > 5:
                    st.sidebar.write(f"... and {len(stats['issues']) - 5} more")
            
            # Show label distribution
            if stats['label_distribution']:
                st.sidebar.write("**Label distribution:**")
                for label, count in sorted(stats['label_distribution'].items(), key=lambda x: x[1], reverse=True):
                    st.sidebar.write(f"• {label}: {count}")
                    
        except Exception as e:
            st.sidebar.error(f"Validation failed: {e}")

    # Main content: annotate images
    img_file_name = idm.get_image(st.session_state["image_index"])
    img_path = os.path.join(img_dir, img_file_name)
    
    # Use AI-enhanced manager if available
    if AI_AVAILABLE and st.session_state.ai_assistant is not None:
        im = AIImageManager(img_path, annotation_format)
    else:
        from streamlit_img_label.manage import ImageManager
        im = ImageManager(img_path, annotation_format)
    
    img = im.get_img()
    resized_img = im.resizing_img()
    resized_rects = im.get_resized_rects()
    
    # AI Assistance Section
    if AI_AVAILABLE and st.session_state.ai_assistant is not None:
        st.write("### 🤖 AI Assistance")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Get AI Suggestions"):
                with st.spinner("Analyzing image..."):
                    try:
                        suggestions = st.session_state.ai_assistant.suggest_annotations(img_path)
                        if suggestions and 'detections' in suggestions:
                            st.session_state.ai_suggestions = suggestions['detections']
                            st.success(f"Found {len(suggestions['detections'])} objects!")
                        else:
                            st.warning("No objects detected")
                    except Exception as e:
                        st.error(f"AI analysis failed: {e}")
        
        with col2:
            if 'ai_suggestions' in st.session_state and st.session_state.ai_suggestions:
                if st.button("Apply AI Suggestions"):
                    try:
                        im.set_ai_annotations(st.session_state.ai_suggestions)
                        st.success("AI suggestions applied!")
                    except Exception as e:
                        st.error(f"Failed to apply suggestions: {e}")
    
    rects = st_img_label(resized_img, box_color="red", rects=resized_rects)

    def annotate():
        im.save_annotation()
        # Update annotation file name based on format
        if annotation_format in ["json", "coco"]:
            image_annotate_file_name = img_file_name.split(".")[0] + ".json"
        else:
            image_annotate_file_name = img_file_name.split(".")[0] + ".xml"
            
        if image_annotate_file_name not in st.session_state["annotation_files"]:
            st.session_state["annotation_files"].append(image_annotate_file_name)
        next_annotate_file()

    if rects:
        st.button(label="Save", on_click=annotate)
        preview_imgs = im.init_annotation(rects)

        for i, prev_img in enumerate(preview_imgs):
            prev_img[0].thumbnail((200, 200))
            col1, col2 = st.columns(2)
            with col1:
                col1.image(prev_img[0])
            with col2:
                default_index = 0
                if prev_img[1]:
                    default_index = labels.index(prev_img[1])

                select_label = col2.selectbox(
                    "Label", labels, key=f"label_{i}", index=default_index
                )
                im.set_annotation(i, select_label)

def main():
    """Main function"""
    st.set_page_config(
        page_title="Integrated Image Labeling & Fine-Tuning",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Integrated Image Labeling & Fine-Tuning")
    st.write("Label images and create supervised fine-tuning jobs with Vertex AI")
    
    # Initialize session state
    initialize_session_state()
    
    # Setup GCP authentication
    gcp_authenticated = setup_gcp_authentication()
    
    # Setup AI models
    setup_ai_models()
    
    # Create tabs
    tab1, tab2 = st.tabs(["📝 Image Labeling", "🎯 Fine-Tuning"])
    
    with tab1:
        st.write("## 📝 Image Labeling")
        
        # Format selection
        annotation_format = st.selectbox(
            "Annotation Format",
            ["json", "coco", "xml"],
            index=0,
            help="JSON: Custom JSON format, COCO: COCO JSON format, XML: Pascal VOC XML format"
        )
        
        custom_labels = ["", "dog", "cat"]
        
        # Add custom labels input
        st.write("Labels (comma-separated):")
        labels_input = st.text_input(
            "Custom Labels", 
            value=", ".join(custom_labels[1:]),  # Skip empty label
            help="Enter labels separated by commas"
        )
        
        if labels_input.strip():
            custom_labels = [""] + [label.strip() for label in labels_input.split(",") if label.strip()]
        
        # Run the integrated app
        run_integrated_app("img_dir", custom_labels, annotation_format)
    
    with tab2:
        create_fine_tuning_tab()

if __name__ == "__main__":
    main() 