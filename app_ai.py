import streamlit as st
import os
from streamlit_img_label import st_img_label
from streamlit_img_label.manage import ImageDirManager
from streamlit_img_label.export import export_coco_dataset, export_yolo_format, export_csv, validate_annotations
from ai_utils import create_ai_detector, create_ai_assistant
from ai_image_manager import AIImageManager
import json
from typing import List, Dict
import time

def initialize_ai_models():
    """Initialize AI models based on user selection"""
    if "ai_detector" not in st.session_state:
        st.session_state.ai_detector = None
        st.session_state.ai_assistant = None

def run_ai_enhanced(img_dir, labels, annotation_format="json"):
    st.set_option("deprecation.showfileUploaderEncoding", False)
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

    # AI Configuration Section
    st.sidebar.write("---")
    st.sidebar.write("**🤖 AI Configuration**")
    
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
    im = AIImageManager(img_path, annotation_format)
    img = im.get_img()
    resized_img = im.resizing_img()
    resized_rects = im.get_resized_rects()
    
    # AI Assistance Section
    if st.session_state.ai_assistant is not None:
        st.write("---")
        st.write("**🤖 AI Assistance**")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🔍 Get AI Suggestions"):
                with st.spinner("Analyzing image with AI..."):
                    try:
                        suggestions = st.session_state.ai_assistant.suggest_annotations(
                            img_path, 
                            labels[1:] if len(labels) > 1 else None  # Skip empty label
                        )
                        
                        st.session_state.ai_suggestions = suggestions
                        st.success("AI analysis complete!")
                        
                    except Exception as e:
                        st.error(f"AI analysis failed: {e}")
        
        with col2:
            if st.button("✅ Apply AI Suggestions"):
                if hasattr(st.session_state, 'ai_suggestions'):
                    suggestions = st.session_state.ai_suggestions
                    detections = suggestions.get('detections', [])
                    
                    if detections:
                        # Convert AI detections to annotation format
                        ai_annotations = []
                        for det in detections:
                            bbox = det['bbox']
                            # Convert to resized coordinates
                            scale_x = resized_img.width / img.width
                            scale_y = resized_img.height / img.height
                            
                            resized_bbox = [
                                int(bbox[0] * scale_x),
                                int(bbox[1] * scale_y),
                                int(bbox[2] * scale_x),
                                int(bbox[3] * scale_y)
                            ]
                            
                            ai_annotations.append({
                                'left': resized_bbox[0],
                                'top': resized_bbox[1],
                                'width': resized_bbox[2],
                                'height': resized_bbox[3],
                                'label': det['label'],
                                'confidence': det.get('confidence', 0.5)
                            })
                        
                        # Update the image manager with AI annotations
                        im.set_ai_annotations(ai_annotations)
                        st.success(f"Applied {len(ai_annotations)} AI suggestions!")
                    else:
                        st.warning("No AI detections found.")
                else:
                    st.warning("Please get AI suggestions first.")
        
        # Display AI suggestions if available
        if hasattr(st.session_state, 'ai_suggestions'):
            suggestions = st.session_state.ai_suggestions
            
            with st.expander("📊 AI Analysis Results", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Detected Objects:**")
                    detections = suggestions.get('detections', [])
                    if detections:
                        for i, det in enumerate(detections[:5]):  # Show first 5
                            st.write(f"• {det['label']} (conf: {det['confidence']:.2f})")
                        if len(detections) > 5:
                            st.write(f"... and {len(detections) - 5} more")
                    else:
                        st.write("No objects detected")
                
                with col2:
                    st.write("**Suggested Labels:**")
                    suggested_labels = suggestions.get('suggested_labels', [])
                    if suggested_labels:
                        for label in suggested_labels[:5]:
                            st.write(f"• {label}")
                        if len(suggested_labels) > 5:
                            st.write(f"... and {len(suggested_labels) - 5} more")
                    else:
                        st.write("No label suggestions")
                
                # Confidence distribution
                confidence_scores = suggestions.get('confidence_scores', [])
                if confidence_scores:
                    st.write("**Confidence Distribution:**")
                    avg_conf = sum(confidence_scores) / len(confidence_scores)
                    st.write(f"Average confidence: {avg_conf:.2f}")
                    st.write(f"High confidence (>0.8): {sum(1 for c in confidence_scores if c > 0.8)}")
                    st.write(f"Medium confidence (0.5-0.8): {sum(1 for c in confidence_scores if 0.5 <= c <= 0.8)}")
                    st.write(f"Low confidence (<0.5): {sum(1 for c in confidence_scores if c < 0.5)}")
        
        # Show annotation statistics
        if st.button("📈 Show Annotation Statistics"):
            stats = im.get_annotation_statistics()
            validation = im.validate_ai_annotations()
            
            st.write("**Annotation Statistics:**")
            st.write(f"Total annotations: {stats['total_annotations']}")
            st.write(f"Manual annotations: {stats['manual_annotations']}")
            st.write(f"AI annotations: {stats['ai_annotations']}")
            
            if stats['labels']:
                st.write("**Label distribution:**")
                for label, count in sorted(stats['labels'].items(), key=lambda x: x[1], reverse=True):
                    st.write(f"• {label}: {count}")
            
            if validation['total_ai_annotations'] > 0:
                st.write("**AI Annotation Quality:**")
                st.write(f"High confidence: {validation['high_confidence']}")
                st.write(f"Medium confidence: {validation['medium_confidence']}")
                st.write(f"Low confidence: {validation['low_confidence']}")
                
                if validation['potential_issues']:
                    st.write("**Potential Issues:**")
                    for issue in validation['potential_issues'][:3]:
                        st.write(f"• {issue}")

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

if __name__ == "__main__":
    st.title("🤖 AI-Enhanced Image Labeling Tool")
    st.write("Annotate images with AI-powered object detection and labeling assistance")
    
    # Initialize AI models
    initialize_ai_models()
    
    # Format selection
    annotation_format = st.sidebar.selectbox(
        "Annotation Format",
        ["json", "coco", "xml"],
        index=0,
        help="JSON: Custom JSON format, COCO: COCO JSON format, XML: Pascal VOC XML format"
    )
    
    custom_labels = ["", "dog", "cat"]
    
    # Add custom labels input
    st.sidebar.write("---")
    st.sidebar.write("Labels (comma-separated):")
    labels_input = st.sidebar.text_input(
        "Custom Labels", 
        value=", ".join(custom_labels[1:]),  # Skip empty label
        help="Enter labels separated by commas"
    )
    
    if labels_input.strip():
        custom_labels = [""] + [label.strip() for label in labels_input.split(",") if label.strip()]
    
    run_ai_enhanced("img_dir", custom_labels, annotation_format) 