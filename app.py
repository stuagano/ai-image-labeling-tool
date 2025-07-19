import streamlit as st
import os
from streamlit_img_label import st_img_label
from streamlit_img_label.manage import ImageManager, ImageDirManager
from core.navigation import create_navigation_controller
from core.export_manager import create_export_manager

def run(img_dir, labels, annotation_format="json"):
    st.set_option("deprecation.showfileUploaderEncoding", False)
    idm = ImageDirManager(img_dir, annotation_format)
    
    # Create centralized controllers
    nav_controller = create_navigation_controller(idm)
    export_manager = create_export_manager(img_dir, annotation_format)
    
    # Show current annotation format
    st.sidebar.write(f"Annotation format: {annotation_format.upper()}")
    
    # Render navigation controls
    nav_controller.render_navigation_sidebar()

    # Render export and validation controls
    export_manager.render_export_sidebar()
    export_manager.render_validation_sidebar()

    # Main content: annotate images
    img_file_name = idm.get_image(st.session_state["image_index"])
    img_path = os.path.join(img_dir, img_file_name)
    im = ImageManager(img_path, annotation_format)
    img = im.get_img()
    resized_img = im.resizing_img()
    resized_rects = im.get_resized_rects()
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
    st.title("Image Labeling Tool")
    st.write("Annotate images with bounding boxes and labels")
    
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
    
    run("img_dir", custom_labels, annotation_format)