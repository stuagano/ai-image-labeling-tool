import streamlit as st
import os
from streamlit_img_label import st_img_label
from streamlit_img_label.manage import ImageManager, ImageDirManager
from streamlit_img_label.export import export_coco_dataset, export_yolo_format, export_csv, validate_annotations

def run(img_dir, labels, annotation_format="json"):
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