import os
import streamlit.components.v1 as components
import numpy as np
from .manage import ImageManager, ImageDirManager
from .annotation import read_json, output_json, read_xml, output_xml, convert_xml_to_json
from .export import export_coco_dataset, export_yolo_format, export_csv, validate_annotations

# Frontend component
_RELEASE = True

if not _RELEASE:
    _component_func = components.declare_component(
        "st_img_label",
        url="http://localhost:3001",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component("st_img_label", path=build_dir)


def st_img_label(img, box_color="red", rects=None, key=None):
    """Create a new instance of "st_img_label".

    Parameters
    ----------
    img: PIL.Image
        The image to be displayed and annotated
    box_color: str
        The color of the bounding box
    rects: list
        List of existing rectangles to display
    key: str or None
        An optional key that uniquely identifies this component. If this is
        None, and the component's arguments are changed, the component will
        be re-mounted in the Streamlit frontend and lose its current state.

    Returns
    -------
    list
        List of rectangles with coordinates and labels
    """
    # Convert PIL Image to numpy array and then to the format expected by frontend
    import numpy as np
    
    # Convert PIL Image to RGB if it's not already
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Get dimensions
    canvas_height, canvas_width = img_array.shape[:2]
    
    # Flatten the array to match frontend expectations
    # Frontend expects RGBA format, so we need to add alpha channel
    if img_array.shape[2] == 3:
        # Add alpha channel (255 for fully opaque)
        rgba_array = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)
        rgba_array[:, :, :3] = img_array
        rgba_array[:, :, 3] = 255
        image_data = rgba_array.flatten().tolist()
    else:
        image_data = img_array.flatten().tolist()
    
    component_value = _component_func(
        canvasWidth=canvas_width,
        canvasHeight=canvas_height,
        imageData=image_data,
        box_color=box_color,
        rects=rects or [],
        key=key,
        default=[],
    )

    return component_value


# Export the main classes and functions for easy access
__all__ = [
    'st_img_label',
    'ImageManager',
    'ImageDirManager',
    'read_json',
    'output_json',
    'read_xml',
    'output_xml',
    'convert_xml_to_json',
    'export_coco_dataset',
    'export_yolo_format',
    'export_csv',
    'validate_annotations'
]
