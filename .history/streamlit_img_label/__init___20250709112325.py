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
    component_value = _component_func(
        img=img,
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
