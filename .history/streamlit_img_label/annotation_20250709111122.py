import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

"""
.. module:: streamlit_img_label
   :synopsis: annotation.
.. moduleauthor:: Tianning Li <ltianningli@gmail.com>
"""


def read_json(img_file: str) -> List[Dict[str, Any]]:
    """read_json
    Read the JSON annotation file and extract the bounding boxes if exists.

    Args:
        img_file(str): the image file.
    Returns:
        rects(list): the bounding boxes of the image.
    """
    file_name = img_file.split(".")[0]
    json_file = f"{file_name}.json"
    
    if not os.path.isfile(json_file):
        return []
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Check if it's COCO format
        if isinstance(data, dict) and 'annotations' in data:
            return _parse_coco_format(data, img_file)
        else:
            # Custom JSON format
            return data if isinstance(data, list) else []
            
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def _parse_coco_format(coco_data: Dict[str, Any], img_file: str) -> List[Dict[str, Any]]:
    """Parse COCO format annotations for a specific image."""
    rects = []
    
    # Find the image info
    img_filename = os.path.basename(img_file)
    image_info = None
    for img in coco_data.get('images', []):
        if img.get('file_name') == img_filename:
            image_info = img
            break
    
    if not image_info:
        return rects
    
    # Get annotations for this image
    image_id = image_info['id']
    for ann in coco_data.get('annotations', []):
        if ann.get('image_id') == image_id:
            bbox = ann.get('bbox', [])
            if len(bbox) == 4:
                x, y, width, height = bbox
                category_id = ann.get('category_id')
                category_name = ""
                
                # Find category name
                for cat in coco_data.get('categories', []):
                    if cat.get('id') == category_id:
                        category_name = cat.get('name', '')
                        break
                
                rects.append({
                    "left": int(x),
                    "top": int(y),
                    "width": int(width),
                    "height": int(height),
                    "label": category_name,
                    "category_id": category_id
                })
    
    return rects


def output_json(img_file: str, img, rects: List[Dict[str, Any]], format_type: str = "custom") -> None:
    """output_json
    Output the JSON image annotation file

    Args:
        img_file(str): the image file.
        img(PIL.Image): the image object.
        rects(list): the bounding boxes of the image.
        format_type(str): "custom" or "coco" format.
    """
    file_name = img_file.split(".")[0]
    
    if format_type == "coco":
        _output_coco_json(img_file, img, rects, file_name)
    else:
        _output_custom_json(rects, file_name)


def _output_custom_json(rects: List[Dict[str, Any]], file_name: str) -> None:
    """Output custom JSON format."""
    with open(f"{file_name}.json", 'w') as f:
        json.dump(rects, f, indent=2)


def _output_coco_json(img_file: str, img, rects: List[Dict[str, Any]], file_name: str) -> None:
    """Output COCO JSON format."""
    img_filename = os.path.basename(img_file)
    
    # Create COCO format structure
    coco_data = {
        "info": {
            "description": "Image annotations in COCO format",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [],
        "images": [
            {
                "id": 1,
                "file_name": img_filename,
                "width": img.width,
                "height": img.height
            }
        ],
        "annotations": [],
        "categories": []
    }
    
    # Add categories and annotations
    category_id_map = {}
    next_category_id = 1
    next_annotation_id = 1
    
    for rect in rects:
        label = rect.get("label", "")
        
        # Add category if not exists
        if label not in category_id_map:
            category_id_map[label] = next_category_id
            coco_data["categories"].append({
                "id": next_category_id,
                "name": label,
                "supercategory": "object"
            })
            next_category_id += 1
        
        # Add annotation
        coco_data["annotations"].append({
            "id": next_annotation_id,
            "image_id": 1,
            "category_id": category_id_map[label],
            "bbox": [rect["left"], rect["top"], rect["width"], rect["height"]],
            "area": rect["width"] * rect["height"],
            "iscrowd": 0
        })
        next_annotation_id += 1
    
    with open(f"{file_name}.json", 'w') as f:
        json.dump(coco_data, f, indent=2)


def convert_xml_to_json(img_file: str, format_type: str = "custom") -> None:
    """Convert existing XML annotation to JSON format."""
    from .annotation import read_xml  # Import the old XML reader
    
    rects = read_xml(img_file)
    if rects:
        from PIL import Image # Import Image for backward compatibility
        img = Image.open(img_file)
        output_json(img_file, img, rects, format_type)


# Legacy XML functions for backward compatibility
def read_xml(img_file):
    """read_xml
    Read the xml annotation file and extract the bounding boxes if exists.

    Args:
        img_file(str): the image file.
    Returns:
        rects(list): the bounding boxes of the image.
    """
    file_name = img_file.split(".")[0]
    if not os.path.isfile(f"{file_name}.xml"):
        return []
    
    try:
        from xml.etree import ElementTree as ET
        tree = ET.parse(f"{file_name}.xml")
        root = tree.getroot()

        rects = []

        for boxes in root.iter("object"):
            label = boxes.find("name").text
            ymin = int(boxes.find("bndbox/ymin").text)
            xmin = int(boxes.find("bndbox/xmin").text)
            ymax = int(boxes.find("bndbox/ymax").text)
            xmax = int(boxes.find("bndbox/xmax").text)
            rects.append(
                {
                    "left": xmin,
                    "top": ymin,
                    "width": xmax - xmin,
                    "height": ymax - ymin,
                    "label": label,
                }
            )
        return rects
    except Exception:
        return []


def output_xml(img_file, img, rects):
    """output_xml
    Output the xml image annotation file

    Args:
        img_file(str): the image file.
        img(PIL.Image): the image object.
        rects(list): the bounding boxes of the image.
    """
    try:
        from pascal_voc_writer import Writer
        file_name = img_file.split(".")[0]
        writer = Writer(img_file, img.width, img.height)
        for box in rects:
            xmin = box["left"]
            ymin = box["top"]
            xmax = box["left"] + box["width"]
            ymax = box["top"] + box["height"]

            writer.addObject(box["label"], xmin, ymin, xmax, ymax)
        writer.save(f"{file_name}.xml")
    except ImportError:
        print("pascal_voc_writer not available for XML output")
