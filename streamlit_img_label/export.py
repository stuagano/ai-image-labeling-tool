"""
Export utilities for image annotations.
Supports multiple formats including COCO, YOLO, and CSV.
"""

import json
import csv
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from PIL import Image


def export_coco_dataset(img_dir: str, output_file: str, format_type: str = "json") -> None:
    """
    Export all annotations in a directory to a single COCO format file.
    
    Args:
        img_dir: Directory containing images and annotations
        output_file: Output file path
        format_type: "json" or "coco" (both produce COCO format)
    """
    from .annotation import read_json, read_xml
    from .manage import ImageDirManager
    
    idm = ImageDirManager(img_dir, format_type)
    image_files = idm.get_all_files()
    
    # Initialize COCO structure
    coco_data = {
        "info": {
            "description": f"Dataset exported from {img_dir}",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Track categories and IDs
    category_map = {}
    next_category_id = 1
    next_annotation_id = 1
    next_image_id = 1
    
    for img_file in image_files:
        img_path = os.path.join(img_dir, img_file)
        
        # Load image info
        try:
            img = Image.open(img_path)
            image_info = {
                "id": next_image_id,
                "file_name": img_file,
                "width": img.width,
                "height": img.height
            }
            coco_data["images"].append(image_info)
            
            # Load annotations
            if format_type in ["json", "coco"]:
                rects = read_json(img_path)
            else:
                rects = read_xml(img_path)
            
            # Add annotations
            for rect in rects:
                label = rect.get("label", "")
                
                # Add category if not exists
                if label and label not in category_map:
                    category_map[label] = next_category_id
                    coco_data["categories"].append({
                        "id": next_category_id,
                        "name": label,
                        "supercategory": "object"
                    })
                    next_category_id += 1
                
                if label:
                    annotation = {
                        "id": next_annotation_id,
                        "image_id": next_image_id,
                        "category_id": category_map[label],
                        "bbox": [rect["left"], rect["top"], rect["width"], rect["height"]],
                        "area": rect["width"] * rect["height"],
                        "iscrowd": 0
                    }
                    coco_data["annotations"].append(annotation)
                    next_annotation_id += 1
            
            next_image_id += 1
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"Exported {len(coco_data['images'])} images with {len(coco_data['annotations'])} annotations to {output_file}")


def export_yolo_format(img_dir: str, output_dir: str, format_type: str = "json") -> None:
    """
    Export annotations to YOLO format.
    
    Args:
        img_dir: Directory containing images and annotations
        output_dir: Output directory for YOLO files
        format_type: Input annotation format
    """
    from .annotation import read_json, read_xml
    from .manage import ImageDirManager
    
    os.makedirs(output_dir, exist_ok=True)
    
    idm = ImageDirManager(img_dir, format_type)
    image_files = idm.get_all_files()
    
    # Create label mapping
    label_map = {}
    next_label_id = 0
    
    for img_file in image_files:
        img_path = os.path.join(img_dir, img_file)
        
        try:
            img = Image.open(img_path)
            img_width, img_height = img.size
            
            # Load annotations
            if format_type in ["json", "coco"]:
                rects = read_json(img_path)
            else:
                rects = read_xml(img_path)
            
            # Create YOLO annotation file
            yolo_file = os.path.join(output_dir, os.path.splitext(img_file)[0] + '.txt')
            
            with open(yolo_file, 'w') as f:
                for rect in rects:
                    label = rect.get("label", "")
                    if not label:
                        continue
                    
                    # Add label to mapping
                    if label not in label_map:
                        label_map[label] = next_label_id
                        next_label_id += 1
                    
                    # Convert to YOLO format (normalized coordinates)
                    x_center = (rect["left"] + rect["width"] / 2) / img_width
                    y_center = (rect["top"] + rect["height"] / 2) / img_height
                    width = rect["width"] / img_width
                    height = rect["height"] / img_height
                    
                    f.write(f"{label_map[label]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            # Copy image to output directory
            import shutil
            shutil.copy2(img_path, os.path.join(output_dir, img_file))
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    # Save label mapping
    with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
        for label, label_id in sorted(label_map.items(), key=lambda x: x[1]):
            f.write(f"{label}\n")
    
    print(f"Exported {len(image_files)} images to YOLO format in {output_dir}")


def export_csv(img_dir: str, output_file: str, format_type: str = "json") -> None:
    """
    Export annotations to CSV format for easy analysis.
    
    Args:
        img_dir: Directory containing images and annotations
        output_file: Output CSV file path
        format_type: Input annotation format
    """
    from .annotation import read_json, read_xml
    from .manage import ImageDirManager
    
    idm = ImageDirManager(img_dir, format_type)
    image_files = idm.get_all_files()
    
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['image_file', 'label', 'x', 'y', 'width', 'height', 'area']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for img_file in image_files:
            img_path = os.path.join(img_dir, img_file)
            
            try:
                # Load annotations
                if format_type in ["json", "coco"]:
                    rects = read_json(img_path)
                else:
                    rects = read_xml(img_path)
                
                # Write annotations
                for rect in rects:
                    label = rect.get("label", "")
                    if label:
                        writer.writerow({
                            'image_file': img_file,
                            'label': label,
                            'x': rect["left"],
                            'y': rect["top"],
                            'width': rect["width"],
                            'height': rect["height"],
                            'area': rect["width"] * rect["height"]
                        })
                        
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
    
    print(f"Exported annotations to {output_file}")


def validate_annotations(img_dir: str, format_type: str = "json") -> Dict[str, Any]:
    """
    Validate annotation quality and return statistics.
    
    Args:
        img_dir: Directory containing images and annotations
        format_type: Input annotation format
    
    Returns:
        Dictionary with validation results and statistics
    """
    from .annotation import read_json, read_xml
    from .manage import ImageDirManager
    
    idm = ImageDirManager(img_dir, format_type)
    image_files = idm.get_all_files()
    
    stats = {
        'total_images': len(image_files),
        'annotated_images': 0,
        'total_annotations': 0,
        'label_distribution': {},
        'issues': [],
        'empty_labels': 0,
        'overlapping_boxes': 0
    }
    
    for img_file in image_files:
        img_path = os.path.join(img_dir, img_file)
        
        try:
            img = Image.open(img_path)
            
            # Load annotations
            if format_type in ["json", "coco"]:
                rects = read_json(img_path)
            else:
                rects = read_xml(img_path)
            
            if rects:
                stats['annotated_images'] += 1
                stats['total_annotations'] += len(rects)
                
                # Check for issues
                for rect in rects:
                    label = rect.get("label", "")
                    
                    if not label:
                        stats['empty_labels'] += 1
                        stats['issues'].append(f"{img_file}: Empty label")
                    else:
                        stats['label_distribution'][label] = stats['label_distribution'].get(label, 0) + 1
                
                # Check for overlapping boxes (simple check)
                for i, rect1 in enumerate(rects):
                    for j, rect2 in enumerate(rects[i+1:], i+1):
                        if _boxes_overlap(rect1, rect2):
                            stats['overlapping_boxes'] += 1
                            stats['issues'].append(f"{img_file}: Overlapping boxes {i} and {j}")
                            
        except Exception as e:
            stats['issues'].append(f"{img_file}: Error - {e}")
    
    return stats


def _boxes_overlap(box1: Dict[str, Any], box2: Dict[str, Any]) -> bool:
    """Check if two bounding boxes overlap."""
    x1_1, y1_1 = box1["left"], box1["top"]
    x2_1, y2_1 = box1["left"] + box1["width"], box1["top"] + box1["height"]
    
    x1_2, y1_2 = box2["left"], box2["top"]
    x2_2, y2_2 = box2["left"] + box2["width"], box2["top"] + box2["height"]
    
    return not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1) 