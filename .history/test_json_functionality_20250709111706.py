#!/usr/bin/env python3
"""
Test script to verify JSON annotation functionality.
"""

import json
import os
from PIL import Image
from streamlit_img_label.annotation import read_json, output_json, _parse_coco_format

def test_custom_json():
    """Test custom JSON format."""
    print("Testing custom JSON format...")
    
    # Create test data
    test_rects = [
        {"left": 100, "top": 50, "width": 200, "height": 150, "label": "dog"},
        {"left": 350, "top": 100, "width": 180, "height": 120, "label": "cat"}
    ]
    
    # Create a dummy image
    img = Image.new('RGB', (800, 600), color='white')
    
    # Test output
    output_json("test_image.jpg", img, test_rects, "custom")
    
    # Test reading
    loaded_rects = read_json("test_image.jpg")
    
    print(f"Original: {test_rects}")
    print(f"Loaded: {loaded_rects}")
    print(f"Match: {test_rects == loaded_rects}")
    
    # Cleanup
    if os.path.exists("test_image.json"):
        os.remove("test_image.json")
    
    print("Custom JSON test completed!\n")

def test_coco_json():
    """Test COCO JSON format."""
    print("Testing COCO JSON format...")
    
    # Create test data
    test_rects = [
        {"left": 100, "top": 50, "width": 200, "height": 150, "label": "dog"},
        {"left": 350, "top": 100, "width": 180, "height": 120, "label": "cat"}
    ]
    
    # Create a dummy image
    img = Image.new('RGB', (800, 600), color='white')
    
    # Test output
    output_json("test_image.jpg", img, test_rects, "coco")
    
    # Test reading
    loaded_rects = read_json("test_image.jpg")
    
    print(f"Original: {test_rects}")
    print(f"Loaded: {loaded_rects}")
    print(f"Match: {len(test_rects) == len(loaded_rects)}")
    
    # Cleanup
    if os.path.exists("test_image.json"):
        os.remove("test_image.json")
    
    print("COCO JSON test completed!\n")

def test_coco_parsing():
    """Test COCO format parsing."""
    print("Testing COCO format parsing...")
    
    # Sample COCO data
    coco_data = {
        "info": {"description": "Test", "version": "1.0"},
        "images": [
            {"id": 1, "file_name": "test_image.jpg", "width": 800, "height": 600}
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 50, 200, 150],
                "area": 30000,
                "iscrowd": 0
            }
        ],
        "categories": [
            {"id": 1, "name": "dog", "supercategory": "animal"}
        ]
    }
    
    # Test parsing
    rects = _parse_coco_format(coco_data, "test_image.jpg")
    print(f"Parsed rects: {rects}")
    print("COCO parsing test completed!\n")

if __name__ == "__main__":
    print("Running JSON functionality tests...\n")
    
    test_custom_json()
    test_coco_json()
    test_coco_parsing()
    
    print("All tests completed successfully!") 