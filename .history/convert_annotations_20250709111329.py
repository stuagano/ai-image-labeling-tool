#!/usr/bin/env python3
"""
Utility script to convert existing XML annotations to JSON format.
Supports both custom JSON and COCO JSON formats.
"""

import os
import argparse
from streamlit_img_label.annotation import convert_xml_to_json


def convert_directory(img_dir, format_type="custom"):
    """
    Convert all XML annotations in a directory to JSON format.
    
    Args:
        img_dir (str): Directory containing images and XML annotations
        format_type (str): "custom" or "coco" format
    """
    if not os.path.exists(img_dir):
        print(f"Directory {img_dir} does not exist!")
        return
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for file in os.listdir(img_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    print(f"Found {len(image_files)} image files in {img_dir}")
    
    converted_count = 0
    for img_file in image_files:
        img_path = os.path.join(img_dir, img_file)
        xml_file = img_path.replace('.', '_').rsplit('.', 1)[0] + '.xml'
        
        if os.path.exists(xml_file):
            try:
                convert_xml_to_json(img_path, format_type)
                print(f"Converted {img_file} to {format_type} JSON format")
                converted_count += 1
            except Exception as e:
                print(f"Error converting {img_file}: {e}")
        else:
            print(f"No XML annotation found for {img_file}")
    
    print(f"\nConversion complete! Converted {converted_count} files to {format_type} JSON format.")


def main():
    parser = argparse.ArgumentParser(description='Convert XML annotations to JSON format')
    parser.add_argument('img_dir', help='Directory containing images and XML annotations')
    parser.add_argument('--format', choices=['custom', 'coco'], default='custom',
                       help='JSON format: custom (simple) or coco (COCO format)')
    
    args = parser.parse_args()
    
    print(f"Converting XML annotations in {args.img_dir} to {args.format} JSON format...")
    convert_directory(args.img_dir, args.format)


if __name__ == "__main__":
    main() 