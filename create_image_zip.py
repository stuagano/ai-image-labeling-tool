#!/usr/bin/env python3
"""
Create Image ZIP for Cloud Labeling Tool
Utility script to create ZIP files from folders of images for bulk upload
"""

import os
import zipfile
import argparse
from pathlib import Path
import glob
from typing import List

def find_image_files(directory: str) -> List[str]:
    """
    Find all image files in a directory and subdirectories
    
    Args:
        directory: Directory to search
        
    Returns:
        List of image file paths
    """
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']
    image_files = []
    
    for ext in image_extensions:
        # Find files with this extension (case insensitive)
        pattern = os.path.join(directory, '**', ext)
        image_files.extend(glob.glob(pattern, recursive=True))
        pattern = os.path.join(directory, '**', ext.upper())
        image_files.extend(glob.glob(pattern, recursive=True))
    
    return sorted(image_files)

def create_image_zip(input_dir: str, output_zip: str, max_files: int = None) -> bool:
    """
    Create a ZIP file containing all images from a directory
    
    Args:
        input_dir: Directory containing images
        output_zip: Output ZIP file path
        max_files: Maximum number of files to include (None for all)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Find all image files
        image_files = find_image_files(input_dir)
        
        if not image_files:
            print(f"No image files found in {input_dir}")
            return False
        
        # Limit files if specified
        if max_files and len(image_files) > max_files:
            image_files = image_files[:max_files]
            print(f"Limited to {max_files} files")
        
        print(f"Found {len(image_files)} image files")
        
        # Create ZIP file
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for img_file in image_files:
                # Get relative path for ZIP
                rel_path = os.path.relpath(img_file, input_dir)
                zipf.write(img_file, rel_path)
                print(f"Added: {rel_path}")
        
        print(f"\n✅ Successfully created {output_zip}")
        print(f"📦 Total files: {len(image_files)}")
        print(f"📏 File size: {os.path.getsize(output_zip) / (1024*1024):.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating ZIP file: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Create ZIP file from folder of images for cloud labeling tool"
    )
    
    parser.add_argument(
        "input_dir",
        help="Directory containing images"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="images_for_labeling.zip",
        help="Output ZIP file name (default: images_for_labeling.zip)"
    )
    
    parser.add_argument(
        "-m", "--max-files",
        type=int,
        help="Maximum number of files to include"
    )
    
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list found images, don't create ZIP"
    )
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"❌ Directory not found: {args.input_dir}")
        return 1
    
    # Find image files
    image_files = find_image_files(args.input_dir)
    
    if not image_files:
        print(f"❌ No image files found in {args.input_dir}")
        return 1
    
    print(f"📁 Found {len(image_files)} image files in {args.input_dir}")
    
    # List files if requested
    if args.list_only:
        print("\n📋 Image files found:")
        for i, img_file in enumerate(image_files, 1):
            rel_path = os.path.relpath(img_file, args.input_dir)
            file_size = os.path.getsize(img_file) / 1024  # KB
            print(f"  {i:3d}. {rel_path} ({file_size:.1f} KB)")
        return 0
    
    # Create ZIP file
    success = create_image_zip(args.input_dir, args.output, args.max_files)
    
    if success:
        print(f"\n🚀 Ready to upload to Cloud Labeling Tool!")
        print(f"📤 Upload '{args.output}' using the 'ZIP Folder' option")
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit(main()) 