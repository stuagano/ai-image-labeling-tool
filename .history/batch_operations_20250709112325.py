#!/usr/bin/env python3
"""
CLI tool for batch operations on image annotations.
Supports export, validation, and conversion operations.
"""

import argparse
import os
import sys
from streamlit_img_label.export import (
    export_coco_dataset, 
    export_yolo_format, 
    export_csv, 
    validate_annotations
)
from streamlit_img_label.annotation import convert_xml_to_json


def main():
    parser = argparse.ArgumentParser(description='Batch operations for image annotations')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Export COCO command
    export_coco_parser = subparsers.add_parser('export-coco', help='Export to COCO format')
    export_coco_parser.add_argument('img_dir', help='Directory containing images and annotations')
    export_coco_parser.add_argument('output_file', help='Output COCO JSON file')
    export_coco_parser.add_argument('--format', choices=['json', 'coco', 'xml'], default='json',
                                   help='Input annotation format')
    
    # Export YOLO command
    export_yolo_parser = subparsers.add_parser('export-yolo', help='Export to YOLO format')
    export_yolo_parser.add_argument('img_dir', help='Directory containing images and annotations')
    export_yolo_parser.add_argument('output_dir', help='Output directory for YOLO files')
    export_yolo_parser.add_argument('--format', choices=['json', 'coco', 'xml'], default='json',
                                   help='Input annotation format')
    
    # Export CSV command
    export_csv_parser = subparsers.add_parser('export-csv', help='Export to CSV format')
    export_csv_parser.add_argument('img_dir', help='Directory containing images and annotations')
    export_csv_parser.add_argument('output_file', help='Output CSV file')
    export_csv_parser.add_argument('--format', choices=['json', 'coco', 'xml'], default='json',
                                  help='Input annotation format')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate annotations')
    validate_parser.add_argument('img_dir', help='Directory containing images and annotations')
    validate_parser.add_argument('--format', choices=['json', 'coco', 'xml'], default='json',
                                help='Input annotation format')
    validate_parser.add_argument('--detailed', action='store_true', help='Show detailed issues')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert XML to JSON')
    convert_parser.add_argument('img_dir', help='Directory containing images and XML annotations')
    convert_parser.add_argument('--format', choices=['custom', 'coco'], default='custom',
                               help='Output JSON format')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'export-coco':
            print(f"Exporting {args.img_dir} to COCO format...")
            export_coco_dataset(args.img_dir, args.output_file, args.format)
            print(f"Successfully exported to {args.output_file}")
            
        elif args.command == 'export-yolo':
            print(f"Exporting {args.img_dir} to YOLO format...")
            export_yolo_format(args.img_dir, args.output_dir, args.format)
            print(f"Successfully exported to {args.output_dir}")
            
        elif args.command == 'export-csv':
            print(f"Exporting {args.img_dir} to CSV format...")
            export_csv(args.img_dir, args.output_file, args.format)
            print(f"Successfully exported to {args.output_file}")
            
        elif args.command == 'validate':
            print(f"Validating annotations in {args.img_dir}...")
            stats = validate_annotations(args.img_dir, args.format)
            
            print(f"\nValidation Results:")
            print(f"Total images: {stats['total_images']}")
            print(f"Annotated images: {stats['annotated_images']}")
            print(f"Total annotations: {stats['total_annotations']}")
            print(f"Empty labels: {stats['empty_labels']}")
            print(f"Overlapping boxes: {stats['overlapping_boxes']}")
            
            if stats['label_distribution']:
                print(f"\nLabel distribution:")
                for label, count in sorted(stats['label_distribution'].items(), key=lambda x: x[1], reverse=True):
                    print(f"  {label}: {count}")
            
            if stats['issues'] and args.detailed:
                print(f"\nIssues found:")
                for issue in stats['issues']:
                    print(f"  • {issue}")
            elif stats['issues']:
                print(f"\nFound {len(stats['issues'])} issues. Use --detailed to see all issues.")
                
        elif args.command == 'convert':
            print(f"Converting XML annotations in {args.img_dir} to {args.format} JSON...")
            # Use the existing conversion script
            from convert_annotations import convert_directory
            convert_directory(args.img_dir, args.format)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 