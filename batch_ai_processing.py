#!/usr/bin/env python3
"""
Batch AI processing for automatic image annotation
"""

import os
import argparse
import json
from typing import List, Dict, Any
from ai_utils import create_ai_detector, create_ai_assistant
from ai_image_manager import AIImageManager
from streamlit_img_label.manage import ImageDirManager
import time
from tqdm import tqdm

def batch_ai_annotation(
    img_dir: str,
    model_type: str = "yolo",
    api_key: str = None,
    confidence_threshold: float = 0.5,
    annotation_format: str = "json",
    output_dir: str = None,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Process all images in a directory with AI annotation
    
    Args:
        img_dir: Directory containing images
        model_type: AI model type ("yolo", "transformers", "gemini")
        api_key: API key for cloud services
        confidence_threshold: Minimum confidence for detections
        annotation_format: Output annotation format
        output_dir: Output directory (if different from img_dir)
        overwrite: Whether to overwrite existing annotations
        
    Returns:
        Processing statistics
    """
    
    # Initialize AI
    print(f"Initializing AI model: {model_type}")
    detector = create_ai_detector(model_type, api_key)
    assistant = create_ai_assistant(detector)
    
    # Setup directories
    if output_dir is None:
        output_dir = img_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image files
    idm = ImageDirManager(img_dir, annotation_format)
    image_files = idm.get_all_files()
    
    print(f"Found {len(image_files)} images to process")
    
    # Processing statistics
    stats = {
        'total_images': len(image_files),
        'processed_images': 0,
        'skipped_images': 0,
        'total_annotations': 0,
        'ai_annotations': 0,
        'manual_annotations': 0,
        'errors': [],
        'processing_time': 0
    }
    
    start_time = time.time()
    
    # Process each image
    for img_file in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(img_dir, img_file)
        
        try:
            # Check if annotation already exists
            annotation_file = os.path.splitext(img_file)[0]
            if annotation_format in ["json", "coco"]:
                annotation_file += ".json"
            else:
                annotation_file += ".xml"
            
            annotation_path = os.path.join(output_dir, annotation_file)
            
            if os.path.exists(annotation_path) and not overwrite:
                stats['skipped_images'] += 1
                continue
            
            # Load image manager
            im = AIImageManager(img_path, annotation_format)
            
            # Get AI suggestions
            suggestions = assistant.suggest_annotations(img_path)
            detections = suggestions.get('detections', [])
            
            if detections:
                # Convert AI detections to annotation format
                ai_annotations = []
                for det in detections:
                    if det.get('confidence', 0) >= confidence_threshold:
                        bbox = det['bbox']
                        ai_annotations.append({
                            'left': bbox[0],
                            'top': bbox[1],
                            'width': bbox[2],
                            'height': bbox[3],
                            'label': det['label'],
                            'confidence': det.get('confidence', 0.5)
                        })
                
                # Apply AI annotations
                im.set_ai_annotations(ai_annotations)
                stats['ai_annotations'] += len(ai_annotations)
            
            # Save annotations
            im.save_annotation()
            stats['processed_images'] += 1
            
            # Update total annotations
            current_stats = im.get_annotation_statistics()
            stats['total_annotations'] += current_stats['total_annotations']
            stats['manual_annotations'] += current_stats['manual_annotations']
            
        except Exception as e:
            error_msg = f"Error processing {img_file}: {e}"
            stats['errors'].append(error_msg)
            print(f"Error: {error_msg}")
    
    stats['processing_time'] = time.time() - start_time
    
    return stats

def batch_ai_validation(
    img_dir: str,
    model_type: str = "yolo",
    api_key: str = None,
    annotation_format: str = "json"
) -> Dict[str, Any]:
    """
    Validate existing annotations using AI
    
    Args:
        img_dir: Directory containing images and annotations
        model_type: AI model type
        api_key: API key for cloud services
        annotation_format: Annotation format
        
    Returns:
        Validation results
    """
    
    # Initialize AI
    print(f"Initializing AI model for validation: {model_type}")
    detector = create_ai_detector(model_type, api_key)
    assistant = create_ai_assistant(detector)
    
    # Get image files
    idm = ImageDirManager(img_dir, annotation_format)
    image_files = idm.get_all_files()
    
    validation_results = {
        'total_images': len(image_files),
        'validated_images': 0,
        'potential_misses': [],
        'potential_duplicates': [],
        'confidence_analysis': [],
        'errors': []
    }
    
    for img_file in tqdm(image_files, desc="Validating annotations"):
        img_path = os.path.join(img_dir, img_file)
        
        try:
            # Load existing annotations
            im = AIImageManager(img_path, annotation_format)
            existing_annotations = im.get_rects()
            
            if not existing_annotations:
                continue
            
            # Get AI validation
            validation = assistant.validate_annotations(img_path, existing_annotations)
            
            validation_results['validated_images'] += 1
            
            # Collect potential misses
            for miss in validation.get('potential_misses', []):
                validation_results['potential_misses'].append({
                    'image': img_file,
                    'detection': miss
                })
            
            # Collect confidence analysis
            for ann in existing_annotations:
                if ann.get('source') == 'ai':
                    validation_results['confidence_analysis'].append({
                        'image': img_file,
                        'label': ann.get('label', ''),
                        'confidence': ann.get('confidence', 0.0)
                    })
                    
        except Exception as e:
            error_msg = f"Error validating {img_file}: {e}"
            validation_results['errors'].append(error_msg)
            print(f"Error: {error_msg}")
    
    return validation_results

def main():
    parser = argparse.ArgumentParser(description="Batch AI processing for image annotation")
    parser.add_argument("img_dir", help="Directory containing images")
    parser.add_argument("--model", default="yolo", choices=["yolo", "transformers", "gemini"],
                       help="AI model to use")
    parser.add_argument("--api-key", help="API key for cloud services")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Confidence threshold for detections")
    parser.add_argument("--format", default="json", choices=["json", "coco", "xml"],
                       help="Annotation format")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing annotations")
    parser.add_argument("--validate", action="store_true",
                       help="Validate existing annotations instead of creating new ones")
    parser.add_argument("--output-file", help="Output file for results")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.img_dir):
        print(f"Error: Directory {args.img_dir} does not exist!")
        return
    
    try:
        if args.validate:
            print("Running AI validation...")
            results = batch_ai_validation(
                args.img_dir,
                args.model,
                args.api_key,
                args.format
            )
        else:
            print("Running batch AI annotation...")
            results = batch_ai_annotation(
                args.img_dir,
                args.model,
                args.api_key,
                args.confidence,
                args.format,
                args.output_dir,
                args.overwrite
            )
        
        # Print results
        print("\n" + "="*50)
        print("PROCESSING RESULTS")
        print("="*50)
        
        for key, value in results.items():
            if key != 'errors':
                print(f"{key}: {value}")
        
        if results.get('errors'):
            print(f"\nErrors ({len(results['errors'])}):")
            for error in results['errors'][:5]:  # Show first 5 errors
                print(f"• {error}")
            if len(results['errors']) > 5:
                print(f"... and {len(results['errors']) - 5} more errors")
        
        # Save results to file if specified
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output_file}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 