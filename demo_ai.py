#!/usr/bin/env python3
"""
Demo script showcasing AI-enhanced image labeling capabilities
"""

import os
import json
import tempfile
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ai_utils import create_ai_detector, create_ai_assistant
from ai_image_manager import AIImageManager

def create_demo_image(width=800, height=600):
    """Create a simple demo image with geometric shapes"""
    # Create a white background
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw some geometric shapes
    # Rectangle (simulating a "box")
    draw.rectangle([100, 100, 300, 250], outline='red', width=3)
    draw.text((110, 110), "Box", fill='red')
    
    # Circle (simulating a "ball")
    draw.ellipse([400, 150, 550, 300], outline='blue', width=3)
    draw.text((450, 200), "Ball", fill='blue')
    
    # Triangle (simulating a "sign")
    points = [(600, 100), (700, 200), (500, 200)]
    draw.polygon(points, outline='green', width=3)
    draw.text((580, 150), "Sign", fill='green')
    
    return img

def demo_ai_detection():
    """Demonstrate AI object detection capabilities"""
    print("🤖 AI-Enhanced Image Labeling Tool Demo")
    print("=" * 50)
    
    # Create demo image
    print("Creating demo image...")
    demo_img = create_demo_image()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        demo_img.save(tmp_file.name)
        img_path = tmp_file.name
    
    try:
        # Initialize AI detector (using YOLO for demo)
        print("Initializing AI model (YOLO)...")
        detector = create_ai_detector("yolo")
        assistant = create_ai_assistant(detector)
        
        # Create image manager
        im = AIImageManager(img_path, "json")
        
        # Get AI suggestions
        print("Getting AI suggestions...")
        suggestions = assistant.suggest_annotations(img_path)
        
        print(f"AI detected {len(suggestions.get('detections', []))} objects")
        
        # Display suggestions
        detections = suggestions.get('detections', [])
        if detections:
            print("\nDetected objects:")
            for i, det in enumerate(detections):
                print(f"  {i+1}. {det['label']} (confidence: {det['confidence']:.2f})")
        
        # Apply AI suggestions
        print("\nApplying AI suggestions...")
        im.set_ai_annotations(detections)
        
        # Get statistics
        stats = im.get_annotation_statistics()
        print(f"\nAnnotation statistics:")
        print(f"  Total annotations: {stats['total_annotations']}")
        print(f"  AI annotations: {stats['ai_annotations']}")
        print(f"  Manual annotations: {stats['manual_annotations']}")
        
        # Validate AI annotations
        validation = im.validate_ai_annotations()
        print(f"\nAI annotation quality:")
        print(f"  High confidence: {validation['high_confidence']}")
        print(f"  Medium confidence: {validation['medium_confidence']}")
        print(f"  Low confidence: {validation['low_confidence']}")
        
        # Save annotations
        output_path = img_path.replace('.png', '_annotated.json')
        im.save_annotation()
        print(f"\nAnnotations saved to: {output_path}")
        
        # Show suggested labels
        suggested_labels = suggestions.get('suggested_labels', [])
        if suggested_labels:
            print(f"\nSuggested labels: {', '.join(suggested_labels)}")
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("This might be due to missing AI model dependencies.")
        print("Try installing: pip install ultralytics torch torchvision")
    
    finally:
        # Clean up
        if os.path.exists(img_path):
            os.unlink(img_path)

def demo_batch_processing():
    """Demonstrate batch processing capabilities"""
    print("\n🔄 Batch Processing Demo")
    print("=" * 30)
    
    # Create multiple demo images
    print("Creating demo images for batch processing...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        # Create 3 demo images
        for i in range(3):
            img = create_demo_image(600, 400)
            img_path = os.path.join(temp_dir, f"demo_{i+1}.png")
            img.save(img_path)
        
        print(f"Created {len(os.listdir(temp_dir))} demo images")
        
        # Simulate batch processing
        print("Simulating batch AI processing...")
        
        # Initialize AI
        detector = create_ai_detector("yolo")
        assistant = create_ai_assistant(detector)
        
        total_annotations = 0
        for img_file in os.listdir(temp_dir):
            if img_file.endswith('.png'):
                img_path = os.path.join(temp_dir, img_file)
                
                # Process with AI
                im = AIImageManager(img_path, "json")
                suggestions = assistant.suggest_annotations(img_path)
                detections = suggestions.get('detections', [])
                
                if detections:
                    im.set_ai_annotations(detections)
                    im.save_annotation()
                    total_annotations += len(detections)
        
        print(f"Batch processing completed!")
        print(f"Total annotations created: {total_annotations}")
        
    except Exception as e:
        print(f"Batch processing demo failed: {e}")
    
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)

def demo_validation():
    """Demonstrate validation capabilities"""
    print("\n🔍 Validation Demo")
    print("=" * 20)
    
    # Create demo image with annotations
    demo_img = create_demo_image()
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        demo_img.save(tmp_file.name)
        img_path = tmp_file.name
    
    try:
        # Create some manual annotations
        im = AIImageManager(img_path, "json")
        
        # Add manual annotations
        manual_annotations = [
            {'left': 100, 'top': 100, 'width': 200, 'height': 150, 'label': 'box'},
            {'left': 400, 'top': 150, 'width': 150, 'height': 150, 'label': 'ball'}
        ]
        
        # Simulate AI annotations
        ai_annotations = [
            {'left': 100, 'top': 100, 'width': 200, 'height': 150, 'label': 'box', 'confidence': 0.9},
            {'left': 400, 'top': 150, 'width': 150, 'height': 150, 'label': 'ball', 'confidence': 0.8},
            {'left': 600, 'top': 100, 'width': 200, 'height': 100, 'label': 'sign', 'confidence': 0.7}
        ]
        
        # Apply annotations
        im.set_ai_annotations(ai_annotations)
        
        # Validate
        validation = im.validate_ai_annotations()
        stats = im.get_annotation_statistics()
        
        print(f"Validation results:")
        print(f"  Total annotations: {stats['total_annotations']}")
        print(f"  AI annotations: {stats['ai_annotations']}")
        print(f"  Manual annotations: {stats['manual_annotations']}")
        print(f"  High confidence AI: {validation['high_confidence']}")
        print(f"  Potential issues: {len(validation['potential_issues'])}")
        
        if validation['potential_issues']:
            print("  Issues found:")
            for issue in validation['potential_issues']:
                print(f"    • {issue}")
        
    except Exception as e:
        print(f"Validation demo failed: {e}")
    
    finally:
        if os.path.exists(img_path):
            os.unlink(img_path)

def main():
    """Run all demos"""
    print("🎯 AI-Enhanced Image Labeling Tool - Demo Suite")
    print("=" * 60)
    
    # Run demos
    demo_ai_detection()
    demo_batch_processing()
    demo_validation()
    
    print("\n" + "=" * 60)
    print("🎉 All demos completed!")
    print("\nTo run the full application:")
    print("  streamlit run app_ai.py")
    print("\nFor batch processing:")
    print("  python batch_ai_processing.py img_dir --model yolo")

if __name__ == "__main__":
    main() 