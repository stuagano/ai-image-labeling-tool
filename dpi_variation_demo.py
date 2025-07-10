#!/usr/bin/env python3
"""
DPI Variation Handling Demo

Demonstrates how the enhanced document processing system handles 
DPI variations, rotation, scaling, and other document inconsistencies.
"""

import os
import sys
import json
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def demonstrate_dpi_handling():
    """Demonstrate how the system handles DPI variations."""
    print("🔍 DPI VARIATION HANDLING DEMONSTRATION")
    print("="*60)
    
    # Simulate different document scanning scenarios
    scenarios = [
        {
            "name": "High DPI Scan (600 DPI)",
            "description": "Professional scanner, high quality",
            "dpi": 600,
            "rotation": 0.0,
            "scale_factor": 2.0,  # 2x scale relative to 300 DPI template
            "quality": "excellent"
        },
        {
            "name": "Standard Office Scan (300 DPI)",
            "description": "Standard office scanner",
            "dpi": 300,
            "rotation": 0.5,  # Slight rotation
            "scale_factor": 1.0,  # Template baseline
            "quality": "good"
        },
        {
            "name": "Mobile Photo (150 DPI equivalent)",
            "description": "Phone camera capture",
            "dpi": 150,
            "rotation": -2.3,  # Noticeable rotation
            "scale_factor": 0.5,  # Half scale
            "quality": "fair"
        },
        {
            "name": "Low Quality Fax (72 DPI)",
            "description": "Fax transmission quality",
            "dpi": 72,
            "rotation": 1.8,
            "scale_factor": 0.24,  # Very small scale
            "quality": "poor"
        }
    ]
    
    print("📊 SIMULATED DOCUMENT SCENARIOS:")
    print("-" * 40)
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        print(f"   DPI: {scenario['dpi']}")
        print(f"   Rotation: {scenario['rotation']}°")
        print(f"   Scale Factor: {scenario['scale_factor']}x")
        print(f"   Quality: {scenario['quality']}")
        
        # Simulate normalization results
        normalization_success = simulate_normalization(scenario)
        print(f"   Normalization: {'✅ Success' if normalization_success['success'] else '❌ Failed'}")
        
        if normalization_success['success']:
            print(f"   Confidence: {normalization_success['confidence']:.1%}")
            print(f"   Adjustments: {', '.join(normalization_success['adjustments'])}")


def simulate_normalization(scenario):
    """Simulate document normalization for a scenario."""
    
    # Calculate normalization success based on scenario quality
    quality_scores = {
        "excellent": 0.95,
        "good": 0.85,
        "fair": 0.70,
        "poor": 0.45
    }
    
    base_confidence = quality_scores.get(scenario['quality'], 0.5)
    
    # Factors that affect normalization success
    adjustments = []
    confidence_modifier = 0
    
    # DPI normalization
    if abs(scenario['scale_factor'] - 1.0) > 0.1:
        adjustments.append("DPI scaling")
        if scenario['dpi'] >= 150:
            confidence_modifier += 0.05
        else:
            confidence_modifier -= 0.1
    
    # Rotation correction
    if abs(scenario['rotation']) > 0.5:
        adjustments.append("rotation correction")
        if abs(scenario['rotation']) <= 3.0:
            confidence_modifier += 0.02
        else:
            confidence_modifier -= 0.05
    
    # Quality-based adjustments
    if scenario['quality'] in ['fair', 'poor']:
        adjustments.extend(["contrast enhancement", "noise reduction"])
        confidence_modifier += 0.03
    
    final_confidence = max(0.1, min(0.95, base_confidence + confidence_modifier))
    success = final_confidence > 0.6
    
    return {
        'success': success,
        'confidence': final_confidence,
        'adjustments': adjustments if adjustments else ['none required']
    }


def demonstrate_template_adaptation():
    """Demonstrate how templates adapt to document variations."""
    print("\n\n🎯 TEMPLATE ADAPTATION DEMONSTRATION")
    print("="*60)
    
    # Original template field coordinates (for 300 DPI)
    original_template = {
        "name": "employment_form",
        "target_dpi": 300,
        "fields": [
            {"name": "first_name", "left": 100, "top": 50, "width": 200, "height": 30},
            {"name": "last_name", "left": 350, "top": 50, "width": 200, "height": 30},
            {"name": "email", "left": 100, "top": 100, "width": 300, "height": 30},
            {"name": "signature", "left": 100, "top": 400, "width": 300, "height": 60}
        ]
    }
    
    print(f"📋 Original Template: {original_template['name']}")
    print(f"   Target DPI: {original_template['target_dpi']}")
    print(f"   Fields: {len(original_template['fields'])}")
    
    # Show how template adapts to different DPI scenarios
    test_scenarios = [
        {"dpi": 600, "description": "High resolution scan"},
        {"dpi": 150, "description": "Mobile photo"},
        {"dpi": 72, "description": "Low quality fax"}
    ]
    
    for scenario in test_scenarios:
        print(f"\n🔧 Adapting for {scenario['description']} ({scenario['dpi']} DPI):")
        
        scale_factor = scenario['dpi'] / original_template['target_dpi']
        
        print(f"   Scale factor: {scale_factor:.2f}x")
        print(f"   Adapted field coordinates:")
        
        for field in original_template['fields']:
            adapted_field = {
                "name": field['name'],
                "left": int(field['left'] * scale_factor),
                "top": int(field['top'] * scale_factor),
                "width": int(field['width'] * scale_factor),
                "height": int(field['height'] * scale_factor)
            }
            
            print(f"     {field['name']}: "
                  f"({adapted_field['left']}, {adapted_field['top']}) "
                  f"{adapted_field['width']}×{adapted_field['height']}")


def demonstrate_preprocessing_pipeline():
    """Demonstrate the document preprocessing pipeline."""
    print("\n\n⚙️ PREPROCESSING PIPELINE DEMONSTRATION")
    print("="*60)
    
    # Simulate different document quality issues
    document_issues = [
        {
            "name": "Rotated Document",
            "issues": ["rotation: -2.3°"],
            "preprocessing_steps": ["rotation_correction", "edge_enhancement"]
        },
        {
            "name": "Low Contrast Scan",
            "issues": ["low contrast", "faded text"],
            "preprocessing_steps": ["contrast_enhancement", "adaptive_histogram_equalization"]
        },
        {
            "name": "Noisy Fax",
            "issues": ["scan noise", "compression artifacts"],
            "preprocessing_steps": ["noise_reduction", "bilateral_filtering", "sharpening"]
        },
        {
            "name": "Uneven Lighting",
            "issues": ["shadows", "bright spots", "uneven illumination"],
            "preprocessing_steps": ["illumination_correction", "background_subtraction"]
        },
        {
            "name": "Skewed Document",
            "issues": ["skew: 1.8°", "perspective distortion"],
            "preprocessing_steps": ["skew_correction", "perspective_correction"]
        }
    ]
    
    for i, doc in enumerate(document_issues, 1):
        print(f"\n{i}. {doc['name']}")
        print(f"   Issues detected: {', '.join(doc['issues'])}")
        print(f"   Preprocessing steps:")
        
        for step in doc['preprocessing_steps']:
            # Simulate processing time and success
            processing_time = len(step) * 0.1  # Simulate based on complexity
            success_rate = 0.9 if len(doc['issues']) <= 2 else 0.8
            
            status = "✅" if success_rate > 0.85 else "⚠️"
            print(f"     {status} {step.replace('_', ' ').title()} ({processing_time:.1f}s)")
        
        # Simulate overall improvement
        improvement = min(95, 60 + len(doc['preprocessing_steps']) * 8)
        print(f"   📈 Quality improvement: {improvement}%")


def demonstrate_quality_assurance():
    """Demonstrate quality assurance for DPI variations."""
    print("\n\n📊 QUALITY ASSURANCE FOR DPI VARIATIONS")
    print("="*60)
    
    # Simulate batch processing results with different DPI scenarios
    batch_results = [
        {
            "document": "form_001_600dpi.jpg",
            "dpi": 600,
            "normalization_confidence": 0.95,
            "extraction_confidence": 0.92,
            "issues": []
        },
        {
            "document": "form_002_300dpi.jpg", 
            "dpi": 300,
            "normalization_confidence": 0.88,
            "extraction_confidence": 0.89,
            "issues": ["slight_rotation"]
        },
        {
            "document": "form_003_150dpi.jpg",
            "dpi": 150,
            "normalization_confidence": 0.72,
            "extraction_confidence": 0.68,
            "issues": ["low_resolution", "rotation"]
        },
        {
            "document": "form_004_72dpi.jpg",
            "dpi": 72,
            "normalization_confidence": 0.45,
            "extraction_confidence": 0.41,
            "issues": ["very_low_resolution", "poor_quality", "noise"]
        }
    ]
    
    print("📈 BATCH PROCESSING RESULTS:")
    print("-" * 50)
    
    total_docs = len(batch_results)
    high_quality = sum(1 for r in batch_results if r['extraction_confidence'] >= 0.8)
    medium_quality = sum(1 for r in batch_results if 0.6 <= r['extraction_confidence'] < 0.8)
    low_quality = sum(1 for r in batch_results if r['extraction_confidence'] < 0.6)
    
    for result in batch_results:
        status = "🟢" if result['extraction_confidence'] >= 0.8 else \
                "🟡" if result['extraction_confidence'] >= 0.6 else "🔴"
        
        print(f"{status} {result['document']}")
        print(f"   DPI: {result['dpi']}")
        print(f"   Normalization: {result['normalization_confidence']:.1%}")
        print(f"   Extraction: {result['extraction_confidence']:.1%}")
        if result['issues']:
            print(f"   Issues: {', '.join(result['issues'])}")
        print()
    
    print("📊 QUALITY DISTRIBUTION:")
    print(f"   🟢 High Quality (≥80%): {high_quality}/{total_docs} ({high_quality/total_docs:.1%})")
    print(f"   🟡 Medium Quality (60-80%): {medium_quality}/{total_docs} ({medium_quality/total_docs:.1%})")
    print(f"   🔴 Low Quality (<60%): {low_quality}/{total_docs} ({low_quality/total_docs:.1%})")
    
    # Quality recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if low_quality > 0:
        print(f"   • {low_quality} documents need manual review")
        print("   • Consider re-scanning low DPI documents at higher resolution")
    if medium_quality > 0:
        print(f"   • {medium_quality} documents have acceptable quality but could be improved")
        print("   • Enable enhanced preprocessing for better results")
    if high_quality == total_docs:
        print("   • All documents processed successfully!")
    
    avg_confidence = sum(r['extraction_confidence'] for r in batch_results) / total_docs
    print(f"   • Average confidence: {avg_confidence:.1%}")


def demonstrate_cli_usage():
    """Demonstrate CLI usage for handling DPI variations."""
    print("\n\n💻 CLI USAGE FOR DPI VARIATIONS")
    print("="*60)
    
    cli_examples = [
        {
            "title": "Process Mixed DPI Documents",
            "command": "python batch_document_processor.py process mixed_dpi_docs/ employment_form output/ --confidence 0.6",
            "description": "Process documents with varying DPI using adaptive templates"
        },
        {
            "title": "Enable Enhanced Preprocessing",
            "command": "python enhanced_batch_processor.py process docs/ form output/ --target-dpi 300 --enable-preprocessing",
            "description": "Use enhanced processor with normalization and preprocessing"
        },
        {
            "title": "Analyze Document Quality",
            "command": "python batch_document_processor.py quality-analysis results.json --dpi-analysis",
            "description": "Analyze quality issues related to DPI variations"
        },
        {
            "title": "Diagnostic Mode",
            "command": "python batch_document_processor.py diagnose problem_doc.jpg --template employment_form",
            "description": "Get detailed diagnostics about document normalization"
        }
    ]
    
    for i, example in enumerate(cli_examples, 1):
        print(f"\n{i}. {example['title']}")
        print(f"   Command: {example['command']}")
        print(f"   Description: {example['description']}")


def main():
    """Main demonstration function."""
    print("🚀 ENHANCED DOCUMENT PROCESSING: DPI VARIATION HANDLING")
    print("="*70)
    print("This demonstrates how the enhanced system handles documents with")
    print("different DPI settings, rotations, and scanning inconsistencies.")
    print("="*70)
    
    try:
        # Demonstrate core DPI handling
        demonstrate_dpi_handling()
        
        # Show template adaptation
        demonstrate_template_adaptation()
        
        # Show preprocessing pipeline
        demonstrate_preprocessing_pipeline()
        
        # Quality assurance
        demonstrate_quality_assurance()
        
        # CLI usage examples
        demonstrate_cli_usage()
        
        print("\n" + "="*70)
        print("✅ DPI VARIATION HANDLING DEMO COMPLETED!")
        print("\n🎯 KEY SOLUTIONS IMPLEMENTED:")
        print("  📏 Automatic DPI detection and normalization")
        print("  🔄 Document rotation and skew correction") 
        print("  📐 Adaptive template scaling")
        print("  🎨 Image quality enhancement and preprocessing")
        print("  📊 Quality assurance with confidence scoring")
        print("  🔧 Enhanced field extraction algorithms")
        print("  ⚙️ Configurable processing pipeline")
        
        print("\n💡 HANDLING DIFFERENT DOCUMENT TYPES:")
        print("  🖥️  High DPI scans (600+ DPI): Automatic downscaling")
        print("  📄 Standard scans (300 DPI): Direct processing")
        print("  📱 Mobile photos (150 DPI): Upscaling and enhancement")
        print("  📠 Fax quality (72 DPI): Aggressive preprocessing")
        print("  🔄 Rotated documents: Automatic rotation correction")
        print("  📐 Skewed documents: Perspective correction")
        print("  💡 Poor lighting: Illumination normalization")
        print("  🔍 Low contrast: Adaptive enhancement")
        
        print("\n🚀 NEXT STEPS:")
        print("  1. Install required dependencies: opencv-python, numpy")
        print("  2. Use EnhancedDocumentProcessor for variable DPI handling")
        print("  3. Enable preprocessing for low-quality documents")
        print("  4. Set appropriate confidence thresholds per document type")
        print("  5. Monitor quality metrics to optimize processing")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()