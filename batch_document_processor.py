#!/usr/bin/env python3
"""
Batch Document Processor CLI

Command-line interface for batch processing documents using templates
with strong field typing and validation.
"""

import argparse
import os
import sys
import json
from pathlib import Path
from datetime import datetime

from document_processing.template_manager import DocumentTemplateManager
from document_processing.batch_processor import BatchDocumentProcessor, BatchProcessingConfig
from document_processing.validators import QualityAssurance


def setup_argparser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Batch process documents using templates with strong field typing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process documents with a specific template
  python batch_document_processor.py process docs/ employment_application output/ --confidence 0.7

  # Validate existing results
  python batch_document_processor.py validate output/batch_results_20231201_143022.json

  # List available templates
  python batch_document_processor.py list-templates

  # Create a new template interactively
  python batch_document_processor.py create-template
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process documents with template')
    process_parser.add_argument('documents_dir', help='Directory containing document images')
    process_parser.add_argument('template_name', help='Name of template to use')
    process_parser.add_argument('output_dir', help='Output directory for results')
    process_parser.add_argument('--confidence', type=float, default=0.5,
                               help='Confidence threshold (0.0-1.0)')
    process_parser.add_argument('--format', choices=['json', 'csv', 'all'], default='json',
                               help='Output format')
    process_parser.add_argument('--ai-model', choices=['yolo', 'transformers', 'gemini'], 
                               default='yolo', help='AI model to use')
    process_parser.add_argument('--api-key', help='API key for cloud AI services')
    process_parser.add_argument('--templates-dir', default='templates',
                               help='Directory containing templates')
    
    # Auto-process command
    auto_parser = subparsers.add_parser('auto-process', 
                                       help='Process documents with automatic template matching')
    auto_parser.add_argument('documents_dir', help='Directory containing document images')
    auto_parser.add_argument('output_dir', help='Output directory for results')
    auto_parser.add_argument('--confidence', type=float, default=0.5,
                            help='Confidence threshold (0.0-1.0)')
    auto_parser.add_argument('--format', choices=['json', 'csv', 'all'], default='json',
                            help='Output format')
    auto_parser.add_argument('--ai-model', choices=['yolo', 'transformers', 'gemini'], 
                            default='yolo', help='AI model to use')
    auto_parser.add_argument('--api-key', help='API key for cloud AI services')
    auto_parser.add_argument('--templates-dir', default='templates',
                            help='Directory containing templates')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate processing results')
    validate_parser.add_argument('results_file', help='Path to batch results JSON file')
    validate_parser.add_argument('--output', help='Save validation report to file')
    
    # List templates command
    list_parser = subparsers.add_parser('list-templates', help='List available templates')
    list_parser.add_argument('--templates-dir', default='templates',
                           help='Directory containing templates')
    list_parser.add_argument('--detailed', action='store_true',
                           help='Show detailed template information')
    
    # Create template command
    create_parser = subparsers.add_parser('create-template', 
                                        help='Create a new template interactively')
    create_parser.add_argument('--templates-dir', default='templates',
                              help='Directory to save templates')
    
    # Quality analysis command
    qa_parser = subparsers.add_parser('quality-analysis', 
                                     help='Perform quality analysis on results')
    qa_parser.add_argument('results_file', help='Path to batch results JSON file')
    qa_parser.add_argument('--output', help='Save analysis report to file')
    
    return parser


def process_documents(args):
    """Process documents using specified template."""
    print(f"🚀 Processing documents from {args.documents_dir}")
    print(f"📋 Using template: {args.template_name}")
    print(f"📁 Output directory: {args.output_dir}")
    
    # Initialize template manager
    template_manager = DocumentTemplateManager(args.templates_dir)
    
    # Check if template exists
    if args.template_name not in template_manager.list_templates():
        print(f"❌ Template '{args.template_name}' not found!")
        print(f"Available templates: {template_manager.list_templates()}")
        return False
    
    # Initialize batch processor
    batch_processor = BatchDocumentProcessor(
        template_manager, 
        ai_model_type=args.ai_model,
        api_key=args.api_key
    )
    
    # Configure processing
    config = BatchProcessingConfig(
        confidence_threshold=args.confidence,
        output_format=args.format,
        include_statistics=True
    )
    
    try:
        # Process documents
        results = batch_processor.process_directory(
            args.documents_dir,
            args.template_name,
            args.output_dir,
            config
        )
        
        # Display summary
        stats = results['batch_statistics']['summary']
        print(f"\n📊 PROCESSING SUMMARY:")
        print(f"   Total documents: {stats['total_documents']}")
        print(f"   Successful: {stats['successful_documents']}")
        print(f"   Failed: {stats['failed_documents']}")
        print(f"   Success rate: {stats['success_rate']:.1%}")
        print(f"   Processing time: {stats['total_processing_time']:.1f}s")
        print(f"   Average time per doc: {stats['average_processing_time']:.1f}s")
        
        print(f"\n✅ Processing completed! Results saved to {args.output_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return False


def auto_process_documents(args):
    """Process documents with automatic template matching."""
    print(f"🤖 Auto-processing documents from {args.documents_dir}")
    
    template_manager = DocumentTemplateManager(args.templates_dir)
    batch_processor = BatchDocumentProcessor(
        template_manager,
        ai_model_type=args.ai_model,
        api_key=args.api_key
    )
    
    config = BatchProcessingConfig(
        confidence_threshold=args.confidence,
        output_format=args.format,
        include_statistics=True
    )
    
    try:
        results = batch_processor.process_documents_with_auto_template(
            args.documents_dir,
            args.output_dir,
            config
        )
        
        print(f"\n📊 AUTO-PROCESSING SUMMARY:")
        for template_name, template_results in results.items():
            stats = template_results['batch_statistics']['summary']
            print(f"   Template '{template_name}': {stats['total_documents']} documents")
        
        print(f"\n✅ Auto-processing completed! Results saved to {args.output_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Auto-processing failed: {e}")
        return False


def validate_results(args):
    """Validate processing results."""
    print(f"🔍 Validating results from {args.results_file}")
    
    try:
        batch_processor = BatchDocumentProcessor(
            DocumentTemplateManager(),  # Dummy instance
            ai_model_type="yolo"
        )
        
        validation_report = batch_processor.validate_batch_results(args.results_file)
        
        print(f"\n📊 VALIDATION REPORT:")
        print(f"   Total documents: {validation_report['total_documents']}")
        print(f"   Validation issues: {len(validation_report['validation_issues'])}")
        
        if validation_report['error_analysis']:
            print(f"   Documents with errors: {len(validation_report['error_analysis'])}")
        
        # Field analysis
        if validation_report['field_analysis']:
            print(f"\n📋 FIELD ANALYSIS:")
            for field_name, stats in validation_report['field_analysis'].items():
                success_rate = stats['valid_extractions'] / stats['total_extractions']
                print(f"   {field_name}: {success_rate:.1%} success rate, "
                      f"avg confidence: {stats['average_confidence']:.2f}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(validation_report, f, indent=2, default=str)
            print(f"\n💾 Validation report saved to {args.output}")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False


def list_templates(args):
    """List available templates."""
    print(f"📚 Templates in {args.templates_dir}:")
    
    try:
        template_manager = DocumentTemplateManager(args.templates_dir)
        templates = template_manager.list_templates()
        
        if not templates:
            print("   No templates found.")
            return True
        
        for template_name in templates:
            if args.detailed:
                template = template_manager.get_template(template_name)
                if template:
                    print(f"\n📋 {template_name}:")
                    print(f"   Version: {template.version}")
                    print(f"   Description: {template.description}")
                    print(f"   Fields: {len(template.fields)}")
                    print(f"   Created: {template.created_date}")
                    
                    # Show field types
                    field_types = {}
                    for field in template.fields:
                        field_type = field.field_type.value
                        field_types[field_type] = field_types.get(field_type, 0) + 1
                    
                    print(f"   Field types: {dict(field_types)}")
                else:
                    print(f"\n📋 {template_name}: [Error loading template]")
            else:
                print(f"   • {template_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to list templates: {e}")
        return False


def create_template_interactive(args):
    """Create a new template interactively."""
    print("🔧 Interactive Template Creator")
    print("=" * 40)
    
    try:
        template_manager = DocumentTemplateManager(args.templates_dir)
        
        # Get template info
        name = input("Template name: ").strip()
        if not name:
            print("❌ Template name is required!")
            return False
        
        description = input("Description: ").strip()
        version = input("Version (default: 1.0.0): ").strip() or "1.0.0"
        
        # Create template
        template = template_manager.create_template(name, description, version=version)
        
        print(f"\n✅ Created template '{name}'")
        print("📝 Note: Use the template manager API to add fields programmatically")
        print("   or edit the JSON file directly in the templates directory.")
        
        # Save empty template
        template_manager.save_template(template)
        print(f"💾 Template saved to {args.templates_dir}/{name}.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create template: {e}")
        return False


def quality_analysis(args):
    """Perform quality analysis on results."""
    print(f"📈 Performing quality analysis on {args.results_file}")
    
    try:
        with open(args.results_file, 'r') as f:
            results_data = json.load(f)
        
        batch_results = results_data.get('document_results', [])
        
        if not batch_results:
            print("❌ No document results found in file!")
            return False
        
        # Convert to format expected by QualityAssurance
        qa_data = []
        for doc_result in batch_results:
            qa_data.append({
                'field_results': doc_result.get('field_results', {})
            })
        
        analysis = QualityAssurance.analyze_batch_quality(qa_data)
        
        print(f"\n📊 QUALITY ANALYSIS REPORT:")
        
        # Summary
        summary = analysis['summary']
        print(f"   Documents: {summary['total_documents']}")
        print(f"   Total fields: {summary['total_fields']}")
        print(f"   Validation rate: {summary['validation_rate']:.1%}")
        
        # Confidence analysis
        confidence = analysis['confidence_analysis']
        print(f"\n🎯 CONFIDENCE ANALYSIS:")
        print(f"   Average confidence: {confidence['average_confidence']:.2f}")
        print(f"   High quality fields: {confidence['high_quality_fields']} "
              f"({confidence['quality_distribution']['high']:.1%})")
        print(f"   Medium quality fields: {confidence['medium_quality_fields']} "
              f"({confidence['quality_distribution']['medium']:.1%})")
        print(f"   Low quality fields: {confidence['low_quality_fields']} "
              f"({confidence['quality_distribution']['low']:.1%})")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in analysis['recommendations']:
            print(f"   • {rec}")
        
        # Identify problematic fields
        problematic = QualityAssurance.identify_problematic_fields(qa_data)
        if problematic:
            print(f"\n⚠️  PROBLEMATIC FIELDS:")
            for field_name, stats in problematic.items():
                print(f"   {field_name}:")
                print(f"     Failure rate: {stats['failure_rate']:.1%}")
                print(f"     Avg confidence: {stats['average_confidence']:.2f}")
                if stats['most_common_errors']:
                    print(f"     Common errors: {', '.join(stats['most_common_errors'][:3])}")
        
        if args.output:
            full_report = {
                'quality_analysis': analysis,
                'problematic_fields': problematic,
                'generated_at': datetime.now().isoformat()
            }
            
            with open(args.output, 'w') as f:
                json.dump(full_report, f, indent=2, default=str)
            print(f"\n💾 Quality analysis saved to {args.output}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quality analysis failed: {e}")
        return False


def main():
    """Main CLI function."""
    parser = setup_argparser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Ensure directories exist
    if hasattr(args, 'templates_dir'):
        os.makedirs(args.templates_dir, exist_ok=True)
    
    if hasattr(args, 'output_dir'):
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Route to appropriate function
    success = False
    
    if args.command == 'process':
        success = process_documents(args)
    elif args.command == 'auto-process':
        success = auto_process_documents(args)
    elif args.command == 'validate':
        success = validate_results(args)
    elif args.command == 'list-templates':
        success = list_templates(args)
    elif args.command == 'create-template':
        success = create_template_interactive(args)
    elif args.command == 'quality-analysis':
        success = quality_analysis(args)
    else:
        parser.print_help()
        return
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()