#!/usr/bin/env python3
"""
Document Processing Demo

Comprehensive demonstration of the enhanced document processing capabilities
including template creation, field definitions, batch processing, and validation.
"""

import os
import sys
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from document_processing.field_types import (
        BoundingBox, TextField, NumberField, DateField, EmailField, PhoneField,
        SignatureField, CheckboxField, RadioButtonField, DropdownField, TableField,
        FieldType, ValidationLevel, FieldExtraction
    )
    from document_processing.template_manager import DocumentTemplateManager
    from document_processing.validators import DocumentValidator, QualityAssurance
except ImportError as e:
    print(f"Import error: {e}")
    print("Note: This demo requires the document_processing module to be properly set up.")
    sys.exit(1)


def create_sample_templates():
    """Create sample document templates for demonstration."""
    print("Creating sample document templates...")
    
    # Initialize template manager
    template_manager = DocumentTemplateManager("demo_templates")
    
    # Template 1: Employment Application Form
    employment_template = template_manager.create_template(
        name="employment_application",
        description="Standard employment application form",
        version="1.0.0"
    )
    
    # Add fields to employment template
    fields = [
        TextField(
            name="first_name",
            bounding_box=BoundingBox(100, 50, 200, 30),
            required=True,
            label="First Name",
            min_length=2,
            max_length=50,
            validation_level=ValidationLevel.STRICT
        ),
        TextField(
            name="last_name",
            bounding_box=BoundingBox(350, 50, 200, 30),
            required=True,
            label="Last Name",
            min_length=2,
            max_length=50
        ),
        EmailField(
            name="email",
            bounding_box=BoundingBox(100, 100, 300, 30),
            required=True,
            label="Email Address"
        ),
        PhoneField(
            name="phone",
            bounding_box=BoundingBox(450, 100, 200, 30),
            required=True,
            label="Phone Number",
            format_pattern=r"^\(\d{3}\) \d{3}-\d{4}$"
        ),
        DateField(
            name="birth_date",
            bounding_box=BoundingBox(100, 150, 150, 30),
            required=True,
            label="Date of Birth",
            date_format="%m/%d/%Y",
            max_date=datetime.now() - timedelta(days=18*365)
        ),
        NumberField(
            name="years_experience",
            bounding_box=BoundingBox(300, 150, 100, 30),
            label="Years of Experience",
            min_value=0,
            max_value=50,
            decimal_places=0
        ),
        CheckboxField(
            name="eligible_to_work",
            bounding_box=BoundingBox(100, 200, 30, 30),
            required=True,
            label="Eligible to work in US"
        ),
        RadioButtonField(
            name="preferred_shift",
            bounding_box=BoundingBox(100, 280, 300, 80),
            label="Preferred Shift",
            options=["Day", "Evening", "Night", "Any"],
            required=True
        ),
        SignatureField(
            name="applicant_signature",
            bounding_box=BoundingBox(100, 400, 300, 60),
            required=True,
            label="Applicant Signature",
            min_signature_area=0.05
        )
    ]
    
    for field in fields:
        template_manager.add_field_to_template(employment_template, field)
    
    template_manager.save_template(employment_template)
    print(f"✓ Created employment application template with {len(fields)} fields")
    
    return template_manager


def demonstrate_field_types():
    """Demonstrate different field types and their validation."""
    print("\n" + "="*50)
    print("FIELD TYPES DEMONSTRATION")
    print("="*50)
    
    # Create sample field instances
    fields_demo = {
        "Text Field": TextField(
            name="sample_text",
            bounding_box=BoundingBox(0, 0, 100, 30),
            pattern=r"^[A-Za-z\s]+$",
            min_length=2,
            max_length=50
        ),
        "Number Field": NumberField(
            name="sample_number",
            bounding_box=BoundingBox(0, 0, 100, 30),
            min_value=0,
            max_value=100,
            decimal_places=2
        ),
        "Email Field": EmailField(
            name="sample_email",
            bounding_box=BoundingBox(0, 0, 200, 30)
        ),
        "Phone Field": PhoneField(
            name="sample_phone",
            bounding_box=BoundingBox(0, 0, 150, 30),
            format_pattern=r"^\(\d{3}\) \d{3}-\d{4}$"
        ),
        "Date Field": DateField(
            name="sample_date",
            bounding_box=BoundingBox(0, 0, 120, 30),
            date_format="%m/%d/%Y"
        ),
        "Checkbox Field": CheckboxField(
            name="sample_checkbox",
            bounding_box=BoundingBox(0, 0, 30, 30)
        ),
        "Radio Button Field": RadioButtonField(
            name="sample_radio",
            bounding_box=BoundingBox(0, 0, 200, 60),
            options=["Option A", "Option B", "Option C"]
        ),
        "Signature Field": SignatureField(
            name="sample_signature",
            bounding_box=BoundingBox(0, 0, 200, 60),
            min_signature_area=0.1
        )
    }
    
    # Test data for each field type
    test_data = {
        "Text Field": [("John Doe", True), ("J", False), ("123", False)],
        "Number Field": [("25.50", True), ("150", False), ("abc", False)],
        "Email Field": [("test@example.com", True), ("invalid-email", False)],
        "Phone Field": [("(555) 123-4567", True), ("555-1234", False)],
        "Date Field": [("12/25/2023", True), ("invalid-date", False)],
        "Checkbox Field": [(True, True), ("checked", True), ("invalid", False)],
        "Radio Button Field": [("Option A", True), ("Invalid Option", False)],
        "Signature Field": [({"coverage": 0.15, "ink_detected": True}, True), 
                           ({"coverage": 0.05, "ink_detected": False}, False)]
    }
    
    for field_name, field_obj in fields_demo.items():
        print(f"\n🔧 {field_name}:")
        print(f"   Type: {field_obj.field_type.value}")
        print(f"   Required: {field_obj.required}")
        
        # Test validation with sample data
        if field_name in test_data:
            for test_value, expected_valid in test_data[field_name]:
                extraction = FieldExtraction(
                    value=test_value,
                    confidence=0.8,
                    extraction_method="test"
                )
                
                validated = field_obj.validate(extraction)
                status = "✅" if validated.is_valid == expected_valid else "❌"
                
                print(f"   {status} Test: {test_value} -> Valid: {validated.is_valid}")
                if validated.validation_errors:
                    print(f"      Errors: {', '.join(validated.validation_errors)}")


def demonstrate_template_management():
    """Demonstrate template management capabilities."""
    print("\n" + "="*50)
    print("TEMPLATE MANAGEMENT DEMONSTRATION")
    print("="*50)
    
    template_manager = DocumentTemplateManager("demo_templates")
    
    # Create a simple template
    form_template = template_manager.create_template(
        name="simple_form",
        description="A simple demonstration form",
        version="1.0.0"
    )
    
    # Add some fields
    sample_fields = [
        TextField(
            name="name",
            bounding_box=BoundingBox(100, 50, 200, 30),
            required=True,
            label="Full Name"
        ),
        EmailField(
            name="email",
            bounding_box=BoundingBox(100, 100, 250, 30),
            required=True,
            label="Email Address"
        ),
        CheckboxField(
            name="agree_terms",
            bounding_box=BoundingBox(100, 150, 30, 30),
            required=True,
            label="I agree to the terms"
        )
    ]
    
    for field in sample_fields:
        template_manager.add_field_to_template(form_template, field)
    
    # Save template
    template_manager.save_template(form_template)
    print(f"✓ Created and saved template: {form_template.name}")
    
    # Validate template
    validation_errors = form_template.validate_template()
    if validation_errors:
        print(f"❌ Template validation errors: {validation_errors}")
    else:
        print("✅ Template validation passed")
    
    # List all templates
    templates = template_manager.list_templates()
    print(f"📚 Available templates: {templates}")
    
    # Load template back
    loaded_template = template_manager.load_template("simple_form")
    print(f"📖 Loaded template: {loaded_template.name} with {len(loaded_template.fields)} fields")
    
    # Show template details
    print(f"\n📋 Template Details:")
    print(f"   Name: {loaded_template.name}")
    print(f"   Version: {loaded_template.version}")
    print(f"   Description: {loaded_template.description}")
    print(f"   Created: {loaded_template.created_date}")
    print(f"   Fields:")
    
    for field in loaded_template.fields:
        print(f"     • {field.name} ({field.field_type.value}) - Required: {field.required}")


def demonstrate_validation_system():
    """Demonstrate the validation and QA system."""
    print("\n" + "="*50)
    print("VALIDATION SYSTEM DEMONSTRATION")
    print("="*50)
    
    # Create validator
    validator = DocumentValidator(min_confidence_threshold=0.6)
    
    # Create sample template
    template_manager = DocumentTemplateManager()
    template = template_manager.create_template("validation_test", "Test form")
    
    # Add fields
    test_fields = [
        TextField(name="name", bounding_box=BoundingBox(0, 0, 100, 30), required=True),
        EmailField(name="email", bounding_box=BoundingBox(0, 40, 200, 30), required=True),
        NumberField(name="age", bounding_box=BoundingBox(0, 80, 80, 30), min_value=18, max_value=120),
        SignatureField(name="signature", bounding_box=BoundingBox(0, 120, 200, 60), required=True)
    ]
    
    for field in test_fields:
        template_manager.add_field_to_template(template, field)
    
    # Create test extraction results with various issues
    test_results = {
        "name": FieldExtraction(
            value="John Doe",
            confidence=0.95,
            is_valid=True,
            extraction_method="ai"
        ),
        "email": FieldExtraction(
            value="invalid-email-format",
            confidence=0.80,
            is_valid=False,
            validation_errors=["Invalid email format"],
            extraction_method="ai"
        ),
        "age": FieldExtraction(
            value="25",
            confidence=0.90,
            is_valid=True,
            extraction_method="ai"
        ),
        "signature": FieldExtraction(
            value={"coverage": 0.03, "ink_detected": False},
            confidence=0.40,
            is_valid=False,
            validation_errors=["Signature coverage too low", "No ink signature detected"],
            extraction_method="computer_vision"
        )
    }
    
    # Validate results
    validation_report = validator.validate_document_results(test_results, template)
    
    print("📊 VALIDATION REPORT:")
    report_dict = validation_report.to_dict()
    
    print(f"   Total fields: {report_dict['total_fields']}")
    print(f"   Valid fields: {report_dict['valid_fields']}")
    print(f"   Validation rate: {report_dict['validation_rate']:.1%}")
    print(f"   Overall score: {report_dict['overall_score']:.1f}/100")
    
    if report_dict['validation_errors']:
        print("\n❌ VALIDATION ERRORS:")
        for error in report_dict['validation_errors']:
            print(f"   • {error}")
    
    if report_dict['confidence_warnings']:
        print("\n⚠️  CONFIDENCE WARNINGS:")
        for warning in report_dict['confidence_warnings']:
            print(f"   • {warning}")
    
    # Demonstrate batch quality analysis
    print(f"\n📈 BATCH QUALITY ANALYSIS:")
    
    # Create mock batch results
    batch_results = []
    for i in range(3):
        doc_result = {
            'field_results': {
                'name': {'confidence': 0.9 - i*0.1, 'is_valid': True, 'extraction_method': 'ai'},
                'email': {'confidence': 0.8 - i*0.1, 'is_valid': i < 2, 'extraction_method': 'ai'},
                'age': {'confidence': 0.85 - i*0.05, 'is_valid': True, 'extraction_method': 'ai'},
                'signature': {'confidence': 0.7 - i*0.2, 'is_valid': i == 0, 'extraction_method': 'cv'}
            }
        }
        batch_results.append(doc_result)
    
    quality_analysis = QualityAssurance.analyze_batch_quality(batch_results)
    
    print(f"   Documents analyzed: {quality_analysis['summary']['total_documents']}")
    print(f"   Average confidence: {quality_analysis['confidence_analysis']['average_confidence']:.2f}")
    print(f"   Validation rate: {quality_analysis['summary']['validation_rate']:.1%}")
    
    print("\n💡 RECOMMENDATIONS:")
    for rec in quality_analysis['recommendations']:
        print(f"   • {rec}")


def main():
    """Main demonstration function."""
    print("🚀 ENHANCED DOCUMENT PROCESSING SYSTEM")
    print("="*60)
    print("This system provides strongly typed field definitions for")
    print("signatures, radio buttons, checkboxes, and other form fields")
    print("with comprehensive validation and batch processing capabilities.")
    print("="*60)
    
    try:
        # Demonstrate field types
        demonstrate_field_types()
        
        # Demonstrate template management
        demonstrate_template_management()
        
        # Create sample templates
        template_manager = create_sample_templates()
        print(f"\n📚 Created templates: {template_manager.list_templates()}")
        
        # Demonstrate validation system
        demonstrate_validation_system()
        
        print("\n" + "="*60)
        print("✅ DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("\nKey Features Demonstrated:")
        print("  🔧 Strongly typed field definitions")
        print("  📋 Template-based document processing")
        print("  ✍️  Signature detection and validation")
        print("  ☑️  Checkbox and radio button extraction")
        print("  📊 Comprehensive validation system")
        print("  📈 Quality assurance and batch analysis")
        print("  💾 Template management and persistence")
        print("  🔍 Error detection and reporting")
        print("\nNext Steps:")
        print("  • Create your own templates using the template manager")
        print("  • Process documents using the DocumentProcessor")
        print("  • Use BatchDocumentProcessor for bulk operations")
        print("  • Implement custom validation rules as needed")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()