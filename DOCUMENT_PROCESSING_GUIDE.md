# Enhanced Document Processing System

A comprehensive document processing tool with strongly typed field definitions for signatures, radio buttons, checkboxes, and other form elements. This system extends the existing image labeling tool with advanced document processing capabilities, template management, and quality assurance features.

## 🚀 Key Features

### Strongly Typed Field Definitions
- **Text Fields**: Pattern validation, length constraints, case sensitivity
- **Number Fields**: Range validation, decimal precision control
- **Date Fields**: Format validation, date range constraints
- **Email Fields**: Email format validation
- **Phone Fields**: Phone number format validation with country codes
- **Signature Fields**: Ink detection, coverage analysis, quality validation
- **Checkbox Fields**: State detection and validation
- **Radio Button Fields**: Option validation, multiple selection control
- **Dropdown Fields**: Option validation, custom value support
- **Table Fields**: Structure validation, row/column constraints

### Template Management
- JSON-based template storage and loading
- Template versioning and metadata
- Field definition inheritance
- Template validation and error detection
- Automatic template matching (future enhancement)

### Batch Processing
- Directory-based document processing
- Multiple output formats (JSON, CSV)
- Progress tracking and statistics
- Error handling and reporting
- Quality assurance metrics

### Validation & Quality Assurance
- Field-level validation with custom rules
- Confidence threshold management
- Batch quality analysis
- Error detection and reporting
- Problematic field identification

## 📋 Installation

### Prerequisites
```bash
# Core dependencies
pip install streamlit opencv-python pillow numpy pandas
pip install ultralytics transformers  # For AI models
pip install tqdm  # For progress bars

# Optional dependencies
pip install google-generativeai  # For Gemini AI
```

### Setup
1. Ensure your existing image labeling tool is working
2. Add the document processing modules to your project
3. Run the demo to verify installation:
```bash
python document_processing_demo.py
```

## 🔧 Quick Start

### 1. Create a Template

```python
from document_processing.template_manager import DocumentTemplateManager
from document_processing.field_types import (
    BoundingBox, TextField, EmailField, SignatureField, CheckboxField
)

# Initialize template manager
template_manager = DocumentTemplateManager("templates")

# Create a new template
template = template_manager.create_template(
    name="employment_form",
    description="Standard employment application",
    version="1.0.0"
)

# Add fields
name_field = TextField(
    name="full_name",
    bounding_box=BoundingBox(100, 50, 300, 30),
    required=True,
    min_length=2,
    max_length=100
)

email_field = EmailField(
    name="email",
    bounding_box=BoundingBox(100, 100, 300, 30),
    required=True
)

signature_field = SignatureField(
    name="signature",
    bounding_box=BoundingBox(100, 400, 300, 60),
    required=True,
    min_signature_area=0.05
)

# Add fields to template
template_manager.add_field_to_template(template, name_field)
template_manager.add_field_to_template(template, email_field)
template_manager.add_field_to_template(template, signature_field)

# Save template
template_manager.save_template(template)
```

### 2. Process a Single Document

```python
from document_processing.document_processor import DocumentProcessor

# Initialize processor
processor = DocumentProcessor(ai_model_type="yolo")

# Process document
results = processor.process_document(
    document_path="application.jpg",
    template=template,
    confidence_threshold=0.7
)

# Check results
for field_name, extraction in results.items():
    print(f"{field_name}: {extraction.value} (confidence: {extraction.confidence:.2f})")
    if not extraction.is_valid:
        print(f"  Errors: {extraction.validation_errors}")
```

### 3. Batch Processing

```python
from document_processing.batch_processor import BatchDocumentProcessor, BatchProcessingConfig

# Initialize batch processor
batch_processor = BatchDocumentProcessor(template_manager)

# Configure processing
config = BatchProcessingConfig(
    confidence_threshold=0.7,
    output_format="all",  # JSON and CSV
    include_statistics=True
)

# Process directory
results = batch_processor.process_directory(
    documents_dir="documents/",
    template_name="employment_form",
    output_dir="output/",
    config=config
)

print(f"Processed {results['batch_statistics']['summary']['total_documents']} documents")
```

## 📊 Field Types Reference

### TextField
```python
TextField(
    name="field_name",
    bounding_box=BoundingBox(x, y, width, height),
    required=False,
    min_length=None,
    max_length=None,
    pattern=None,  # Regex pattern
    case_sensitive=False,
    validation_level=ValidationLevel.MODERATE
)
```

### NumberField
```python
NumberField(
    name="amount",
    bounding_box=BoundingBox(x, y, width, height),
    min_value=0,
    max_value=1000000,
    decimal_places=2  # None for integers
)
```

### DateField
```python
DateField(
    name="birth_date",
    bounding_box=BoundingBox(x, y, width, height),
    date_format="%m/%d/%Y",
    min_date=datetime(1900, 1, 1),
    max_date=datetime.now()
)
```

### SignatureField
```python
SignatureField(
    name="signature",
    bounding_box=BoundingBox(x, y, width, height),
    required=True,
    min_signature_area=0.05,  # 5% of bounding box
    require_ink_detection=True
)
```

### CheckboxField
```python
CheckboxField(
    name="agree_terms",
    bounding_box=BoundingBox(x, y, width, height),
    default_value=False
)
```

### RadioButtonField
```python
RadioButtonField(
    name="preferred_contact",
    bounding_box=BoundingBox(x, y, width, height),
    options=["Email", "Phone", "Mail"],
    allow_multiple=False
)
```

## 🎯 Advanced Usage

### Custom Validation Rules

```python
class CustomTextField(TextField):
    def validate(self, extraction):
        extraction = super().validate(extraction)
        
        # Custom business rule
        if extraction.value and "prohibited_word" in extraction.value.lower():
            extraction.is_valid = False
            extraction.validation_errors.append("Contains prohibited content")
        
        return extraction
```

### Template Inheritance

```python
# Base template
base_template = template_manager.create_template("base_form", "Base form template")

# Extended template
extended_template = template_manager.create_template("extended_form", "Extended form")

# Copy fields from base template
for field in base_template.fields:
    template_manager.add_field_to_template(extended_template, field)

# Add additional fields
additional_field = TextField(name="additional_info", bounding_box=BoundingBox(0, 500, 300, 30))
template_manager.add_field_to_template(extended_template, additional_field)
```

### Quality Assurance

```python
from document_processing.validators import QualityAssurance, DocumentValidator

# Analyze batch quality
quality_report = QualityAssurance.analyze_batch_quality(batch_results)

# Identify problematic fields
problematic = QualityAssurance.identify_problematic_fields(batch_results)

# Custom validation
validator = DocumentValidator(min_confidence_threshold=0.8)
validation_report = validator.validate_document_results(field_results, template)
```

## 📱 CLI Usage

### Process Documents
```bash
# Process with specific template
python batch_document_processor.py process documents/ employment_form output/ --confidence 0.7

# Auto-process with template matching
python batch_document_processor.py auto-process documents/ output/ --ai-model yolo

# Use different AI models
python batch_document_processor.py process docs/ form output/ --ai-model transformers
python batch_document_processor.py process docs/ form output/ --ai-model gemini --api-key YOUR_KEY
```

### Template Management
```bash
# List templates
python batch_document_processor.py list-templates --detailed

# Create template interactively
python batch_document_processor.py create-template

# Validate results
python batch_document_processor.py validate output/batch_results_20231201.json
```

### Quality Analysis
```bash
# Analyze processing quality
python batch_document_processor.py quality-analysis output/batch_results.json --output qa_report.json
```

## 🔍 Output Formats

### JSON Format
```json
{
  "batch_statistics": {
    "summary": {
      "total_documents": 100,
      "successful_documents": 95,
      "success_rate": 0.95,
      "processing_time": 120.5
    }
  },
  "document_results": [
    {
      "document_path": "doc1.jpg",
      "template_name": "employment_form",
      "field_results": {
        "full_name": {
          "value": "John Doe",
          "confidence": 0.95,
          "is_valid": true,
          "extraction_method": "ai"
        }
      }
    }
  ]
}
```

### CSV Format
```csv
document_path,template_name,full_name_value,full_name_confidence,full_name_valid,email_value,email_confidence,email_valid
doc1.jpg,employment_form,John Doe,0.95,true,john@example.com,0.88,true
doc2.jpg,employment_form,Jane Smith,0.92,true,jane@example.com,0.91,true
```

## 🛠️ Integration Examples

### Streamlit Integration
```python
import streamlit as st
from document_processing.template_manager import DocumentTemplateManager

st.title("Document Processing Interface")

# Template selection
template_manager = DocumentTemplateManager()
templates = template_manager.list_templates()
selected_template = st.selectbox("Select Template", templates)

# File upload
uploaded_file = st.file_uploader("Upload Document", type=['jpg', 'png', 'pdf'])

if uploaded_file and selected_template:
    # Process document
    template = template_manager.get_template(selected_template)
    processor = DocumentProcessor()
    
    # Save uploaded file temporarily
    with open("temp_doc.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Process
    results = processor.process_document("temp_doc.jpg", template)
    
    # Display results
    st.subheader("Extraction Results")
    for field_name, extraction in results.items():
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write(f"**{field_name}**: {extraction.value}")
        with col2:
            st.write(f"Confidence: {extraction.confidence:.2f}")
        with col3:
            status = "✅" if extraction.is_valid else "❌"
            st.write(status)
```

### Flask API Integration
```python
from flask import Flask, request, jsonify
from document_processing.document_processor import DocumentProcessor

app = Flask(__name__)
processor = DocumentProcessor()
template_manager = DocumentTemplateManager()

@app.route('/process', methods=['POST'])
def process_document():
    file = request.files['document']
    template_name = request.form['template']
    
    # Save file
    file.save('temp_doc.jpg')
    
    # Get template
    template = template_manager.get_template(template_name)
    
    # Process
    results = processor.process_document('temp_doc.jpg', template)
    
    # Convert results to JSON-serializable format
    json_results = {}
    for field_name, extraction in results.items():
        json_results[field_name] = {
            'value': extraction.value,
            'confidence': extraction.confidence,
            'is_valid': extraction.is_valid,
            'errors': extraction.validation_errors
        }
    
    return jsonify(json_results)
```

## 🔧 Configuration

### Processing Configuration
```python
config = BatchProcessingConfig(
    confidence_threshold=0.7,
    output_format="all",
    include_raw_data=True,
    include_statistics=True,
    parallel_processing=False,  # Future feature
    max_workers=4,
    validation_level="strict"
)
```

### AI Model Configuration
```python
# YOLO (fast, local)
processor = DocumentProcessor(ai_model_type="yolo")

# Transformers (accurate, local)
processor = DocumentProcessor(ai_model_type="transformers")

# Gemini (cloud-based, requires API key)
processor = DocumentProcessor(ai_model_type="gemini", api_key="your-api-key")
```

## 🚨 Error Handling

### Common Issues and Solutions

**Template Not Found**
```python
try:
    template = template_manager.get_template("nonexistent")
except FileNotFoundError:
    print("Template not found. Available templates:", template_manager.list_templates())
```

**Processing Errors**
```python
try:
    results = processor.process_document(doc_path, template)
except Exception as e:
    print(f"Processing failed: {e}")
    # Check document format, template validity, AI model availability
```

**Validation Failures**
```python
for field_name, extraction in results.items():
    if not extraction.is_valid:
        print(f"Field {field_name} validation failed:")
        for error in extraction.validation_errors:
            print(f"  - {error}")
```

## 📈 Performance Optimization

### For Large Batches
- Use appropriate confidence thresholds to balance speed vs accuracy
- Process during off-peak hours for cloud AI services
- Monitor memory usage for very large documents
- Use batch size limits for memory management

### For High Accuracy
- Use multiple AI models for cross-validation
- Implement manual review workflows for low-confidence extractions
- Fine-tune confidence thresholds based on field importance
- Use field-specific validation rules

## 🤝 Contributing

### Adding New Field Types
1. Extend `BaseField` class in `field_types.py`
2. Implement custom validation logic
3. Add serialization/deserialization support
4. Update template manager to handle new field type
5. Add tests and documentation

### Extending AI Capabilities
1. Implement new AI detector in `ai_utils.py`
2. Add integration to `DocumentProcessor`
3. Update CLI with new model option
4. Add configuration options

## 📚 Additional Resources

- [Original Image Labeling Tool Documentation](README.md)
- [AI Processing Guide](README_AI.md)
- [Batch Processing Examples](examples/)
- [Template Examples](templates/)

## 🐛 Troubleshooting

### Common Error Messages

**"AI not available"**
- Check AI model installation
- Verify API keys for cloud services
- Ensure required dependencies are installed

**"Invalid template format"**
- Validate template JSON structure
- Check field definitions for required properties
- Verify bounding box coordinates

**"Low confidence warnings"**
- Adjust confidence thresholds
- Improve document image quality
- Review field bounding box accuracy
- Consider alternative AI models

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed processing logs
processor = DocumentProcessor(ai_model_type="yolo")
results = processor.process_document(doc_path, template, confidence_threshold=0.5)
```

This enhanced document processing system provides a robust foundation for extracting structured data from forms and documents with strong typing, validation, and quality assurance capabilities.