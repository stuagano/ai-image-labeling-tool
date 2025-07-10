"""
Document Processor

Core document processing engine that extracts form fields from documents
using AI models and template-based field definitions.
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image

from .field_types import (
    FormField, FieldExtraction, FieldType, BoundingBox,
    SignatureField, CheckboxField, RadioButtonField, TableField
)
from .template_manager import DocumentTemplate
from ..ai_utils import create_ai_detector, create_ai_assistant


class DocumentProcessor:
    """Process documents to extract form field data using AI and templates."""
    
    def __init__(self, ai_model_type: str = "yolo", api_key: Optional[str] = None):
        """Initialize document processor.
        
        Args:
            ai_model_type: Type of AI model to use
            api_key: API key for cloud-based AI services
        """
        self.ai_model_type = ai_model_type
        self.api_key = api_key
        self.ai_detector = None
        self.ai_assistant = None
        self._initialize_ai()
    
    def _initialize_ai(self):
        """Initialize AI models for document processing."""
        try:
            self.ai_detector = create_ai_detector(self.ai_model_type, self.api_key)
            self.ai_assistant = create_ai_assistant(self.ai_detector)
        except Exception as e:
            print(f"Warning: Could not initialize AI models: {e}")
    
    def process_document(
        self,
        document_path: str,
        template: DocumentTemplate,
        confidence_threshold: float = 0.5
    ) -> Dict[str, FieldExtraction]:
        """Process a document using a template to extract field data.
        
        Args:
            document_path: Path to the document image
            template: Document template with field definitions
            confidence_threshold: Minimum confidence for AI extractions
            
        Returns:
            Dictionary mapping field names to extraction results
        """
        # Load document image
        document_image = Image.open(document_path)
        
        # Process each field in the template
        field_results = {}
        
        for field in template.fields:
            try:
                extraction = self._extract_field(
                    document_image, field, document_path, confidence_threshold
                )
                
                # Validate extraction
                extraction = field.validate(extraction)
                field_results[field.name] = extraction
                
            except Exception as e:
                # Create error extraction
                field_results[field.name] = FieldExtraction(
                    value=None,
                    confidence=0.0,
                    is_valid=False,
                    validation_errors=[f"Extraction failed: {e}"],
                    extraction_method="error"
                )
        
        return field_results
    
    def _extract_field(
        self,
        document_image: Image.Image,
        field: FormField,
        document_path: str,
        confidence_threshold: float
    ) -> FieldExtraction:
        """Extract a single field from the document.
        
        Args:
            document_image: PIL Image of the document
            field: Field definition
            document_path: Path to document (for AI processing)
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            FieldExtraction result
        """
        # Extract field region from document
        field_image = self._extract_field_region(document_image, field.bounding_box)
        
        # Process based on field type
        if field.field_type == FieldType.SIGNATURE:
            return self._extract_signature(field_image, field)
        elif field.field_type == FieldType.CHECKBOX:
            return self._extract_checkbox(field_image, field)
        elif field.field_type == FieldType.RADIO_BUTTON:
            return self._extract_radio_button(field_image, field)
        elif field.field_type == FieldType.TABLE:
            return self._extract_table(field_image, field)
        else:
            # Use AI for text-based fields
            return self._extract_text_with_ai(
                document_path, field, confidence_threshold
            )
    
    def _extract_field_region(
        self, document_image: Image.Image, bounding_box: BoundingBox
    ) -> Image.Image:
        """Extract a specific region from the document image.
        
        Args:
            document_image: Full document image
            bounding_box: Region to extract
            
        Returns:
            Cropped field image
        """
        left = bounding_box.left
        top = bounding_box.top
        right = left + bounding_box.width
        bottom = top + bounding_box.height
        
        return document_image.crop((left, top, right, bottom))
    
    def _extract_signature(
        self, field_image: Image.Image, field: SignatureField
    ) -> FieldExtraction:
        """Extract signature field data.
        
        Args:
            field_image: Image of the signature field
            field: Signature field definition
            
        Returns:
            FieldExtraction with signature analysis
        """
        # Convert to grayscale for processing
        gray_image = cv2.cvtColor(np.array(field_image), cv2.COLOR_RGB2GRAY)
        
        # Calculate signature coverage
        total_pixels = gray_image.size
        
        # Detect ink (darker pixels)
        ink_threshold = 200  # Adjust based on document quality
        ink_pixels = np.sum(gray_image < ink_threshold)
        coverage = ink_pixels / total_pixels
        
        # Detect if there's actual ink vs just background
        ink_detected = coverage > 0.01  # At least 1% coverage
        
        signature_data = {
            'coverage': coverage,
            'ink_detected': ink_detected,
            'total_pixels': total_pixels,
            'ink_pixels': ink_pixels
        }
        
        return FieldExtraction(
            value=signature_data,
            confidence=0.9 if ink_detected else 0.1,
            extraction_method="computer_vision"
        )
    
    def _extract_checkbox(
        self, field_image: Image.Image, field: CheckboxField
    ) -> FieldExtraction:
        """Extract checkbox field data.
        
        Args:
            field_image: Image of the checkbox field
            field: Checkbox field definition
            
        Returns:
            FieldExtraction with checkbox state
        """
        # Convert to grayscale
        gray_image = cv2.cvtColor(np.array(field_image), cv2.COLOR_RGB2GRAY)
        
        # Apply threshold to get binary image
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Calculate fill percentage
        white_pixels = np.sum(binary_image == 255)
        total_pixels = binary_image.size
        fill_percentage = white_pixels / total_pixels
        
        # Determine if checkbox is checked (>20% filled)
        is_checked = fill_percentage > 0.2
        confidence = min(0.9, max(0.1, fill_percentage * 2))
        
        return FieldExtraction(
            value=is_checked,
            confidence=confidence,
            extraction_method="computer_vision"
        )
    
    def _extract_radio_button(
        self, field_image: Image.Image, field: RadioButtonField
    ) -> FieldExtraction:
        """Extract radio button field data.
        
        Args:
            field_image: Image of the radio button field
            field: Radio button field definition
            
        Returns:
            FieldExtraction with selected option
        """
        # For now, use similar logic to checkbox
        # In a real implementation, this would detect multiple radio buttons
        # and determine which ones are selected
        
        gray_image = cv2.cvtColor(np.array(field_image), cv2.COLOR_RGB2GRAY)
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
        
        white_pixels = np.sum(binary_image == 255)
        total_pixels = binary_image.size
        fill_percentage = white_pixels / total_pixels
        
        # Simple heuristic - if significantly filled, assume first option is selected
        if fill_percentage > 0.2 and field.options:
            selected_option = field.options[0]
            confidence = min(0.8, fill_percentage * 2)
        else:
            selected_option = None
            confidence = 0.2
        
        return FieldExtraction(
            value=selected_option,
            confidence=confidence,
            extraction_method="computer_vision"
        )
    
    def _extract_table(
        self, field_image: Image.Image, field: TableField
    ) -> FieldExtraction:
        """Extract table field data.
        
        Args:
            field_image: Image of the table field
            field: Table field definition
            
        Returns:
            FieldExtraction with table data
        """
        # This is a simplified implementation
        # A real implementation would use table detection algorithms
        
        # For now, return empty table structure
        table_data = []
        if field.columns:
            # Create one empty row with the expected columns
            empty_row = {col: "" for col in field.columns}
            table_data.append(empty_row)
        
        return FieldExtraction(
            value=table_data,
            confidence=0.3,  # Low confidence since this is a placeholder
            extraction_method="computer_vision",
            validation_errors=["Table extraction not fully implemented"]
        )
    
    def _extract_text_with_ai(
        self,
        document_path: str,
        field: FormField,
        confidence_threshold: float
    ) -> FieldExtraction:
        """Extract text-based field using AI.
        
        Args:
            document_path: Path to the document image
            field: Field definition
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            FieldExtraction with AI-extracted text
        """
        if not self.ai_assistant:
            return FieldExtraction(
                value="",
                confidence=0.0,
                is_valid=False,
                validation_errors=["AI not available"],
                extraction_method="error"
            )
        
        try:
            # Use AI to extract text from field region
            # This would need to be adapted based on your AI implementation
            ai_result = self._ai_extract_field_text(document_path, field)
            
            if ai_result and ai_result.get('confidence', 0) >= confidence_threshold:
                return FieldExtraction(
                    value=ai_result['text'],
                    confidence=ai_result['confidence'],
                    extraction_method="ai"
                )
            else:
                return FieldExtraction(
                    value="",
                    confidence=ai_result.get('confidence', 0) if ai_result else 0,
                    extraction_method="ai"
                )
                
        except Exception as e:
            return FieldExtraction(
                value="",
                confidence=0.0,
                is_valid=False,
                validation_errors=[f"AI extraction failed: {e}"],
                extraction_method="error"
            )
    
    def _ai_extract_field_text(
        self, document_path: str, field: FormField
    ) -> Optional[Dict[str, Any]]:
        """Use AI to extract text from a specific field.
        
        This is a placeholder that would need to be implemented
        based on your specific AI model capabilities.
        """
        # Placeholder implementation
        # In a real system, this would use OCR or multimodal AI
        # to extract text from the specific field region
        
        return {
            'text': '',
            'confidence': 0.5
        }
    
    def get_processing_statistics(
        self, field_results: Dict[str, FieldExtraction]
    ) -> Dict[str, Any]:
        """Get statistics about the processing results.
        
        Args:
            field_results: Results from process_document
            
        Returns:
            Statistics dictionary
        """
        total_fields = len(field_results)
        valid_fields = sum(1 for r in field_results.values() if r.is_valid)
        ai_extracted = sum(1 for r in field_results.values() if r.extraction_method == "ai")
        cv_extracted = sum(1 for r in field_results.values() if r.extraction_method == "computer_vision")
        
        avg_confidence = (
            sum(r.confidence for r in field_results.values()) / total_fields
            if total_fields > 0 else 0
        )
        
        return {
            'total_fields': total_fields,
            'valid_fields': valid_fields,
            'invalid_fields': total_fields - valid_fields,
            'validation_rate': valid_fields / total_fields if total_fields > 0 else 0,
            'ai_extracted_fields': ai_extracted,
            'cv_extracted_fields': cv_extracted,
            'average_confidence': avg_confidence,
            'extraction_methods': {
                'ai': ai_extracted,
                'computer_vision': cv_extracted,
                'error': sum(1 for r in field_results.values() if r.extraction_method == "error")
            }
        }