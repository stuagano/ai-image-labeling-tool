"""
Enhanced Document Processor

Integrates document normalization to handle DPI variations, rotation, 
scaling, and other scanning inconsistencies.
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Optional
from PIL import Image

from .document_processor import DocumentProcessor
from .document_normalizer import DocumentNormalizer, AdaptiveTemplateManager, DocumentPreprocessor
from .field_types import FieldExtraction, FormField
from .template_manager import DocumentTemplate


class EnhancedDocumentProcessor(DocumentProcessor):
    """Enhanced document processor with normalization capabilities."""
    
    def __init__(
        self, 
        ai_model_type: str = "yolo", 
        api_key: Optional[str] = None,
        target_dpi: float = 300.0,
        enable_preprocessing: bool = True
    ):
        """Initialize enhanced document processor.
        
        Args:
            ai_model_type: Type of AI model to use
            api_key: API key for cloud-based AI services
            target_dpi: Target DPI for document normalization
            enable_preprocessing: Whether to enable document preprocessing
        """
        super().__init__(ai_model_type, api_key)
        
        self.normalizer = DocumentNormalizer(target_dpi=target_dpi)
        self.preprocessor = DocumentPreprocessor() if enable_preprocessing else None
        self.adaptive_template_manager = None
        
    def set_template_manager(self, template_manager):
        """Set template manager for adaptive template functionality."""
        self.adaptive_template_manager = AdaptiveTemplateManager(
            template_manager, self.normalizer
        )
    
    def process_document(
        self,
        document_path: str,
        template: DocumentTemplate,
        confidence_threshold: float = 0.5,
        use_adaptive_template: bool = True
    ) -> Dict[str, FieldExtraction]:
        """Process document with normalization and adaptation.
        
        Args:
            document_path: Path to the document image
            template: Document template with field definitions
            confidence_threshold: Minimum confidence for AI extractions
            use_adaptive_template: Whether to adapt template to document variations
            
        Returns:
            Dictionary mapping field names to extraction results
        """
        try:
            # Step 1: Preprocess document if enabled
            if self.preprocessor:
                document_image, preprocessing_info = self.preprocessor.preprocess_document(document_path)
                print(f"📋 Preprocessing applied: {preprocessing_info['steps_applied']}")
            else:
                document_image = cv2.imread(document_path)
                preprocessing_info = {'steps_applied': []}
            
            if document_image is None:
                raise ValueError(f"Could not load document: {document_path}")
            
            # Step 2: Get adaptive template if requested
            if use_adaptive_template and self.adaptive_template_manager:
                adapted_template = self.adaptive_template_manager.get_adaptive_template(
                    template.name, document_image
                )
                if adapted_template:
                    template = adapted_template
                    print(f"✨ Using adaptive template for {template.name}")
            
            # Step 3: Process with potentially adapted template
            field_results = {}
            
            for field in template.fields:
                try:
                    extraction = self._extract_field_enhanced(
                        document_image, field, document_path, confidence_threshold, preprocessing_info
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
                        validation_errors=[f"Enhanced extraction failed: {e}"],
                        extraction_method="error"
                    )
            
            return field_results
            
        except Exception as e:
            print(f"❌ Enhanced document processing failed: {e}")
            # Fallback to basic processing
            return super().process_document(document_path, template, confidence_threshold)
    
    def _extract_field_enhanced(
        self,
        document_image: np.ndarray,
        field: FormField,
        document_path: str,
        confidence_threshold: float,
        preprocessing_info: Dict[str, Any]
    ) -> FieldExtraction:
        """Extract field with enhanced processing capabilities."""
        
        # Extract field region with better handling
        field_image = self._extract_field_region_enhanced(document_image, field.bounding_box)
        
        # Process based on field type with enhanced methods
        if field.field_type.value == "signature":
            return self._extract_signature_enhanced(field_image, field, preprocessing_info)
        elif field.field_type.value == "checkbox":
            return self._extract_checkbox_enhanced(field_image, field, preprocessing_info)
        elif field.field_type.value == "radio_button":
            return self._extract_radio_button_enhanced(field_image, field, preprocessing_info)
        elif field.field_type.value == "table":
            return self._extract_table_enhanced(field_image, field, preprocessing_info)
        else:
            # Use enhanced AI for text-based fields
            return self._extract_text_with_ai_enhanced(
                document_path, field, confidence_threshold, preprocessing_info
            )
    
    def _extract_field_region_enhanced(
        self, document_image: np.ndarray, bounding_box
    ) -> np.ndarray:
        """Extract field region with enhanced boundary handling."""
        # Ensure coordinates are within image bounds
        height, width = document_image.shape[:2]
        
        left = max(0, min(bounding_box.left, width - 1))
        top = max(0, min(bounding_box.top, height - 1))
        right = max(left + 1, min(left + bounding_box.width, width))
        bottom = max(top + 1, min(top + bounding_box.height, height))
        
        return document_image[top:bottom, left:right]
    
    def _extract_signature_enhanced(
        self, field_image: np.ndarray, field, preprocessing_info: Dict[str, Any]
    ) -> FieldExtraction:
        """Enhanced signature extraction with better analysis."""
        if field_image.size == 0:
            return FieldExtraction(
                value={"coverage": 0, "ink_detected": False},
                confidence=0.0,
                is_valid=False,
                validation_errors=["Empty field region"],
                extraction_method="computer_vision"
            )
        
        # Convert to grayscale for processing
        if len(field_image.shape) == 3:
            gray_image = cv2.cvtColor(field_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = field_image
        
        # Enhanced ink detection with adaptive thresholding
        # Use Otsu's method for better threshold selection
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Calculate signature metrics
        total_pixels = gray_image.size
        dark_pixels = np.sum(binary < 128)  # Count dark pixels
        coverage = dark_pixels / total_pixels
        
        # Enhanced ink detection using edge analysis
        edges = cv2.Canny(gray_image, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Combine coverage and edge density for better detection
        ink_detected = coverage > 0.005 and edge_density > 0.001  # More sensitive thresholds
        
        # Enhanced confidence calculation
        confidence = min(0.95, max(0.1, coverage * 20 + edge_density * 10))
        
        # Additional signature quality metrics
        signature_data = {
            'coverage': coverage,
            'ink_detected': ink_detected,
            'edge_density': edge_density,
            'total_pixels': total_pixels,
            'dark_pixels': dark_pixels,
            'preprocessing_applied': preprocessing_info.get('steps_applied', [])
        }
        
        return FieldExtraction(
            value=signature_data,
            confidence=confidence,
            extraction_method="enhanced_computer_vision"
        )
    
    def _extract_checkbox_enhanced(
        self, field_image: np.ndarray, field, preprocessing_info: Dict[str, Any]
    ) -> FieldExtraction:
        """Enhanced checkbox extraction with better state detection."""
        if field_image.size == 0:
            return FieldExtraction(
                value=False,
                confidence=0.0,
                is_valid=False,
                validation_errors=["Empty field region"],
                extraction_method="computer_vision"
            )
        
        # Convert to grayscale
        if len(field_image.shape) == 3:
            gray_image = cv2.cvtColor(field_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = field_image
        
        # Enhanced checkbox detection
        # Method 1: Adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Method 2: Look for checkbox patterns
        # Detect checkbox outline
        edges = cv2.Canny(gray_image, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find rectangular contours (checkbox outline)
        checkbox_outline = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4 and cv2.contourArea(contour) > 50:
                checkbox_outline = contour
                break
        
        # Calculate fill percentage
        total_pixels = gray_image.size
        dark_pixels = np.sum(adaptive_thresh == 0)
        fill_percentage = dark_pixels / total_pixels
        
        # Enhanced logic for checkbox state
        is_checked = False
        confidence = 0.5
        
        if checkbox_outline is not None:
            # Check content inside the checkbox
            mask = np.zeros(gray_image.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [checkbox_outline], 255)
            
            # Calculate fill within checkbox area
            checkbox_area = cv2.countNonZero(mask)
            if checkbox_area > 0:
                dark_in_checkbox = np.sum((adaptive_thresh == 0) & (mask == 255))
                checkbox_fill = dark_in_checkbox / checkbox_area
                
                # Checkbox is checked if significant fill inside the box
                is_checked = checkbox_fill > 0.15  # 15% threshold
                confidence = min(0.9, max(0.2, checkbox_fill * 3))
            else:
                # Fallback to overall fill percentage
                is_checked = fill_percentage > 0.2
                confidence = min(0.8, fill_percentage * 2)
        else:
            # No clear checkbox outline found, use fill percentage
            is_checked = fill_percentage > 0.25
            confidence = min(0.7, fill_percentage * 1.5)
        
        return FieldExtraction(
            value=is_checked,
            confidence=confidence,
            extraction_method="enhanced_computer_vision"
        )
    
    def _extract_radio_button_enhanced(
        self, field_image: np.ndarray, field, preprocessing_info: Dict[str, Any]
    ) -> FieldExtraction:
        """Enhanced radio button extraction with better option detection."""
        if field_image.size == 0:
            return FieldExtraction(
                value=None,
                confidence=0.0,
                is_valid=False,
                validation_errors=["Empty field region"],
                extraction_method="computer_vision"
            )
        
        # Convert to grayscale
        if len(field_image.shape) == 3:
            gray_image = cv2.cvtColor(field_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = field_image
        
        # Enhanced radio button detection
        # Look for circular patterns
        circles = cv2.HoughCircles(
            gray_image, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=50, param2=30, minRadius=5, maxRadius=50
        )
        
        selected_option = None
        confidence = 0.3
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            # Analyze each circle to see if it's filled
            for (x, y, r) in circles:
                # Extract circle region
                mask = np.zeros(gray_image.shape, dtype=np.uint8)
                cv2.circle(mask, (x, y), r, 255, -1)
                
                # Calculate fill within circle
                circle_region = gray_image[mask == 255]
                if len(circle_region) > 0:
                    avg_intensity = np.mean(circle_region)
                    
                    # Dark circle indicates selection
                    if avg_intensity < 128:  # Threshold for selection
                        # Map to available options
                        if hasattr(field, 'options') and field.options:
                            # Simple heuristic: map position to option
                            option_index = min(len(field.options) - 1, 
                                             int(len(circles) * y / gray_image.shape[0]))
                            selected_option = field.options[option_index]
                            confidence = min(0.8, (128 - avg_intensity) / 128)
                            break
        
        # Fallback: look for any filled areas if no circles found
        if selected_option is None and hasattr(field, 'options') and field.options:
            _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            fill_percentage = np.sum(binary == 0) / binary.size
            
            if fill_percentage > 0.1:
                selected_option = field.options[0]  # Default to first option
                confidence = min(0.6, fill_percentage * 2)
        
        return FieldExtraction(
            value=selected_option,
            confidence=confidence,
            extraction_method="enhanced_computer_vision"
        )
    
    def _extract_table_enhanced(
        self, field_image: np.ndarray, field, preprocessing_info: Dict[str, Any]
    ) -> FieldExtraction:
        """Enhanced table extraction with structure detection."""
        # This is still a placeholder for table extraction
        # A full implementation would use advanced table detection algorithms
        
        table_data = []
        if hasattr(field, 'columns') and field.columns:
            # Create placeholder structure
            empty_row = {col: "" for col in field.columns}
            table_data.append(empty_row)
        
        return FieldExtraction(
            value=table_data,
            confidence=0.4,  # Slightly higher confidence with enhanced processing
            extraction_method="enhanced_computer_vision",
            validation_errors=["Advanced table extraction not yet implemented"]
        )
    
    def _extract_text_with_ai_enhanced(
        self,
        document_path: str,
        field,
        confidence_threshold: float,
        preprocessing_info: Dict[str, Any]
    ) -> FieldExtraction:
        """Enhanced text extraction with preprocessing context."""
        # Add preprocessing information to the extraction metadata
        extraction = super()._extract_text_with_ai(document_path, field, confidence_threshold)
        
        # Boost confidence if preprocessing improved the image
        if 'normalization' in preprocessing_info.get('steps_applied', []):
            extraction.confidence = min(0.95, extraction.confidence * 1.1)
        
        if 'contrast_enhancement' in preprocessing_info.get('steps_applied', []):
            extraction.confidence = min(0.95, extraction.confidence * 1.05)
        
        return extraction
    
    def get_normalization_info(self, document_path: str) -> Dict[str, Any]:
        """Get document normalization information for diagnostics."""
        try:
            image = cv2.imread(document_path)
            if image is None:
                return {"error": "Could not load image"}
            
            # Get normalization result
            normalization_result = self.normalizer.normalize_document(image)
            
            return {
                "success": normalization_result.success,
                "confidence": normalization_result.confidence,
                "original_metrics": {
                    "width": normalization_result.original_metrics.width,
                    "height": normalization_result.original_metrics.height,
                    "estimated_dpi": normalization_result.original_metrics.dpi,
                    "rotation_angle": normalization_result.original_metrics.rotation_angle,
                    "skew_angle": normalization_result.original_metrics.skew_angle
                },
                "scale_factors": normalization_result.scale_factors,
                "preprocessing_available": self.preprocessor is not None
            }
            
        except Exception as e:
            return {"error": str(e)}