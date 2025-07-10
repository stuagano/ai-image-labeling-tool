"""
Document Normalizer

Handles document variations including DPI differences, rotation, scaling,
and alignment issues commonly found in scanned documents.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List, Any
from PIL import Image
import math
from dataclasses import dataclass

from .field_types import BoundingBox


@dataclass
class DocumentMetrics:
    """Metrics about a document's physical and scanning properties."""
    width: int
    height: int
    dpi: Tuple[float, float]  # (x_dpi, y_dpi)
    rotation_angle: float
    scale_factor: float
    skew_angle: float
    confidence: float


@dataclass
class NormalizationResult:
    """Result of document normalization process."""
    normalized_image: np.ndarray
    original_metrics: DocumentMetrics
    normalization_transform: np.ndarray
    scale_factors: Tuple[float, float]  # (x_scale, y_scale)
    success: bool
    confidence: float


class DocumentNormalizer:
    """Normalizes documents to handle DPI, rotation, and scaling variations."""
    
    def __init__(
        self,
        target_dpi: float = 300.0,
        max_rotation_correction: float = 45.0,
        min_confidence: float = 0.6
    ):
        """Initialize document normalizer.
        
        Args:
            target_dpi: Target DPI for normalization
            max_rotation_correction: Maximum rotation to correct (degrees)
            min_confidence: Minimum confidence for normalization
        """
        self.target_dpi = target_dpi
        self.max_rotation_correction = max_rotation_correction
        self.min_confidence = min_confidence
    
    def normalize_document(
        self, 
        image: np.ndarray, 
        template_dpi: Optional[float] = None
    ) -> NormalizationResult:
        """Normalize a document to standard format.
        
        Args:
            image: Input document image
            template_dpi: Expected DPI of template (if known)
            
        Returns:
            Normalization result with corrected image and metrics
        """
        if template_dpi is None:
            template_dpi = self.target_dpi
        
        # Detect document metrics
        metrics = self._detect_document_metrics(image)
        
        # Create normalization pipeline
        normalized_image = image.copy()
        transform_matrix = np.eye(3)
        
        # Step 1: Rotation correction
        if abs(metrics.rotation_angle) > 1.0:  # Only correct significant rotations
            normalized_image, rotation_transform = self._correct_rotation(
                normalized_image, metrics.rotation_angle
            )
            transform_matrix = rotation_transform @ transform_matrix
        
        # Step 2: Skew correction
        if abs(metrics.skew_angle) > 0.5:
            normalized_image, skew_transform = self._correct_skew(
                normalized_image, metrics.skew_angle
            )
            transform_matrix = skew_transform @ transform_matrix
        
        # Step 3: DPI/Scale normalization
        current_dpi = max(metrics.dpi)  # Use higher DPI as reference
        scale_factor = template_dpi / current_dpi
        
        if abs(scale_factor - 1.0) > 0.1:  # Only scale if significant difference
            normalized_image, scale_transform = self._scale_document(
                normalized_image, scale_factor
            )
            transform_matrix = scale_transform @ transform_matrix
        
        # Calculate final scale factors
        height_ratio = normalized_image.shape[0] / image.shape[0]
        width_ratio = normalized_image.shape[1] / image.shape[1]
        
        return NormalizationResult(
            normalized_image=normalized_image,
            original_metrics=metrics,
            normalization_transform=transform_matrix,
            scale_factors=(width_ratio, height_ratio),
            success=metrics.confidence > self.min_confidence,
            confidence=metrics.confidence
        )
    
    def transform_bounding_boxes(
        self, 
        bounding_boxes: List[BoundingBox], 
        normalization_result: NormalizationResult
    ) -> List[BoundingBox]:
        """Transform bounding boxes to match normalized document.
        
        Args:
            bounding_boxes: Original bounding boxes
            normalization_result: Result from normalize_document
            
        Returns:
            Transformed bounding boxes
        """
        if not normalization_result.success:
            return bounding_boxes
        
        transformed_boxes = []
        width_scale, height_scale = normalization_result.scale_factors
        
        for bbox in bounding_boxes:
            # Apply scaling
            new_left = int(bbox.left * width_scale)
            new_top = int(bbox.top * height_scale)
            new_width = int(bbox.width * width_scale)
            new_height = int(bbox.height * height_scale)
            
            transformed_boxes.append(BoundingBox(
                left=new_left,
                top=new_top,
                width=new_width,
                height=new_height
            ))
        
        return transformed_boxes
    
    def _detect_document_metrics(self, image: np.ndarray) -> DocumentMetrics:
        """Detect document metrics including DPI, rotation, and scale."""
        height, width = image.shape[:2]
        
        # Estimate DPI based on document content
        estimated_dpi = self._estimate_dpi(image)
        
        # Detect rotation angle
        rotation_angle = self._detect_rotation(image)
        
        # Detect skew
        skew_angle = self._detect_skew(image)
        
        # Calculate confidence based on edge detection quality
        confidence = self._calculate_detection_confidence(image)
        
        return DocumentMetrics(
            width=width,
            height=height,
            dpi=(estimated_dpi, estimated_dpi),
            rotation_angle=rotation_angle,
            scale_factor=1.0,
            skew_angle=skew_angle,
            confidence=confidence
        )
    
    def _estimate_dpi(self, image: np.ndarray) -> float:
        """Estimate document DPI based on content analysis."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Use text line detection to estimate DPI
        # Typical text height at 300 DPI is 10-20 pixels for 12pt font
        
        # Find horizontal lines (text baselines)
        edges = cv2.Canny(gray, 50, 150)
        
        # Horizontal line detection
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Find line spacing
        horizontal_projection = np.sum(horizontal_lines, axis=1)
        peaks = []
        for i in range(1, len(horizontal_projection) - 1):
            if (horizontal_projection[i] > horizontal_projection[i-1] and 
                horizontal_projection[i] > horizontal_projection[i+1] and
                horizontal_projection[i] > np.max(horizontal_projection) * 0.1):
                peaks.append(i)
        
        if len(peaks) > 1:
            # Calculate average line spacing
            spacings = np.diff(peaks)
            avg_spacing = np.median(spacings)
            
            # Typical line spacing at 300 DPI is 15-25 pixels
            # Scale accordingly
            if avg_spacing > 0:
                estimated_dpi = 300.0 * (20.0 / avg_spacing)
                # Clamp to reasonable range
                estimated_dpi = max(72, min(600, estimated_dpi))
                return estimated_dpi
        
        # Fallback: estimate based on image size
        # Typical 8.5x11 inch document
        height, width = gray.shape
        if width > height:  # Landscape
            estimated_dpi = width / 11.0
        else:  # Portrait
            estimated_dpi = width / 8.5
        
        return max(72, min(600, estimated_dpi))
    
    def _detect_rotation(self, image: np.ndarray) -> float:
        """Detect document rotation angle using line detection."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Hough line detection
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None:
            return 0.0
        
        # Analyze line angles
        angles = []
        for rho, theta in lines[:20]:  # Use first 20 lines
            angle = theta * 180 / np.pi
            # Convert to rotation angle
            if angle > 90:
                angle -= 180
            angles.append(angle)
        
        if angles:
            # Use median angle as rotation
            rotation_angle = np.median(angles)
            
            # Only return significant rotations
            if abs(rotation_angle) > 0.5:
                return -rotation_angle  # Negative for counter-clockwise correction
        
        return 0.0
    
    def _detect_skew(self, image: np.ndarray) -> float:
        """Detect document skew using text line analysis."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Find text lines using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        
        # Threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.bitwise_not(binary)
        
        # Dilate to connect text
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate skew from text line contours
        angles = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small contours
                rect = cv2.minAreaRect(contour)
                angle = rect[2]
                
                # Normalize angle
                if angle < -45:
                    angle += 90
                elif angle > 45:
                    angle -= 90
                
                angles.append(angle)
        
        if angles:
            return np.median(angles)
        
        return 0.0
    
    def _calculate_detection_confidence(self, image: np.ndarray) -> float:
        """Calculate confidence in document detection."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Factors affecting confidence:
        # 1. Edge strength
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 2. Contrast
        contrast = np.std(gray)
        
        # 3. Image quality (blur detection)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Combine factors
        confidence = min(1.0, (edge_density * 10 + contrast / 255 + blur_score / 1000) / 3)
        
        return max(0.0, confidence)
    
    def _correct_rotation(self, image: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
        """Correct document rotation."""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new dimensions
        cos_val = abs(rotation_matrix[0, 0])
        sin_val = abs(rotation_matrix[0, 1])
        new_width = int((height * sin_val) + (width * cos_val))
        new_height = int((height * cos_val) + (width * sin_val))
        
        # Adjust rotation matrix for new center
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Apply rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=(255, 255, 255))
        
        # Convert 2x3 to 3x3 matrix
        transform_3x3 = np.vstack([rotation_matrix, [0, 0, 1]])
        
        return rotated, transform_3x3
    
    def _correct_skew(self, image: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
        """Correct document skew."""
        # For small skew corrections, use simple rotation
        return self._correct_rotation(image, angle)
    
    def _scale_document(self, image: np.ndarray, scale_factor: float) -> Tuple[np.ndarray, np.ndarray]:
        """Scale document to target DPI."""
        height, width = image.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Resize image
        scaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Create scale transformation matrix
        scale_matrix = np.array([
            [scale_factor, 0, 0],
            [0, scale_factor, 0],
            [0, 0, 1]
        ])
        
        return scaled, scale_matrix


class AdaptiveTemplateManager:
    """Template manager that adapts to document variations."""
    
    def __init__(self, base_template_manager, normalizer: DocumentNormalizer):
        """Initialize adaptive template manager.
        
        Args:
            base_template_manager: Base template manager
            normalizer: Document normalizer
        """
        self.base_manager = base_template_manager
        self.normalizer = normalizer
        self._template_cache = {}
    
    def get_adaptive_template(self, template_name: str, document_image: np.ndarray):
        """Get template adapted for specific document.
        
        Args:
            template_name: Name of base template
            document_image: Document image to adapt for
            
        Returns:
            Adapted template with corrected bounding boxes
        """
        # Get base template
        base_template = self.base_manager.get_template(template_name)
        if not base_template:
            return None
        
        # Normalize document
        normalization_result = self.normalizer.normalize_document(document_image)
        
        if not normalization_result.success:
            print(f"Warning: Document normalization failed (confidence: {normalization_result.confidence:.2f})")
            return base_template
        
        # Create cache key
        cache_key = f"{template_name}_{hash(str(normalization_result.scale_factors))}"
        
        if cache_key in self._template_cache:
            return self._template_cache[cache_key]
        
        # Create adapted template
        adapted_template = self._create_adapted_template(base_template, normalization_result)
        self._template_cache[cache_key] = adapted_template
        
        return adapted_template
    
    def _create_adapted_template(self, base_template, normalization_result):
        """Create template with adapted bounding boxes."""
        from copy import deepcopy
        
        adapted_template = deepcopy(base_template)
        
        # Get original bounding boxes
        original_boxes = [field.bounding_box for field in adapted_template.fields]
        
        # Transform bounding boxes
        transformed_boxes = self.normalizer.transform_bounding_boxes(
            original_boxes, normalization_result
        )
        
        # Update field bounding boxes
        for field, new_bbox in zip(adapted_template.fields, transformed_boxes):
            field.bounding_box = new_bbox
        
        return adapted_template


class DocumentPreprocessor:
    """Preprocesses documents for optimal extraction quality."""
    
    def __init__(self):
        """Initialize document preprocessor."""
        self.normalizer = DocumentNormalizer()
    
    def preprocess_document(self, image_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Preprocess document for optimal extraction.
        
        Args:
            image_path: Path to document image
            
        Returns:
            Tuple of (preprocessed_image, preprocessing_info)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        preprocessing_info = {
            'original_size': image.shape[:2],
            'steps_applied': []
        }
        
        # Step 1: Noise reduction
        if self._needs_noise_reduction(image):
            image = cv2.bilateralFilter(image, 9, 75, 75)
            preprocessing_info['steps_applied'].append('noise_reduction')
        
        # Step 2: Normalize document
        normalization_result = self.normalizer.normalize_document(image)
        if normalization_result.success:
            image = normalization_result.normalized_image
            preprocessing_info['normalization'] = {
                'scale_factors': normalization_result.scale_factors,
                'confidence': normalization_result.confidence
            }
            preprocessing_info['steps_applied'].append('normalization')
        
        # Step 3: Enhance contrast if needed
        if self._needs_contrast_enhancement(image):
            image = self._enhance_contrast(image)
            preprocessing_info['steps_applied'].append('contrast_enhancement')
        
        # Step 4: Remove shadows/uneven illumination
        if self._has_uneven_illumination(image):
            image = self._correct_illumination(image)
            preprocessing_info['steps_applied'].append('illumination_correction')
        
        preprocessing_info['final_size'] = image.shape[:2]
        
        return image, preprocessing_info
    
    def _needs_noise_reduction(self, image: np.ndarray) -> bool:
        """Check if image needs noise reduction."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        noise_level = np.std(cv2.Laplacian(gray, cv2.CV_64F))
        return noise_level > 50  # Threshold for noisy images
    
    def _needs_contrast_enhancement(self, image: np.ndarray) -> bool:
        """Check if image needs contrast enhancement."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray)
        return contrast < 40  # Low contrast threshold
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    def _has_uneven_illumination(self, image: np.ndarray) -> bool:
        """Check for uneven illumination."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate illumination variation
        height, width = gray.shape
        center_brightness = np.mean(gray[height//4:3*height//4, width//4:3*width//4])
        edge_brightness = np.mean([
            np.mean(gray[:height//4, :]),  # Top
            np.mean(gray[3*height//4:, :]),  # Bottom
            np.mean(gray[:, :width//4]),  # Left
            np.mean(gray[:, 3*width//4:])  # Right
        ])
        
        brightness_ratio = abs(center_brightness - edge_brightness) / center_brightness
        return brightness_ratio > 0.2  # 20% variation threshold
    
    def _correct_illumination(self, image: np.ndarray) -> np.ndarray:
        """Correct uneven illumination."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create background model using large gaussian blur
        background = cv2.GaussianBlur(gray, (0, 0), sigmaX=20, sigmaY=20)
        
        # Subtract background to correct illumination
        corrected_gray = cv2.addWeighted(gray, 1.5, background, -0.5, 0)
        
        # Convert back to color by replacing luminance
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = corrected_gray
        
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)