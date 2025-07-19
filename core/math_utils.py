"""
Mathematical Utilities for Image Labeling Application
Centralizes mathematical operations to eliminate DRY violations
"""

from typing import List, Dict, Tuple, Any, Union
import statistics
import math


class BoundingBoxUtils:
    """Utilities for bounding box operations"""
    
    @staticmethod
    def calculate_iou(bbox1: Union[List, Dict], bbox2: Union[List, Dict]) -> float:
        """
        Calculate Intersection over Union between two bounding boxes
        
        Args:
            bbox1: First bounding box [x, y, width, height] or dict with keys
            bbox2: Second bounding box [x, y, width, height] or dict with keys
            
        Returns:
            IoU value between 0 and 1
        """
        # Normalize input formats
        x1_1, y1_1, w1, h1 = BoundingBoxUtils._normalize_bbox(bbox1)
        x1_2, y1_2, w2, h2 = BoundingBoxUtils._normalize_bbox(bbox2)
        
        # Convert to corner coordinates
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def _normalize_bbox(bbox: Union[List, Dict]) -> Tuple[float, float, float, float]:
        """
        Normalize bounding box to (x, y, width, height) format
        
        Args:
            bbox: Bounding box in various formats
            
        Returns:
            Tuple of (x, y, width, height)
        """
        if isinstance(bbox, dict):
            # Handle dictionary format
            if all(key in bbox for key in ['left', 'top', 'width', 'height']):
                return bbox['left'], bbox['top'], bbox['width'], bbox['height']
            elif all(key in bbox for key in ['x', 'y', 'width', 'height']):
                return bbox['x'], bbox['y'], bbox['width'], bbox['height']
            elif all(key in bbox for key in ['bbox']):
                # Handle nested bbox format
                return BoundingBoxUtils._normalize_bbox(bbox['bbox'])
            else:
                raise ValueError(f"Unsupported bbox dictionary format: {bbox.keys()}")
        elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        else:
            raise ValueError(f"Unsupported bbox format: {type(bbox)}")
    
    @staticmethod
    def calculate_area(bbox: Union[List, Dict]) -> float:
        """
        Calculate area of a bounding box
        
        Args:
            bbox: Bounding box in supported format
            
        Returns:
            Area of the bounding box
        """
        _, _, width, height = BoundingBoxUtils._normalize_bbox(bbox)
        return width * height
    
    @staticmethod
    def calculate_overlap_percentage(bbox1: Union[List, Dict], bbox2: Union[List, Dict]) -> float:
        """
        Calculate what percentage of bbox1 is overlapped by bbox2
        
        Args:
            bbox1: First bounding box
            bbox2: Second bounding box
            
        Returns:
            Percentage of bbox1 overlapped by bbox2 (0.0 to 1.0)
        """
        iou = BoundingBoxUtils.calculate_iou(bbox1, bbox2)
        area1 = BoundingBoxUtils.calculate_area(bbox1)
        area2 = BoundingBoxUtils.calculate_area(bbox2)
        
        if area1 == 0:
            return 0.0
        
        # Calculate intersection area from IoU
        union = area1 + area2 - (iou * (area1 + area2) / (1 + iou)) if iou > 0 else area1 + area2
        intersection = iou * union
        
        return intersection / area1
    
    @staticmethod
    def is_bbox_valid(bbox: Union[List, Dict], img_width: int = None, img_height: int = None) -> bool:
        """
        Validate bounding box
        
        Args:
            bbox: Bounding box to validate
            img_width: Optional image width for bounds checking
            img_height: Optional image height for bounds checking
            
        Returns:
            True if valid, False otherwise
        """
        try:
            x, y, width, height = BoundingBoxUtils._normalize_bbox(bbox)
            
            # Check for positive dimensions
            if width <= 0 or height <= 0:
                return False
            
            # Check for non-negative coordinates
            if x < 0 or y < 0:
                return False
            
            # Check bounds if image dimensions provided
            if img_width is not None and img_height is not None:
                if x + width > img_width or y + height > img_height:
                    return False
            
            return True
            
        except (ValueError, KeyError, TypeError):
            return False
    
    @staticmethod
    def convert_bbox_format(bbox: Union[List, Dict], 
                           from_format: str = "xywh", 
                           to_format: str = "xyxy") -> List[float]:
        """
        Convert bounding box between different formats
        
        Args:
            bbox: Input bounding box
            from_format: Source format ("xywh", "xyxy", "cxcywh")
            to_format: Target format ("xywh", "xyxy", "cxcywh")
            
        Returns:
            Converted bounding box as list
        """
        if from_format == to_format:
            x, y, w, h = BoundingBoxUtils._normalize_bbox(bbox)
            return [x, y, w, h]
        
        # First normalize to xywh
        x, y, w, h = BoundingBoxUtils._normalize_bbox(bbox)
        
        if from_format == "xyxy":
            # Input is [x1, y1, x2, y2], convert to [x, y, w, h]
            w = w - x  # w was actually x2
            h = h - y  # h was actually y2
        elif from_format == "cxcywh":
            # Input is [center_x, center_y, width, height], convert to [x, y, w, h]
            x = x - w / 2  # x was center_x
            y = y - h / 2  # y was center_y
        
        # Now convert from xywh to target format
        if to_format == "xyxy":
            return [x, y, x + w, y + h]
        elif to_format == "cxcywh":
            return [x + w / 2, y + h / 2, w, h]
        else:  # to_format == "xywh"
            return [x, y, w, h]


class StatisticsUtils:
    """Utilities for statistical operations"""
    
    @staticmethod
    def calculate_confidence_statistics(confidences: List[float]) -> Dict[str, float]:
        """
        Calculate confidence statistics
        
        Args:
            confidences: List of confidence values
            
        Returns:
            Dictionary with statistics
        """
        if not confidences:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "count": 0
            }
        
        return {
            "min": float(min(confidences)),
            "max": float(max(confidences)),
            "mean": float(statistics.mean(confidences)),
            "median": float(statistics.median(confidences)),
            "std": float(statistics.stdev(confidences) if len(confidences) > 1 else 0.0),
            "count": len(confidences)
        }
    
    @staticmethod
    def calculate_label_distribution(annotations: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Calculate label distribution from annotations
        
        Args:
            annotations: List of annotation dictionaries
            
        Returns:
            Dictionary mapping labels to counts
        """
        distribution = {}
        
        for annotation in annotations:
            label = annotation.get('label', 'unknown')
            distribution[label] = distribution.get(label, 0) + 1
        
        return distribution
    
    @staticmethod
    def find_outliers(values: List[float], threshold: float = 2.0) -> List[int]:
        """
        Find outlier indices using standard deviation
        
        Args:
            values: List of numeric values
            threshold: Standard deviation threshold for outliers
            
        Returns:
            List of indices that are outliers
        """
        if len(values) < 2:
            return []
        
        mean = statistics.mean(values)
        std = statistics.stdev(values)
        
        if std == 0:
            return []
        
        outlier_indices = []
        for i, value in enumerate(values):
            z_score = abs((value - mean) / std)
            if z_score > threshold:
                outlier_indices.append(i)
        
        return outlier_indices


class ValidationUtils:
    """Utilities for validation operations"""
    
    @staticmethod
    def validate_annotations_quality(annotations: List[Dict[str, Any]], 
                                   image_width: int, 
                                   image_height: int) -> Dict[str, Any]:
        """
        Comprehensive validation of annotation quality
        
        Args:
            annotations: List of annotations to validate
            image_width: Width of the image
            image_height: Height of the image
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "total_annotations": len(annotations),
            "valid_annotations": 0,
            "invalid_annotations": 0,
            "out_of_bounds": 0,
            "overlapping_pairs": [],
            "tiny_boxes": 0,
            "large_boxes": 0,
            "issues": []
        }
        
        valid_annotations = []
        
        for i, annotation in enumerate(annotations):
            try:
                bbox = annotation.get('bbox', annotation)
                
                # Validate bbox format and bounds
                if not BoundingBoxUtils.is_bbox_valid(bbox, image_width, image_height):
                    results["invalid_annotations"] += 1
                    x, y, w, h = BoundingBoxUtils._normalize_bbox(bbox)
                    
                    if x < 0 or y < 0 or x + w > image_width or y + h > image_height:
                        results["out_of_bounds"] += 1
                        results["issues"].append(f"Annotation {i}: out of bounds")
                    
                    continue
                
                results["valid_annotations"] += 1
                valid_annotations.append((i, annotation))
                
                # Check for tiny boxes (less than 1% of image area)
                area = BoundingBoxUtils.calculate_area(bbox)
                image_area = image_width * image_height
                
                if area < 0.01 * image_area:
                    results["tiny_boxes"] += 1
                    results["issues"].append(f"Annotation {i}: tiny box (area={area:.1f})")
                
                # Check for large boxes (more than 50% of image area)
                if area > 0.5 * image_area:
                    results["large_boxes"] += 1
                    results["issues"].append(f"Annotation {i}: large box (area={area:.1f})")
                    
            except Exception as e:
                results["invalid_annotations"] += 1
                results["issues"].append(f"Annotation {i}: error - {e}")
        
        # Check for overlapping annotations
        for i, (idx1, ann1) in enumerate(valid_annotations):
            for j, (idx2, ann2) in enumerate(valid_annotations[i+1:], i+1):
                bbox1 = ann1.get('bbox', ann1)
                bbox2 = ann2.get('bbox', ann2)
                
                iou = BoundingBoxUtils.calculate_iou(bbox1, bbox2)
                if iou > 0.5:  # Significant overlap
                    results["overlapping_pairs"].append({
                        "indices": [idx1, idx2],
                        "iou": iou
                    })
                    results["issues"].append(f"Annotations {idx1} and {idx2}: high overlap (IoU={iou:.2f})")
        
        return results