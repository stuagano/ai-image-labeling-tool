"""
Extended ImageManager with AI annotation support
"""

import os
import numpy as np
from PIL import Image
from streamlit_img_label.manage import ImageManager as BaseImageManager
from typing import List, Dict, Any, Optional

class AIImageManager(BaseImageManager):
    """Extended ImageManager with AI annotation capabilities"""
    
    def __init__(self, filename, annotation_format="json"):
        super().__init__(filename, annotation_format)
        self._ai_annotations = []
        self._ai_suggestions = {}
    
    def set_ai_annotations(self, ai_annotations: List[Dict[str, Any]]):
        """
        Set AI-generated annotations
        
        Args:
            ai_annotations: List of AI detection dictionaries with bbox and label
        """
        self._ai_annotations = ai_annotations
        # Convert AI annotations to the format expected by the base class
        converted_annotations = []
        
        for ann in ai_annotations:
            converted_ann = {
                'left': ann.get('left', 0),
                'top': ann.get('top', 0),
                'width': ann.get('width', 0),
                'height': ann.get('height', 0),
                'label': ann.get('label', ''),
                'confidence': ann.get('confidence', 0.0),
                'source': 'ai'
            }
            converted_annotations.append(converted_ann)
        
        # Merge with existing annotations, avoiding duplicates
        existing_rects = self.get_rects()
        merged_rects = existing_rects.copy()
        
        for ai_ann in converted_annotations:
            # Check if this AI annotation overlaps significantly with existing ones
            is_duplicate = False
            for existing_ann in existing_rects:
                if self._calculate_iou(ai_ann, existing_ann) > 0.7:  # 70% overlap threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged_rects.append(ai_ann)
        
        # Update the internal rects
        self._rects = merged_rects
    
    def get_ai_annotations(self) -> List[Dict[str, Any]]:
        """Get AI-generated annotations"""
        return self._ai_annotations
    
    def set_ai_suggestions(self, suggestions: Dict[str, Any]):
        """Set AI suggestions for the image"""
        self._ai_suggestions = suggestions
    
    def get_ai_suggestions(self) -> Dict[str, Any]:
        """Get AI suggestions for the image"""
        return self._ai_suggestions
    
    def _calculate_iou(self, bbox1: Dict[str, Any], bbox2: Dict[str, Any]) -> float:
        """Calculate Intersection over Union between two bounding boxes"""
        x1_1, y1_1 = bbox1['left'], bbox1['top']
        x2_1, y2_1 = x1_1 + bbox1['width'], y1_1 + bbox1['height']
        
        x1_2, y1_2 = bbox2['left'], bbox2['top']
        x2_2, y2_2 = x1_2 + bbox2['width'], y1_2 + bbox2['height']
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = bbox1['width'] * bbox1['height']
        area2 = bbox2['width'] * bbox2['height']
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_annotations_with_source(self) -> List[Dict[str, Any]]:
        """Get all annotations with their source (manual or AI)"""
        annotations = []
        
        # Add manual annotations
        for rect in self.get_rects():
            if 'source' not in rect:
                rect['source'] = 'manual'
            annotations.append(rect)
        
        return annotations
    
    def filter_annotations_by_confidence(self, min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """Filter annotations by confidence threshold"""
        filtered = []
        for rect in self.get_rects():
            confidence = rect.get('confidence', 1.0)  # Manual annotations have full confidence
            if confidence >= min_confidence:
                filtered.append(rect)
        return filtered
    
    def get_annotation_statistics(self) -> Dict[str, Any]:
        """Get statistics about annotations"""
        rects = self.get_rects()
        
        stats = {
            'total_annotations': len(rects),
            'manual_annotations': len([r for r in rects if r.get('source') == 'manual']),
            'ai_annotations': len([r for r in rects if r.get('source') == 'ai']),
            'labels': {},
            'confidence_stats': {
                'min': 1.0,
                'max': 0.0,
                'avg': 0.0
            }
        }
        
        confidences = []
        for rect in rects:
            label = rect.get('label', '')
            if label:
                stats['labels'][label] = stats['labels'].get(label, 0) + 1
            
            confidence = rect.get('confidence', 1.0)
            confidences.append(confidence)
        
        if confidences:
            stats['confidence_stats']['min'] = min(confidences)
            stats['confidence_stats']['max'] = max(confidences)
            stats['confidence_stats']['avg'] = sum(confidences) / len(confidences)
        
        return stats
    
    def validate_ai_annotations(self) -> Dict[str, Any]:
        """Validate AI annotations for quality"""
        ai_rects = [r for r in self.get_rects() if r.get('source') == 'ai']
        
        validation = {
            'total_ai_annotations': len(ai_rects),
            'high_confidence': len([r for r in ai_rects if r.get('confidence', 0) > 0.8]),
            'medium_confidence': len([r for r in ai_rects if 0.5 <= r.get('confidence', 0) <= 0.8]),
            'low_confidence': len([r for r in ai_rects if r.get('confidence', 0) < 0.5]),
            'potential_issues': []
        }
        
        # Check for overlapping AI annotations
        for i, rect1 in enumerate(ai_rects):
            for j, rect2 in enumerate(ai_rects[i+1:], i+1):
                if self._calculate_iou(rect1, rect2) > 0.5:
                    validation['potential_issues'].append(
                        f"Overlapping AI annotations {i} and {j} (IoU: {self._calculate_iou(rect1, rect2):.2f})"
                    )
        
        # Check for annotations outside image bounds
        img_width, img_height = self.get_img().size
        for i, rect in enumerate(ai_rects):
            if (rect['left'] < 0 or rect['top'] < 0 or 
                rect['left'] + rect['width'] > img_width or 
                rect['top'] + rect['height'] > img_height):
                validation['potential_issues'].append(
                    f"AI annotation {i} extends outside image bounds"
                )
        
        return validation 