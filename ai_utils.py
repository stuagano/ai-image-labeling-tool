"""
AI utilities for automatic object detection and image labeling
"""

import os
import cv2
import numpy as np
import requests
from PIL import Image
import torch
from ultralytics import YOLO
from transformers import pipeline
import google.generativeai as genai
from typing import List, Dict, Tuple, Optional
import json
import base64
from io import BytesIO

class AIObjectDetector:
    """AI-powered object detection and labeling"""
    
    def __init__(self, model_type="yolo", api_key=None):
        """
        Initialize AI object detector
        
        Args:
            model_type: "yolo", "transformers", or "gemini"
            api_key: API key for cloud services
        """
        self.model_type = model_type
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        # Initialize models
        if model_type == "yolo":
            self._init_yolo()
        elif model_type == "transformers":
            self._init_transformers()
        elif model_type == "gemini":
            self._init_gemini()
    
    def _init_yolo(self):
        """Initialize YOLO model"""
        try:
            # Try to load a pre-trained YOLO model
            self.yolo_model = YOLO('yolov8n.pt')  # nano model for speed
        except Exception as e:
            print(f"Warning: Could not load YOLO model: {e}")
            self.yolo_model = None
    
    def _init_transformers(self):
        """Initialize Transformers model"""
        try:
            self.detection_pipeline = pipeline(
                "object-detection",
                model="facebook/detr-resnet-50",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            print(f"Warning: Could not load Transformers model: {e}")
            self.detection_pipeline = None
    
    def _init_gemini(self):
        """Initialize Google Gemini model"""
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.gemini_model = genai.GenerativeModel('gemini-pro-vision')
            except Exception as e:
                print(f"Warning: Could not initialize Gemini: {e}")
                self.gemini_model = None
        else:
            print("Warning: No API key provided for Gemini")
            self.gemini_model = None
    
    def detect_objects_yolo(self, image_path: str) -> List[Dict]:
        """Detect objects using YOLO"""
        if not self.yolo_model:
            return []
        
        try:
            results = self.yolo_model(image_path)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        label = result.names[cls]
                        
                        detections.append({
                            'label': label,
                            'confidence': float(conf),
                            'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                            'bbox_xyxy': [int(x1), int(y1), int(x2), int(y2)]
                        })
            
            return detections
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return []
    
    def detect_objects_transformers(self, image_path: str) -> List[Dict]:
        """Detect objects using Transformers"""
        if not self.detection_pipeline:
            return []
        
        try:
            image = Image.open(image_path)
            results = self.detection_pipeline(image)
            
            detections = []
            for result in results:
                detections.append({
                    'label': result['label'],
                    'confidence': result['score'],
                    'bbox': [
                        int(result['box']['xmin']),
                        int(result['box']['ymin']),
                        int(result['box']['xmax'] - result['box']['xmin']),
                        int(result['box']['ymax'] - result['box']['ymin'])
                    ],
                    'bbox_xyxy': [
                        int(result['box']['xmin']),
                        int(result['box']['ymin']),
                        int(result['box']['xmax']),
                        int(result['box']['ymax'])
                    ]
                })
            
            return detections
        except Exception as e:
            print(f"Transformers detection error: {e}")
            return []
    
    def analyze_image_gemini(self, image_path: str, custom_prompt: str = None) -> Dict:
        """Analyze image using Google Gemini"""
        if not self.gemini_model:
            return {}
        
        try:
            image = Image.open(image_path)
            
            if custom_prompt:
                prompt = custom_prompt
            else:
                prompt = """
                Analyze this image and provide:
                1. List of objects visible in the image
                2. Their approximate locations (top-left, bottom-right coordinates)
                3. Confidence level for each detection
                4. Any relevant context or scene description
                
                Format the response as JSON with the following structure:
                {
                    "objects": [
                        {
                            "label": "object_name",
                            "confidence": 0.95,
                            "bbox": [x, y, width, height],
                            "description": "brief description"
                        }
                    ],
                    "scene_description": "overall scene description",
                    "suggested_labels": ["label1", "label2", "label3"]
                }
                """
            
            response = self.gemini_model.generate_content([prompt, image])
            
            try:
                # Try to parse JSON response
                result = json.loads(response.text)
                return result
            except json.JSONDecodeError:
                # If not JSON, return as text
                return {
                    "raw_response": response.text,
                    "objects": [],
                    "scene_description": response.text
                }
                
        except Exception as e:
            print(f"Gemini analysis error: {e}")
            return {}
    
    def detect_objects(self, image_path: str, confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Detect objects using the specified model
        
        Args:
            image_path: Path to the image
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            List of detection dictionaries
        """
        if self.model_type == "yolo":
            detections = self.detect_objects_yolo(image_path)
        elif self.model_type == "transformers":
            detections = self.detect_objects_transformers(image_path)
        elif self.model_type == "gemini":
            result = self.analyze_image_gemini(image_path)
            detections = result.get("objects", [])
        else:
            detections = []
        
        # Filter by confidence threshold
        filtered_detections = [
            det for det in detections 
            if det.get('confidence', 0) >= confidence_threshold
        ]
        
        return filtered_detections
    
    def get_suggested_labels(self, image_path: str) -> List[str]:
        """Get suggested labels for the image"""
        if self.model_type == "gemini":
            result = self.analyze_image_gemini(image_path)
            return result.get("suggested_labels", [])
        else:
            # For other models, extract unique labels from detections
            detections = self.detect_objects(image_path)
            labels = list(set([det['label'] for det in detections]))
            return labels

class AILabelingAssistant:
    """AI assistant for image labeling workflow"""
    
    def __init__(self, detector: AIObjectDetector):
        self.detector = detector
    
    def suggest_annotations(self, image_path: str, existing_labels: List[str] = None) -> Dict:
        """
        Suggest annotations for an image
        
        Args:
            image_path: Path to the image
            existing_labels: List of existing label categories
            
        Returns:
            Dictionary with suggestions
        """
        # Get AI detections
        detections = self.detector.detect_objects(image_path)
        
        # Get suggested labels
        suggested_labels = self.detector.get_suggested_labels(image_path)
        
        # Filter detections to match existing labels if provided
        if existing_labels:
            filtered_detections = [
                det for det in detections 
                if det['label'].lower() in [label.lower() for label in existing_labels]
            ]
        else:
            filtered_detections = detections
        
        return {
            'detections': filtered_detections,
            'suggested_labels': suggested_labels,
            'all_detections': detections,
            'confidence_scores': [det['confidence'] for det in detections],
            'unique_labels': list(set([det['label'] for det in detections]))
        }
    
    def validate_annotations(self, image_path: str, annotations: List[Dict]) -> Dict:
        """
        Validate existing annotations using AI
        
        Args:
            image_path: Path to the image
            annotations: List of existing annotations
            
        Returns:
            Validation results
        """
        ai_detections = self.detector.detect_objects(image_path)
        
        # Compare AI detections with manual annotations
        validation_results = {
            'ai_detections': ai_detections,
            'manual_annotations': annotations,
            'potential_misses': [],
            'potential_duplicates': [],
            'confidence_analysis': []
        }
        
        # Find potential misses (AI detected but not manually annotated)
        ai_bboxes = [det['bbox'] for det in ai_detections]
        manual_bboxes = [ann.get('bbox', []) for ann in annotations]
        
        for i, ai_det in enumerate(ai_detections):
            ai_bbox = ai_det['bbox']
            is_matched = False
            
            for manual_bbox in manual_bboxes:
                if self._calculate_iou(ai_bbox, manual_bbox) > 0.5:
                    is_matched = True
                    break
            
            if not is_matched:
                validation_results['potential_misses'].append(ai_det)
        
        return validation_results
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union between two bounding boxes"""
        if len(bbox1) != 4 or len(bbox2) != 4:
            return 0.0
        
        x1_1, y1_1, w1, h1 = bbox1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = bbox2
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

def create_ai_detector(model_type: str = "yolo", api_key: str = None) -> AIObjectDetector:
    """Factory function to create AI detector"""
    return AIObjectDetector(model_type, api_key)

def create_ai_assistant(detector: AIObjectDetector) -> AILabelingAssistant:
    """Factory function to create AI assistant"""
    return AILabelingAssistant(detector) 