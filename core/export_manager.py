"""
Centralized Export Manager for Image Labeling Application
Eliminates DRY violations in export functionality across different modules
"""

import os
import streamlit as st
from typing import Dict, Any, Optional
from streamlit_img_label.export import export_coco_dataset, export_yolo_format, export_csv, validate_annotations


class ExportManager:
    """Centralized export functionality for image labeling applications"""
    
    def __init__(self, img_dir: str, annotation_format: str = "json"):
        """
        Initialize export manager
        
        Args:
            img_dir: Directory containing images and annotations
            annotation_format: Format of annotations ("json", "coco", "xml")
        """
        self.img_dir = img_dir
        self.annotation_format = annotation_format
    
    def export_coco_dataset(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Export dataset in COCO format
        
        Args:
            output_file: Optional custom output file path
            
        Returns:
            Dictionary with status and message
        """
        if output_file is None:
            output_file = os.path.join(self.img_dir, "annotations_coco.json")
        
        try:
            export_coco_dataset(self.img_dir, output_file, self.annotation_format)
            return {
                "status": "success",
                "message": f"Exported to {output_file}",
                "output_file": output_file
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Export failed: {e}",
                "output_file": None
            }
    
    def export_yolo_format(self, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Export dataset in YOLO format
        
        Args:
            output_dir: Optional custom output directory
            
        Returns:
            Dictionary with status and message
        """
        if output_dir is None:
            output_dir = os.path.join(self.img_dir, "yolo_export")
        
        try:
            export_yolo_format(self.img_dir, output_dir, self.annotation_format)
            return {
                "status": "success",
                "message": f"Exported to {output_dir}",
                "output_dir": output_dir
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Export failed: {e}",
                "output_dir": None
            }
    
    def export_csv(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Export annotations in CSV format
        
        Args:
            output_file: Optional custom output file path
            
        Returns:
            Dictionary with status and message
        """
        if output_file is None:
            output_file = os.path.join(self.img_dir, "annotations.csv")
        
        try:
            export_csv(self.img_dir, output_file, self.annotation_format)
            return {
                "status": "success",
                "message": f"Exported to {output_file}",
                "output_file": output_file
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Export failed: {e}",
                "output_file": None
            }
    
    def validate_annotations(self) -> Dict[str, Any]:
        """
        Validate annotations and return comprehensive statistics
        
        Returns:
            Dictionary with validation results and statistics
        """
        try:
            stats = validate_annotations(self.img_dir, self.annotation_format)
            return {
                "status": "success",
                "stats": stats,
                "message": "Validation completed successfully"
            }
        except Exception as e:
            return {
                "status": "error",
                "stats": None,
                "message": f"Validation failed: {e}"
            }
    
    def render_export_sidebar(self):
        """Render export controls in sidebar"""
        st.sidebar.write("---")
        st.sidebar.write("**Export Options**")
        
        # COCO Export
        if st.sidebar.button("Export COCO Dataset"):
            result = self.export_coco_dataset()
            if result["status"] == "success":
                st.sidebar.success(result["message"])
            else:
                st.sidebar.error(result["message"])
        
        # YOLO Export
        if st.sidebar.button("Export YOLO Format"):
            result = self.export_yolo_format()
            if result["status"] == "success":
                st.sidebar.success(result["message"])
            else:
                st.sidebar.error(result["message"])
        
        # CSV Export
        if st.sidebar.button("Export CSV"):
            result = self.export_csv()
            if result["status"] == "success":
                st.sidebar.success(result["message"])
            else:
                st.sidebar.error(result["message"])
    
    def render_validation_sidebar(self):
        """Render validation controls in sidebar"""
        st.sidebar.write("---")
        st.sidebar.write("**Validation**")
        
        if st.sidebar.button("Validate Annotations"):
            result = self.validate_annotations()
            
            if result["status"] == "success":
                stats = result["stats"]
                
                # Display validation results
                st.sidebar.write("**Validation Results:**")
                st.sidebar.write(f"Total images: {stats.get('total_images', 0)}")
                st.sidebar.write(f"Annotated images: {stats.get('annotated_images', 0)}")
                st.sidebar.write(f"Total annotations: {stats.get('total_annotations', 0)}")
                st.sidebar.write(f"Empty labels: {stats.get('empty_labels', 0)}")
                st.sidebar.write(f"Overlapping boxes: {stats.get('overlapping_boxes', 0)}")
                
                # Show issues if any
                issues = stats.get('issues', [])
                if issues:
                    st.sidebar.write("**Issues found:**")
                    for issue in issues[:5]:  # Show first 5 issues
                        st.sidebar.write(f"• {issue}")
                    if len(issues) > 5:
                        st.sidebar.write(f"... and {len(issues) - 5} more")
                
                # Show label distribution
                label_distribution = stats.get('label_distribution', {})
                if label_distribution:
                    st.sidebar.write("**Label distribution:**")
                    sorted_labels = sorted(label_distribution.items(), key=lambda x: x[1], reverse=True)
                    for label, count in sorted_labels:
                        st.sidebar.write(f"• {label}: {count}")
            else:
                st.sidebar.error(result["message"])
    
    def get_export_formats(self) -> list:
        """Get list of available export formats"""
        return ["COCO", "YOLO", "CSV"]
    
    def export_all_formats(self, base_output_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Export dataset in all available formats
        
        Args:
            base_output_path: Base path for outputs
            
        Returns:
            Dictionary with results for each format
        """
        results = {}
        
        # Export COCO
        coco_file = os.path.join(base_output_path, "annotations_coco.json")
        results["coco"] = self.export_coco_dataset(coco_file)
        
        # Export YOLO
        yolo_dir = os.path.join(base_output_path, "yolo_export")
        results["yolo"] = self.export_yolo_format(yolo_dir)
        
        # Export CSV
        csv_file = os.path.join(base_output_path, "annotations.csv")
        results["csv"] = self.export_csv(csv_file)
        
        return results


def create_export_manager(img_dir: str, annotation_format: str = "json") -> ExportManager:
    """Factory function to create an export manager"""
    return ExportManager(img_dir, annotation_format)