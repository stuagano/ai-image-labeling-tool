"""
Comprehensive test suite for refactored core modules
Tests DRY compliance and class method functionality
"""

import unittest
import tempfile
import shutil
import os
import json
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Import the refactored modules
import sys
sys.path.append('..')

from core.navigation import NavigationController, create_navigation_controller
from core.export_manager import ExportManager, create_export_manager
from core.storage_interface import StorageManagerInterface, BaseImageManager, StorageUtilities
from core.math_utils import BoundingBoxUtils, StatisticsUtils, ValidationUtils


class TestNavigationController(unittest.TestCase):
    """Test navigation controller functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_idm = Mock()
        self.mock_idm.get_all_files.return_value = ['img1.jpg', 'img2.jpg', 'img3.jpg']
        self.mock_idm.get_exist_annotation_files.return_value = ['img1.json']
        self.mock_idm.get_next_annotation_image.return_value = 2
        
    @patch('streamlit.session_state', {})
    def test_initialization(self):
        """Test controller initialization"""
        nav_controller = NavigationController(self.mock_idm)
        
        # Verify session state is initialized
        import streamlit as st
        self.assertIn("files", st.session_state)
        self.assertIn("annotation_files", st.session_state)
        self.assertIn("image_index", st.session_state)
        
        # Verify initial values
        self.assertEqual(st.session_state["files"], ['img1.jpg', 'img2.jpg', 'img3.jpg'])
        self.assertEqual(st.session_state["annotation_files"], ['img1.json'])
        self.assertEqual(st.session_state["image_index"], 0)
    
    @patch('streamlit.session_state', {'files': ['img1.jpg', 'img2.jpg'], 'image_index': 0})
    def test_next_image(self):
        """Test next image navigation"""
        nav_controller = NavigationController(self.mock_idm)
        
        # Test normal navigation
        nav_controller.next_image()
        import streamlit as st
        self.assertEqual(st.session_state["image_index"], 1)
        
        # Test navigation at end
        with patch('streamlit.warning') as mock_warning:
            nav_controller.next_image()
            mock_warning.assert_called_with('This is the last image.')
    
    @patch('streamlit.session_state', {'files': ['img1.jpg', 'img2.jpg'], 'image_index': 1})
    def test_previous_image(self):
        """Test previous image navigation"""
        nav_controller = NavigationController(self.mock_idm)
        
        # Test normal navigation
        nav_controller.previous_image()
        import streamlit as st
        self.assertEqual(st.session_state["image_index"], 0)
        
        # Test navigation at beginning
        with patch('streamlit.warning') as mock_warning:
            nav_controller.previous_image()
            mock_warning.assert_called_with('This is the first image.')
    
    @patch('streamlit.session_state', {'files': ['img1.jpg', 'img2.jpg'], 'image_index': 0})
    def test_get_current_image_info(self):
        """Test getting current image information"""
        nav_controller = NavigationController(self.mock_idm)
        nav_controller.idm.get_all_files.return_value = ['img1.jpg', 'img2.jpg']
        
        import streamlit as st
        st.session_state["annotation_files"] = ['img1.json']
        
        info = nav_controller.get_current_image_info()
        
        expected = {
            "total_files": 2,
            "annotated_files": 1,
            "remaining_files": 1,
            "current_index": 0,
            "current_file": 'img1.jpg'
        }
        
        self.assertEqual(info, expected)


class TestExportManager(unittest.TestCase):
    """Test export manager functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.export_manager = ExportManager(self.temp_dir, "json")
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('streamlit_img_label.export.export_coco_dataset')
    def test_export_coco_dataset_success(self, mock_export):
        """Test successful COCO export"""
        mock_export.return_value = None  # Successful export
        
        result = self.export_manager.export_coco_dataset()
        
        self.assertEqual(result["status"], "success")
        self.assertIn("Exported to", result["message"])
        self.assertIsNotNone(result["output_file"])
    
    @patch('streamlit_img_label.export.export_coco_dataset')
    def test_export_coco_dataset_failure(self, mock_export):
        """Test failed COCO export"""
        mock_export.side_effect = Exception("Export failed")
        
        result = self.export_manager.export_coco_dataset()
        
        self.assertEqual(result["status"], "error")
        self.assertIn("Export failed", result["message"])
        self.assertIsNone(result["output_file"])
    
    @patch('streamlit_img_label.export.validate_annotations')
    def test_validate_annotations_success(self, mock_validate):
        """Test successful annotation validation"""
        mock_stats = {
            'total_images': 10,
            'annotated_images': 8,
            'total_annotations': 25,
            'empty_labels': 2,
            'overlapping_boxes': 1,
            'issues': ['Issue 1', 'Issue 2'],
            'label_distribution': {'cat': 10, 'dog': 15}
        }
        mock_validate.return_value = mock_stats
        
        result = self.export_manager.validate_annotations()
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["stats"], mock_stats)
    
    def test_get_export_formats(self):
        """Test getting available export formats"""
        formats = self.export_manager.get_export_formats()
        expected = ["COCO", "YOLO", "CSV"]
        self.assertEqual(formats, expected)


class TestMathUtils(unittest.TestCase):
    """Test mathematical utilities"""
    
    def test_calculate_iou(self):
        """Test IoU calculation"""
        bbox1 = [0, 0, 10, 10]  # x, y, width, height
        bbox2 = [5, 5, 10, 10]
        
        iou = BoundingBoxUtils.calculate_iou(bbox1, bbox2)
        
        # Expected IoU: intersection = 5*5 = 25, union = 100 + 100 - 25 = 175
        expected_iou = 25 / 175
        self.assertAlmostEqual(iou, expected_iou, places=4)
    
    def test_calculate_iou_no_overlap(self):
        """Test IoU calculation with no overlap"""
        bbox1 = [0, 0, 5, 5]
        bbox2 = [10, 10, 5, 5]
        
        iou = BoundingBoxUtils.calculate_iou(bbox1, bbox2)
        self.assertEqual(iou, 0.0)
    
    def test_calculate_iou_dict_format(self):
        """Test IoU calculation with dictionary format"""
        bbox1 = {'left': 0, 'top': 0, 'width': 10, 'height': 10}
        bbox2 = {'left': 5, 'top': 5, 'width': 10, 'height': 10}
        
        iou = BoundingBoxUtils.calculate_iou(bbox1, bbox2)
        
        expected_iou = 25 / 175
        self.assertAlmostEqual(iou, expected_iou, places=4)
    
    def test_normalize_bbox(self):
        """Test bounding box normalization"""
        # Test list format
        bbox_list = [10, 20, 30, 40]
        result = BoundingBoxUtils._normalize_bbox(bbox_list)
        self.assertEqual(result, (10.0, 20.0, 30.0, 40.0))
        
        # Test dict format
        bbox_dict = {'left': 10, 'top': 20, 'width': 30, 'height': 40}
        result = BoundingBoxUtils._normalize_bbox(bbox_dict)
        self.assertEqual(result, (10.0, 20.0, 30.0, 40.0))
    
    def test_is_bbox_valid(self):
        """Test bounding box validation"""
        # Valid bbox
        valid_bbox = [10, 20, 30, 40]
        self.assertTrue(BoundingBoxUtils.is_bbox_valid(valid_bbox))
        
        # Invalid bbox (negative dimensions)
        invalid_bbox = [10, 20, -30, 40]
        self.assertFalse(BoundingBoxUtils.is_bbox_valid(invalid_bbox))
        
        # Invalid bbox (out of bounds)
        out_of_bounds = [10, 20, 30, 40]
        self.assertFalse(BoundingBoxUtils.is_bbox_valid(out_of_bounds, 35, 35))
    
    def test_convert_bbox_format(self):
        """Test bounding box format conversion"""
        bbox_xywh = [10, 20, 30, 40]
        
        # Convert to xyxy
        bbox_xyxy = BoundingBoxUtils.convert_bbox_format(bbox_xywh, "xywh", "xyxy")
        expected_xyxy = [10, 20, 40, 60]
        self.assertEqual(bbox_xyxy, expected_xyxy)
        
        # Convert to center format
        bbox_cxcywh = BoundingBoxUtils.convert_bbox_format(bbox_xywh, "xywh", "cxcywh")
        expected_cxcywh = [25, 40, 30, 40]
        self.assertEqual(bbox_cxcywh, expected_cxcywh)


class TestStatisticsUtils(unittest.TestCase):
    """Test statistics utilities"""
    
    def test_calculate_confidence_statistics(self):
        """Test confidence statistics calculation"""
        confidences = [0.1, 0.5, 0.8, 0.9, 0.6]
        
        stats = StatisticsUtils.calculate_confidence_statistics(confidences)
        
        self.assertEqual(stats["min"], 0.1)
        self.assertEqual(stats["max"], 0.9)
        self.assertEqual(stats["count"], 5)
        self.assertAlmostEqual(stats["mean"], 0.58, places=2)
    
    def test_calculate_confidence_statistics_empty(self):
        """Test confidence statistics with empty list"""
        stats = StatisticsUtils.calculate_confidence_statistics([])
        
        expected = {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "count": 0
        }
        self.assertEqual(stats, expected)
    
    def test_calculate_label_distribution(self):
        """Test label distribution calculation"""
        annotations = [
            {'label': 'cat'},
            {'label': 'dog'},
            {'label': 'cat'},
            {'label': 'bird'},
            {'label': 'cat'}
        ]
        
        distribution = StatisticsUtils.calculate_label_distribution(annotations)
        
        expected = {'cat': 3, 'dog': 1, 'bird': 1}
        self.assertEqual(distribution, expected)
    
    def test_find_outliers(self):
        """Test outlier detection"""
        values = [1, 2, 3, 4, 5, 100]  # 100 is an outlier
        
        outliers = StatisticsUtils.find_outliers(values, threshold=2.0)
        
        self.assertIn(5, outliers)  # Index of the outlier (100)


class TestValidationUtils(unittest.TestCase):
    """Test validation utilities"""
    
    def test_validate_annotations_quality(self):
        """Test annotation quality validation"""
        annotations = [
            {'bbox': [10, 10, 20, 20], 'label': 'cat'},  # Valid
            {'bbox': [100, 100, 50, 50], 'label': 'dog'},  # Out of bounds
            {'bbox': [5, 5, 2, 2], 'label': 'bird'},  # Tiny box
        ]
        
        result = ValidationUtils.validate_annotations_quality(annotations, 100, 100)
        
        self.assertEqual(result["total_annotations"], 3)
        self.assertEqual(result["valid_annotations"], 2)  # First and third are valid
        self.assertEqual(result["out_of_bounds"], 1)
        self.assertEqual(result["tiny_boxes"], 1)


class TestStorageInterface(unittest.TestCase):
    """Test storage interface classes"""
    
    def test_storage_utilities_validate_json_data(self):
        """Test JSON data validation"""
        # Valid JSON data
        valid_data = {"key": "value", "number": 123}
        self.assertTrue(StorageUtilities.validate_json_data(valid_data))
        
        # Invalid JSON data (contains non-serializable object)
        invalid_data = {"key": lambda x: x}
        self.assertFalse(StorageUtilities.validate_json_data(invalid_data))
    
    def test_storage_utilities_sanitize_file_path(self):
        """Test file path sanitization"""
        # Test path with various issues
        messy_path = "//path\\to\\file//"
        sanitized = StorageUtilities.sanitize_file_path(messy_path)
        self.assertEqual(sanitized, "path/to/file")
    
    def test_storage_utilities_generate_unique_filename(self):
        """Test unique filename generation"""
        existing_files = ["test.txt", "test_1.txt"]
        
        unique_name = StorageUtilities.generate_unique_filename("test", "txt", existing_files)
        self.assertEqual(unique_name, "test_2.txt")
    
    def test_storage_utilities_batch_operation(self):
        """Test batch operation utility"""
        def test_operation(item):
            if item == "fail":
                raise Exception("Test error")
            return f"processed_{item}"
        
        items = ["item1", "item2", "fail", "item3"]
        result = StorageUtilities.batch_operation(test_operation, items)
        
        self.assertEqual(result["total"], 4)
        self.assertEqual(result["success_count"], 3)
        self.assertEqual(result["error_count"], 1)
        self.assertEqual(len(result["successful"]), 3)
        self.assertEqual(len(result["failed"]), 1)


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions"""
    
    def test_create_navigation_controller(self):
        """Test navigation controller factory"""
        mock_idm = Mock()
        mock_idm.get_all_files.return_value = []
        mock_idm.get_exist_annotation_files.return_value = []
        
        with patch('streamlit.session_state', {}):
            controller = create_navigation_controller(mock_idm)
            self.assertIsInstance(controller, NavigationController)
    
    def test_create_export_manager(self):
        """Test export manager factory"""
        manager = create_export_manager("test_dir", "json")
        self.assertIsInstance(manager, ExportManager)
        self.assertEqual(manager.img_dir, "test_dir")
        self.assertEqual(manager.annotation_format, "json")


class TestIntegration(unittest.TestCase):
    """Integration tests for multiple components"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up integration test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_annotation_workflow(self):
        """Test complete annotation workflow with refactored components"""
        # Create test annotations
        annotations = [
            {'bbox': [10, 10, 20, 20], 'label': 'cat', 'confidence': 0.9},
            {'bbox': [50, 50, 30, 30], 'label': 'dog', 'confidence': 0.8}
        ]
        
        # Test bounding box calculations
        iou = BoundingBoxUtils.calculate_iou(
            annotations[0]['bbox'], 
            annotations[1]['bbox']
        )
        self.assertEqual(iou, 0.0)  # No overlap
        
        # Test statistics
        confidences = [ann['confidence'] for ann in annotations]
        stats = StatisticsUtils.calculate_confidence_statistics(confidences)
        self.assertEqual(stats['count'], 2)
        self.assertAlmostEqual(stats['mean'], 0.85, places=2)
        
        # Test validation
        validation_result = ValidationUtils.validate_annotations_quality(
            annotations, 100, 100
        )
        self.assertEqual(validation_result['valid_annotations'], 2)
        self.assertEqual(validation_result['total_annotations'], 2)


if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestNavigationController,
        TestExportManager,
        TestMathUtils,
        TestStatisticsUtils,
        TestValidationUtils,
        TestStorageInterface,
        TestFactoryFunctions,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")