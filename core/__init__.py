"""
Core modules for AI Image Labeling Tool
Centralized functionality to eliminate DRY violations
"""

# Import modules that don't require external dependencies
from .storage_interface import StorageManagerInterface, BaseImageManager, StorageUtilities
from .math_utils import BoundingBoxUtils, StatisticsUtils, ValidationUtils

__all__ = [
    'StorageManagerInterface',
    'BaseImageManager',
    'StorageUtilities',
    'BoundingBoxUtils',
    'StatisticsUtils',
    'ValidationUtils'
]

# Conditionally import modules that require external dependencies
try:
    from .navigation import NavigationController, create_navigation_controller
    from .export_manager import ExportManager, create_export_manager
    __all__.extend([
        'NavigationController',
        'create_navigation_controller',
        'ExportManager', 
        'create_export_manager'
    ])
except ImportError:
    # These modules require streamlit and other dependencies
    pass

__version__ = '1.0.0'