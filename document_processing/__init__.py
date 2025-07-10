"""
Document Processing Module

Enhanced document processing tool for extracting structured data from forms
with strong typing for signatures, radio buttons, checkboxes, and other field types.

Now includes DPI variation handling, document normalization, and preprocessing
for handling real-world scanning inconsistencies.
"""

from .field_types import *
from .template_manager import DocumentTemplateManager
from .document_processor import DocumentProcessor
from .batch_processor import BatchDocumentProcessor
from .validators import *

# Enhanced components for DPI and quality handling
try:
    from .document_normalizer import DocumentNormalizer, AdaptiveTemplateManager, DocumentPreprocessor
    from .enhanced_document_processor import EnhancedDocumentProcessor
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    # Enhanced features require opencv-python and numpy
    ENHANCED_FEATURES_AVAILABLE = False
    print("Enhanced DPI handling features require: pip install opencv-python numpy")

__version__ = "1.1.0"  # Updated for DPI handling features

def get_recommended_processor(enable_dpi_handling=True, target_dpi=300.0):
    """Get the recommended document processor for your use case.
    
    Args:
        enable_dpi_handling: Whether to enable DPI variation handling
        target_dpi: Target DPI for normalization
        
    Returns:
        DocumentProcessor or EnhancedDocumentProcessor
    """
    if enable_dpi_handling and ENHANCED_FEATURES_AVAILABLE:
        return EnhancedDocumentProcessor(target_dpi=target_dpi, enable_preprocessing=True)
    else:
        if enable_dpi_handling and not ENHANCED_FEATURES_AVAILABLE:
            print("Warning: DPI handling requested but enhanced features not available")
            print("Install with: pip install opencv-python numpy")
        return DocumentProcessor()