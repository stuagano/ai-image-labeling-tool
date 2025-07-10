"""
Document Processing Module

Enhanced document processing tool for extracting structured data from forms
with strong typing for signatures, radio buttons, checkboxes, and other field types.
"""

from .field_types import *
from .template_manager import DocumentTemplateManager
from .document_processor import DocumentProcessor
from .batch_processor import BatchDocumentProcessor
from .validators import *

__version__ = "1.0.0"