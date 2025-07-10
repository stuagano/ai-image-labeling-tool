"""
Batch Document Processor

Handles batch processing of multiple documents using templates
and generates comprehensive reports on extraction results.
"""

import os
import json
import csv
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
import time
from tqdm import tqdm

from .document_processor import DocumentProcessor
from .template_manager import DocumentTemplate, DocumentTemplateManager
from .field_types import FieldExtraction


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing operations."""
    confidence_threshold: float = 0.5
    output_format: str = "json"  # json, csv, excel
    include_raw_data: bool = True
    include_statistics: bool = True
    parallel_processing: bool = False
    max_workers: int = 4
    validation_level: str = "moderate"


@dataclass
class DocumentResult:
    """Result of processing a single document."""
    document_path: str
    template_name: str
    field_results: Dict[str, FieldExtraction]
    processing_time: float
    statistics: Dict[str, Any]
    errors: List[str]
    timestamp: datetime


class BatchDocumentProcessor:
    """Batch processor for handling multiple documents with templates."""
    
    def __init__(
        self,
        template_manager: DocumentTemplateManager,
        ai_model_type: str = "yolo",
        api_key: Optional[str] = None
    ):
        """Initialize batch processor.
        
        Args:
            template_manager: Template manager instance
            ai_model_type: AI model type for document processing
            api_key: API key for cloud-based AI services
        """
        self.template_manager = template_manager
        self.document_processor = DocumentProcessor(ai_model_type, api_key)
        self.processing_results = []
    
    def process_directory(
        self,
        documents_dir: str,
        template_name: str,
        output_dir: str,
        config: BatchProcessingConfig = None
    ) -> Dict[str, Any]:
        """Process all documents in a directory using a template.
        
        Args:
            documents_dir: Directory containing document images
            template_name: Name of template to use
            output_dir: Directory to save results
            config: Processing configuration
            
        Returns:
            Batch processing results and statistics
        """
        if config is None:
            config = BatchProcessingConfig()
        
        # Get template
        template = self.template_manager.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        document_files = self._get_document_files(documents_dir)
        
        print(f"Processing {len(document_files)} documents with template '{template_name}'")
        
        # Process documents
        batch_results = []
        start_time = time.time()
        
        for doc_file in tqdm(document_files, desc="Processing documents"):
            doc_path = os.path.join(documents_dir, doc_file)
            
            try:
                doc_start_time = time.time()
                
                # Process single document
                field_results = self.document_processor.process_document(
                    doc_path, template, config.confidence_threshold
                )
                
                processing_time = time.time() - doc_start_time
                
                # Get statistics
                statistics = self.document_processor.get_processing_statistics(field_results)
                
                # Create document result
                doc_result = DocumentResult(
                    document_path=doc_path,
                    template_name=template_name,
                    field_results=field_results,
                    processing_time=processing_time,
                    statistics=statistics,
                    errors=[],
                    timestamp=datetime.now()
                )
                
                batch_results.append(doc_result)
                
            except Exception as e:
                # Handle processing errors
                error_result = DocumentResult(
                    document_path=doc_path,
                    template_name=template_name,
                    field_results={},
                    processing_time=0.0,
                    statistics={},
                    errors=[str(e)],
                    timestamp=datetime.now()
                )
                batch_results.append(error_result)
        
        total_processing_time = time.time() - start_time
        
        # Generate batch statistics
        batch_stats = self._generate_batch_statistics(batch_results, total_processing_time)
        
        # Save results
        self._save_batch_results(batch_results, batch_stats, output_dir, config)
        
        return {
            'batch_statistics': batch_stats,
            'document_results': batch_results,
            'total_processing_time': total_processing_time
        }
    
    def process_documents_with_auto_template(
        self,
        documents_dir: str,
        output_dir: str,
        config: BatchProcessingConfig = None
    ) -> Dict[str, Any]:
        """Process documents with automatic template matching.
        
        Args:
            documents_dir: Directory containing document images
            output_dir: Directory to save results
            config: Processing configuration
            
        Returns:
            Batch processing results grouped by template
        """
        if config is None:
            config = BatchProcessingConfig()
        
        document_files = self._get_document_files(documents_dir)
        template_groups = {}
        
        # Group documents by matched template
        for doc_file in document_files:
            doc_path = os.path.join(documents_dir, doc_file)
            
            # Try to match template
            matched_template = self.template_manager.match_template(doc_path)
            
            if matched_template:
                if matched_template not in template_groups:
                    template_groups[matched_template] = []
                template_groups[matched_template].append(doc_file)
            else:
                # Handle unmatched documents
                if 'unmatched' not in template_groups:
                    template_groups['unmatched'] = []
                template_groups['unmatched'].append(doc_file)
        
        # Process each template group
        all_results = {}
        for template_name, files in template_groups.items():
            if template_name != 'unmatched':
                # Create temporary directory for this template's files
                temp_dir = os.path.join(documents_dir, f"_temp_{template_name}")
                os.makedirs(temp_dir, exist_ok=True)
                
                # Copy files to temp directory
                for file in files:
                    src = os.path.join(documents_dir, file)
                    dst = os.path.join(temp_dir, file)
                    os.link(src, dst)  # Hard link to avoid copying
                
                # Process with specific template
                template_output_dir = os.path.join(output_dir, template_name)
                results = self.process_directory(temp_dir, template_name, template_output_dir, config)
                all_results[template_name] = results
                
                # Clean up temp directory
                import shutil
                shutil.rmtree(temp_dir)
        
        return all_results
    
    def validate_batch_results(
        self, results_file: str
    ) -> Dict[str, Any]:
        """Validate and analyze batch processing results.
        
        Args:
            results_file: Path to batch results JSON file
            
        Returns:
            Validation report
        """
        with open(results_file, 'r') as f:
            results_data = json.load(f)
        
        validation_report = {
            'total_documents': len(results_data.get('document_results', [])),
            'validation_issues': [],
            'field_analysis': {},
            'confidence_analysis': {},
            'error_analysis': {}
        }
        
        # Analyze each document result
        for doc_result in results_data.get('document_results', []):
            doc_path = doc_result['document_path']
            
            # Check for errors
            if doc_result.get('errors'):
                validation_report['error_analysis'][doc_path] = doc_result['errors']
            
            # Analyze field results
            for field_name, field_data in doc_result.get('field_results', {}).items():
                if field_name not in validation_report['field_analysis']:
                    validation_report['field_analysis'][field_name] = {
                        'total_extractions': 0,
                        'valid_extractions': 0,
                        'average_confidence': 0,
                        'common_errors': []
                    }
                
                field_stats = validation_report['field_analysis'][field_name]
                field_stats['total_extractions'] += 1
                
                if field_data.get('is_valid', False):
                    field_stats['valid_extractions'] += 1
                
                confidence = field_data.get('confidence', 0)
                field_stats['average_confidence'] = (
                    (field_stats['average_confidence'] * (field_stats['total_extractions'] - 1) + confidence)
                    / field_stats['total_extractions']
                )
                
                # Collect validation errors
                errors = field_data.get('validation_errors', [])
                field_stats['common_errors'].extend(errors)
        
        return validation_report
    
    def _get_document_files(self, documents_dir: str) -> List[str]:
        """Get all supported document files from directory.
        
        Args:
            documents_dir: Directory to scan
            
        Returns:
            List of document filenames
        """
        supported_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.pdf']
        document_files = []
        
        for file in os.listdir(documents_dir):
            if any(file.lower().endswith(ext) for ext in supported_extensions):
                document_files.append(file)
        
        return sorted(document_files)
    
    def _generate_batch_statistics(
        self, batch_results: List[DocumentResult], total_time: float
    ) -> Dict[str, Any]:
        """Generate comprehensive statistics for batch processing.
        
        Args:
            batch_results: List of document processing results
            total_time: Total processing time
            
        Returns:
            Batch statistics dictionary
        """
        total_docs = len(batch_results)
        successful_docs = sum(1 for r in batch_results if not r.errors)
        failed_docs = total_docs - successful_docs
        
        # Calculate average processing time
        avg_processing_time = (
            sum(r.processing_time for r in batch_results) / total_docs
            if total_docs > 0 else 0
        )
        
        # Field-level statistics
        field_stats = {}
        for result in batch_results:
            for field_name, field_result in result.field_results.items():
                if field_name not in field_stats:
                    field_stats[field_name] = {
                        'total_extractions': 0,
                        'successful_extractions': 0,
                        'average_confidence': 0,
                        'validation_errors': 0
                    }
                
                stats = field_stats[field_name]
                stats['total_extractions'] += 1
                
                if field_result.is_valid:
                    stats['successful_extractions'] += 1
                
                if field_result.validation_errors:
                    stats['validation_errors'] += len(field_result.validation_errors)
                
                # Update running average of confidence
                current_avg = stats['average_confidence']
                new_confidence = field_result.confidence
                stats['average_confidence'] = (
                    (current_avg * (stats['total_extractions'] - 1) + new_confidence)
                    / stats['total_extractions']
                )
        
        return {
            'summary': {
                'total_documents': total_docs,
                'successful_documents': successful_docs,
                'failed_documents': failed_docs,
                'success_rate': successful_docs / total_docs if total_docs > 0 else 0,
                'total_processing_time': total_time,
                'average_processing_time': avg_processing_time,
                'documents_per_minute': total_docs / (total_time / 60) if total_time > 0 else 0
            },
            'field_statistics': field_stats,
            'processing_timestamp': datetime.now().isoformat()
        }
    
    def _save_batch_results(
        self,
        batch_results: List[DocumentResult],
        batch_stats: Dict[str, Any],
        output_dir: str,
        config: BatchProcessingConfig
    ):
        """Save batch processing results in specified format.
        
        Args:
            batch_results: List of document results
            batch_stats: Batch statistics
            output_dir: Output directory
            config: Processing configuration
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save complete results as JSON
        if config.output_format in ["json", "all"]:
            results_data = {
                'batch_statistics': batch_stats,
                'document_results': [self._document_result_to_dict(r) for r in batch_results],
                'configuration': config.__dict__,
                'timestamp': timestamp
            }
            
            json_file = os.path.join(output_dir, f"batch_results_{timestamp}.json")
            with open(json_file, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
        
        # Save field data as CSV
        if config.output_format in ["csv", "all"]:
            csv_file = os.path.join(output_dir, f"extracted_data_{timestamp}.csv")
            self._save_results_as_csv(batch_results, csv_file)
        
        # Save statistics summary
        stats_file = os.path.join(output_dir, f"statistics_{timestamp}.json")
        with open(stats_file, 'w') as f:
            json.dump(batch_stats, f, indent=2, default=str)
    
    def _document_result_to_dict(self, result: DocumentResult) -> Dict[str, Any]:
        """Convert DocumentResult to dictionary for JSON serialization."""
        return {
            'document_path': result.document_path,
            'template_name': result.template_name,
            'field_results': {
                name: {
                    'value': extraction.value,
                    'confidence': extraction.confidence,
                    'is_valid': extraction.is_valid,
                    'validation_errors': extraction.validation_errors,
                    'extraction_method': extraction.extraction_method,
                    'timestamp': extraction.timestamp.isoformat()
                }
                for name, extraction in result.field_results.items()
            },
            'processing_time': result.processing_time,
            'statistics': result.statistics,
            'errors': result.errors,
            'timestamp': result.timestamp.isoformat()
        }
    
    def _save_results_as_csv(self, batch_results: List[DocumentResult], csv_file: str):
        """Save extraction results as CSV file.
        
        Args:
            batch_results: List of document results
            csv_file: Output CSV file path
        """
        if not batch_results:
            return
        
        # Get all unique field names
        all_field_names = set()
        for result in batch_results:
            all_field_names.update(result.field_results.keys())
        
        field_names = sorted(list(all_field_names))
        
        # Create CSV headers
        headers = ['document_path', 'template_name', 'processing_time']
        for field_name in field_names:
            headers.extend([
                f"{field_name}_value",
                f"{field_name}_confidence",
                f"{field_name}_valid"
            ])
        
        # Write CSV data
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for result in batch_results:
                row = [
                    result.document_path,
                    result.template_name,
                    result.processing_time
                ]
                
                for field_name in field_names:
                    field_result = result.field_results.get(field_name)
                    if field_result:
                        row.extend([
                            field_result.value,
                            field_result.confidence,
                            field_result.is_valid
                        ])
                    else:
                        row.extend(['', 0, False])
                
                writer.writerow(row)