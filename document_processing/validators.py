"""
Validation Utilities

Comprehensive validation utilities for document processing results
with quality assurance and error detection capabilities.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import re

from .field_types import FieldExtraction, FormField, FieldType
from .template_manager import DocumentTemplate


class ValidationReport:
    """Comprehensive validation report for document processing results."""
    
    def __init__(self):
        self.total_fields = 0
        self.valid_fields = 0
        self.validation_errors = []
        self.confidence_warnings = []
        self.type_mismatches = []
        self.missing_required_fields = []
        self.duplicate_field_warnings = []
        self.overall_score = 0.0
        
    def add_field_result(self, field_name: str, extraction: FieldExtraction, field_def: FormField):
        """Add a field result to the validation report."""
        self.total_fields += 1
        
        if extraction.is_valid:
            self.valid_fields += 1
        else:
            self.validation_errors.extend([
                f"{field_name}: {error}" for error in extraction.validation_errors
            ])
        
        # Check confidence level
        if extraction.confidence < 0.5:
            self.confidence_warnings.append(
                f"{field_name}: Low confidence ({extraction.confidence:.2f})"
            )
        
        # Check if required field is missing
        if field_def.required and not extraction.value:
            self.missing_required_fields.append(field_name)
    
    def calculate_overall_score(self) -> float:
        """Calculate overall validation score (0-100)."""
        if self.total_fields == 0:
            return 0.0
        
        base_score = (self.valid_fields / self.total_fields) * 100
        
        # Apply penalties
        confidence_penalty = len(self.confidence_warnings) * 5
        missing_required_penalty = len(self.missing_required_fields) * 10
        
        self.overall_score = max(0, base_score - confidence_penalty - missing_required_penalty)
        return self.overall_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            'total_fields': self.total_fields,
            'valid_fields': self.valid_fields,
            'validation_rate': self.valid_fields / self.total_fields if self.total_fields > 0 else 0,
            'validation_errors': self.validation_errors,
            'confidence_warnings': self.confidence_warnings,
            'missing_required_fields': self.missing_required_fields,
            'overall_score': self.overall_score,
            'timestamp': datetime.now().isoformat()
        }


class DocumentValidator:
    """Validates document processing results against templates and business rules."""
    
    def __init__(self, min_confidence_threshold: float = 0.5):
        """Initialize validator.
        
        Args:
            min_confidence_threshold: Minimum confidence for field acceptance
        """
        self.min_confidence_threshold = min_confidence_threshold
    
    def validate_document_results(
        self,
        field_results: Dict[str, FieldExtraction],
        template: DocumentTemplate
    ) -> ValidationReport:
        """Validate document processing results against template.
        
        Args:
            field_results: Results from document processing
            template: Document template used
            
        Returns:
            Validation report
        """
        report = ValidationReport()
        
        # Validate each field
        for field in template.fields:
            extraction = field_results.get(field.name)
            
            if extraction:
                report.add_field_result(field.name, extraction, field)
            else:
                # Missing field
                if field.required:
                    report.missing_required_fields.append(field.name)
                report.total_fields += 1
        
        # Calculate overall score
        report.calculate_overall_score()
        
        return report
    
    def validate_signature_field(self, extraction: FieldExtraction) -> List[str]:
        """Validate signature field specifically.
        
        Args:
            extraction: Signature field extraction
            
        Returns:
            List of validation issues
        """
        issues = []
        
        if not extraction.value:
            issues.append("No signature data found")
            return issues
        
        if isinstance(extraction.value, dict):
            signature_data = extraction.value
            
            # Check ink detection
            if not signature_data.get('ink_detected', False):
                issues.append("No ink signature detected")
            
            # Check coverage
            coverage = signature_data.get('coverage', 0)
            if coverage < 0.01:
                issues.append(f"Signature coverage too low: {coverage*100:.1f}%")
            
            # Check confidence
            if extraction.confidence < 0.7:
                issues.append(f"Low signature confidence: {extraction.confidence:.2f}")
        
        return issues
    
    def validate_checkbox_consistency(
        self, field_results: Dict[str, FieldExtraction]
    ) -> List[str]:
        """Validate checkbox field consistency.
        
        Args:
            field_results: All field results
            
        Returns:
            List of consistency issues
        """
        issues = []
        checkbox_fields = {}
        
        # Collect checkbox fields
        for field_name, extraction in field_results.items():
            if extraction.extraction_method == "computer_vision" and isinstance(extraction.value, bool):
                checkbox_fields[field_name] = extraction
        
        # Check for mutual exclusivity issues (this would need business rules)
        # For now, just check confidence levels
        for field_name, extraction in checkbox_fields.items():
            if extraction.confidence < 0.6:
                issues.append(f"Checkbox '{field_name}' has low confidence: {extraction.confidence:.2f}")
        
        return issues
    
    def validate_data_consistency(
        self, field_results: Dict[str, FieldExtraction]
    ) -> List[str]:
        """Validate data consistency across fields.
        
        Args:
            field_results: All field results
            
        Returns:
            List of consistency issues
        """
        issues = []
        
        # Example: Check date consistency
        date_fields = []
        for field_name, extraction in field_results.items():
            if extraction.value and isinstance(extraction.value, datetime):
                date_fields.append((field_name, extraction.value))
        
        # Check for logical date ordering (example business rule)
        if len(date_fields) >= 2:
            date_fields.sort(key=lambda x: x[1])
            # Add custom date validation logic here
        
        return issues


class QualityAssurance:
    """Quality assurance utilities for batch processing results."""
    
    @staticmethod
    def analyze_batch_quality(
        batch_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze quality metrics across batch results.
        
        Args:
            batch_results: List of document processing results
            
        Returns:
            Quality analysis report
        """
        total_documents = len(batch_results)
        
        if total_documents == 0:
            return {'error': 'No documents to analyze'}
        
        # Aggregate statistics
        total_fields = 0
        valid_fields = 0
        confidence_scores = []
        field_types_stats = {}
        
        for doc_result in batch_results:
            field_results = doc_result.get('field_results', {})
            
            for field_name, field_data in field_results.items():
                total_fields += 1
                
                if field_data.get('is_valid', False):
                    valid_fields += 1
                
                confidence = field_data.get('confidence', 0)
                confidence_scores.append(confidence)
                
                # Track by extraction method
                method = field_data.get('extraction_method', 'unknown')
                if method not in field_types_stats:
                    field_types_stats[method] = {'count': 0, 'avg_confidence': 0}
                
                field_types_stats[method]['count'] += 1
                current_avg = field_types_stats[method]['avg_confidence']
                new_avg = (current_avg * (field_types_stats[method]['count'] - 1) + confidence) / field_types_stats[method]['count']
                field_types_stats[method]['avg_confidence'] = new_avg
        
        # Calculate metrics
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        min_confidence = min(confidence_scores) if confidence_scores else 0
        max_confidence = max(confidence_scores) if confidence_scores else 0
        
        # Quality thresholds
        high_quality_threshold = 0.8
        medium_quality_threshold = 0.6
        
        high_quality_fields = sum(1 for c in confidence_scores if c >= high_quality_threshold)
        medium_quality_fields = sum(1 for c in confidence_scores if medium_quality_threshold <= c < high_quality_threshold)
        low_quality_fields = sum(1 for c in confidence_scores if c < medium_quality_threshold)
        
        return {
            'summary': {
                'total_documents': total_documents,
                'total_fields': total_fields,
                'valid_fields': valid_fields,
                'validation_rate': valid_fields / total_fields if total_fields > 0 else 0
            },
            'confidence_analysis': {
                'average_confidence': avg_confidence,
                'min_confidence': min_confidence,
                'max_confidence': max_confidence,
                'high_quality_fields': high_quality_fields,
                'medium_quality_fields': medium_quality_fields,
                'low_quality_fields': low_quality_fields,
                'quality_distribution': {
                    'high': high_quality_fields / total_fields if total_fields > 0 else 0,
                    'medium': medium_quality_fields / total_fields if total_fields > 0 else 0,
                    'low': low_quality_fields / total_fields if total_fields > 0 else 0
                }
            },
            'extraction_methods': field_types_stats,
            'recommendations': QualityAssurance._generate_recommendations(
                avg_confidence, validation_rate=valid_fields / total_fields if total_fields > 0 else 0
            )
        }
    
    @staticmethod
    def _generate_recommendations(avg_confidence: float, validation_rate: float) -> List[str]:
        """Generate recommendations based on quality metrics."""
        recommendations = []
        
        if avg_confidence < 0.6:
            recommendations.append("Consider using higher resolution images or better AI models")
        
        if validation_rate < 0.8:
            recommendations.append("Review template field definitions and validation rules")
        
        if avg_confidence < 0.7 and validation_rate > 0.9:
            recommendations.append("Confidence scores are low but validation is high - consider adjusting confidence thresholds")
        
        if len(recommendations) == 0:
            recommendations.append("Quality metrics look good!")
        
        return recommendations
    
    @staticmethod
    def identify_problematic_fields(
        batch_results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Identify fields that consistently have issues across documents.
        
        Args:
            batch_results: List of document processing results
            
        Returns:
            Dictionary of problematic fields with statistics
        """
        field_issues = {}
        
        for doc_result in batch_results:
            field_results = doc_result.get('field_results', {})
            
            for field_name, field_data in field_results.items():
                if field_name not in field_issues:
                    field_issues[field_name] = {
                        'total_occurrences': 0,
                        'validation_failures': 0,
                        'low_confidence_count': 0,
                        'confidence_scores': [],
                        'common_errors': []
                    }
                
                stats = field_issues[field_name]
                stats['total_occurrences'] += 1
                
                if not field_data.get('is_valid', False):
                    stats['validation_failures'] += 1
                
                confidence = field_data.get('confidence', 0)
                stats['confidence_scores'].append(confidence)
                
                if confidence < 0.5:
                    stats['low_confidence_count'] += 1
                
                # Collect errors
                errors = field_data.get('validation_errors', [])
                stats['common_errors'].extend(errors)
        
        # Calculate final statistics and identify problematic fields
        problematic_fields = {}
        
        for field_name, stats in field_issues.items():
            failure_rate = stats['validation_failures'] / stats['total_occurrences']
            avg_confidence = sum(stats['confidence_scores']) / len(stats['confidence_scores'])
            low_confidence_rate = stats['low_confidence_count'] / stats['total_occurrences']
            
            # Consider a field problematic if it has high failure rate or low confidence
            if failure_rate > 0.3 or avg_confidence < 0.6 or low_confidence_rate > 0.4:
                problematic_fields[field_name] = {
                    'failure_rate': failure_rate,
                    'average_confidence': avg_confidence,
                    'low_confidence_rate': low_confidence_rate,
                    'total_occurrences': stats['total_occurrences'],
                    'most_common_errors': list(set(stats['common_errors']))[:5]  # Top 5 unique errors
                }
        
        return problematic_fields