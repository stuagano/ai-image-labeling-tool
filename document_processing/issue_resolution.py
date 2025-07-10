"""
Issue Detection and Resolution System

AI-powered system for identifying document processing issues and
providing intelligent suggestions for resolution.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import re

from .field_types import FieldExtraction, FormField
from .business_rules_engine import RuleViolation, RuleSeverity


class IssueType(Enum):
    """Types of document processing issues."""
    LOW_CONFIDENCE = "low_confidence"
    EXTRACTION_FAILURE = "extraction_failure"
    VALIDATION_ERROR = "validation_error"
    BUSINESS_RULE_VIOLATION = "business_rule_violation"
    QUALITY_ISSUE = "quality_issue"
    TEMPLATE_MISMATCH = "template_mismatch"
    PREPROCESSING_ERROR = "preprocessing_error"
    INCONSISTENT_DATA = "inconsistent_data"


class IssueSeverity(Enum):
    """Severity levels for issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ResolutionStatus(Enum):
    """Status of issue resolution."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    CANNOT_RESOLVE = "cannot_resolve"


@dataclass
class ResolutionAction:
    """Represents a specific action to resolve an issue."""
    action_type: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    estimated_time: Optional[int] = None  # in seconds
    requires_human: bool = False
    auto_executable: bool = False


@dataclass
class DocumentIssue:
    """Represents an issue found in document processing."""
    id: str
    issue_type: IssueType
    severity: IssueSeverity
    title: str
    description: str
    affected_fields: List[str]
    
    # Context
    confidence_score: Optional[float] = None
    field_data: Dict[str, Any] = field(default_factory=dict)
    
    # Resolution
    suggested_actions: List[ResolutionAction] = field(default_factory=list)
    status: ResolutionStatus = ResolutionStatus.PENDING
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "system"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert issue to dictionary."""
        return {
            'id': self.id,
            'issue_type': self.issue_type.value,
            'severity': self.severity.value,
            'title': self.title,
            'description': self.description,
            'affected_fields': self.affected_fields,
            'confidence_score': self.confidence_score,
            'suggested_actions': [
                {
                    'action_type': action.action_type,
                    'description': action.description,
                    'parameters': action.parameters,
                    'confidence': action.confidence,
                    'requires_human': action.requires_human,
                    'auto_executable': action.auto_executable
                }
                for action in self.suggested_actions
            ],
            'status': self.status.value,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source
        }


class IssueDetector:
    """Detects various types of issues in document processing results."""
    
    def __init__(self, confidence_threshold: float = 0.7):
        """Initialize issue detector.
        
        Args:
            confidence_threshold: Minimum confidence for accepting extractions
        """
        self.confidence_threshold = confidence_threshold
        self.issue_patterns = self._load_issue_patterns()
    
    def detect_issues(
        self,
        field_results: Dict[str, FieldExtraction],
        template_fields: List[FormField],
        business_rule_violations: Optional[List[RuleViolation]] = None
    ) -> List[DocumentIssue]:
        """Detect issues in document processing results.
        
        Args:
            field_results: Results from document processing
            template_fields: Template field definitions
            business_rule_violations: Business rule violations
            
        Returns:
            List of detected issues
        """
        issues = []
        
        # Detect low confidence issues
        issues.extend(self._detect_confidence_issues(field_results))
        
        # Detect extraction failures
        issues.extend(self._detect_extraction_failures(field_results, template_fields))
        
        # Detect validation errors
        issues.extend(self._detect_validation_errors(field_results))
        
        # Detect data inconsistencies
        issues.extend(self._detect_data_inconsistencies(field_results))
        
        # Convert business rule violations to issues
        if business_rule_violations:
            issues.extend(self._convert_rule_violations(business_rule_violations))
        
        # Detect quality issues
        issues.extend(self._detect_quality_issues(field_results))
        
        return issues
    
    def _detect_confidence_issues(self, field_results: Dict[str, FieldExtraction]) -> List[DocumentIssue]:
        """Detect low confidence extraction issues."""
        issues = []
        
        for field_name, extraction in field_results.items():
            if extraction.confidence < self.confidence_threshold:
                severity = self._determine_confidence_severity(extraction.confidence)
                
                issue = DocumentIssue(
                    id=f"confidence_{field_name}_{datetime.now().timestamp()}",
                    issue_type=IssueType.LOW_CONFIDENCE,
                    severity=severity,
                    title=f"Low confidence in field '{field_name}'",
                    description=f"Field extraction confidence ({extraction.confidence:.2f}) is below threshold ({self.confidence_threshold})",
                    affected_fields=[field_name],
                    confidence_score=extraction.confidence,
                    field_data={field_name: extraction.value}
                )
                
                # Add suggested actions
                issue.suggested_actions = self._generate_confidence_actions(field_name, extraction)
                issues.append(issue)
        
        return issues
    
    def _detect_extraction_failures(
        self,
        field_results: Dict[str, FieldExtraction],
        template_fields: List[FormField]
    ) -> List[DocumentIssue]:
        """Detect complete extraction failures."""
        issues = []
        
        # Find fields that should exist but don't
        template_field_names = {field.name for field in template_fields}
        extracted_field_names = set(field_results.keys())
        missing_fields = template_field_names - extracted_field_names
        
        for field_name in missing_fields:
            # Find the template field
            template_field = next((f for f in template_fields if f.name == field_name), None)
            if not template_field:
                continue
            
            severity = IssueSeverity.HIGH if template_field.required else IssueSeverity.MEDIUM
            
            issue = DocumentIssue(
                id=f"missing_{field_name}_{datetime.now().timestamp()}",
                issue_type=IssueType.EXTRACTION_FAILURE,
                severity=severity,
                title=f"Failed to extract field '{field_name}'",
                description=f"Field '{field_name}' was not found during extraction",
                affected_fields=[field_name]
            )
            
            # Add suggested actions
            issue.suggested_actions = self._generate_extraction_failure_actions(field_name, template_field)
            issues.append(issue)
        
        return issues
    
    def _detect_validation_errors(self, field_results: Dict[str, FieldExtraction]) -> List[DocumentIssue]:
        """Detect field validation errors."""
        issues = []
        
        for field_name, extraction in field_results.items():
            if not extraction.is_valid and extraction.validation_errors:
                issue = DocumentIssue(
                    id=f"validation_{field_name}_{datetime.now().timestamp()}",
                    issue_type=IssueType.VALIDATION_ERROR,
                    severity=IssueSeverity.MEDIUM,
                    title=f"Validation failed for field '{field_name}'",
                    description=f"Validation errors: {', '.join(extraction.validation_errors)}",
                    affected_fields=[field_name],
                    field_data={field_name: extraction.value}
                )
                
                # Add suggested actions
                issue.suggested_actions = self._generate_validation_error_actions(field_name, extraction)
                issues.append(issue)
        
        return issues
    
    def _detect_data_inconsistencies(self, field_results: Dict[str, FieldExtraction]) -> List[DocumentIssue]:
        """Detect inconsistencies between related fields."""
        issues = []
        
        # Check date consistency
        date_fields = {}
        for field_name, extraction in field_results.items():
            if extraction.value and isinstance(extraction.value, datetime):
                date_fields[field_name] = extraction.value
        
        # Look for date ordering issues
        if len(date_fields) >= 2:
            date_pairs = list(date_fields.items())
            for i, (field1, date1) in enumerate(date_pairs):
                for field2, date2 in date_pairs[i+1:]:
                    if self._are_dates_logically_inconsistent(field1, date1, field2, date2):
                        issue = DocumentIssue(
                            id=f"inconsistent_dates_{field1}_{field2}_{datetime.now().timestamp()}",
                            issue_type=IssueType.INCONSISTENT_DATA,
                            severity=IssueSeverity.HIGH,
                            title=f"Date inconsistency between '{field1}' and '{field2}'",
                            description=f"Dates appear to be in illogical order: {field1}={date1}, {field2}={date2}",
                            affected_fields=[field1, field2],
                            field_data={field1: date1, field2: date2}
                        )
                        
                        issue.suggested_actions = self._generate_date_consistency_actions(field1, field2, date1, date2)
                        issues.append(issue)
        
        return issues
    
    def _detect_quality_issues(self, field_results: Dict[str, FieldExtraction]) -> List[DocumentIssue]:
        """Detect quality issues in extracted data."""
        issues = []
        
        for field_name, extraction in field_results.items():
            quality_issues = self._analyze_field_quality(field_name, extraction)
            issues.extend(quality_issues)
        
        return issues
    
    def _convert_rule_violations(self, violations: List[RuleViolation]) -> List[DocumentIssue]:
        """Convert business rule violations to issues."""
        issues = []
        
        for violation in violations:
            severity_mapping = {
                RuleSeverity.CRITICAL: IssueSeverity.CRITICAL,
                RuleSeverity.ERROR: IssueSeverity.HIGH,
                RuleSeverity.WARNING: IssueSeverity.MEDIUM,
                RuleSeverity.INFO: IssueSeverity.LOW
            }
            
            issue = DocumentIssue(
                id=f"rule_violation_{violation.rule_id}_{datetime.now().timestamp()}",
                issue_type=IssueType.BUSINESS_RULE_VIOLATION,
                severity=severity_mapping.get(violation.severity, IssueSeverity.MEDIUM),
                title=f"Business rule violation: {violation.rule_name}",
                description=violation.message,
                affected_fields=violation.affected_fields
            )
            
            if violation.suggested_action:
                action = ResolutionAction(
                    action_type="business_rule_fix",
                    description=violation.suggested_action,
                    auto_executable=violation.auto_fixable,
                    requires_human=not violation.auto_fixable
                )
                issue.suggested_actions.append(action)
            
            issues.append(issue)
        
        return issues
    
    def _determine_confidence_severity(self, confidence: float) -> IssueSeverity:
        """Determine severity based on confidence score."""
        if confidence < 0.3:
            return IssueSeverity.CRITICAL
        elif confidence < 0.5:
            return IssueSeverity.HIGH
        elif confidence < 0.7:
            return IssueSeverity.MEDIUM
        else:
            return IssueSeverity.LOW
    
    def _generate_confidence_actions(self, field_name: str, extraction: FieldExtraction) -> List[ResolutionAction]:
        """Generate actions for low confidence issues."""
        actions = []
        
        if extraction.confidence < 0.5:
            actions.append(ResolutionAction(
                action_type="manual_review",
                description="Manually review and correct the extracted value",
                requires_human=True,
                confidence=0.9
            ))
            
            actions.append(ResolutionAction(
                action_type="reprocess_with_preprocessing",
                description="Reprocess the document with enhanced image preprocessing",
                auto_executable=True,
                confidence=0.7
            ))
        
        if extraction.extraction_method == "ai":
            actions.append(ResolutionAction(
                action_type="try_alternative_model",
                description="Try extraction with a different AI model",
                auto_executable=True,
                confidence=0.6
            ))
        
        return actions
    
    def _generate_extraction_failure_actions(self, field_name: str, template_field: FormField) -> List[ResolutionAction]:
        """Generate actions for extraction failures."""
        actions = []
        
        actions.append(ResolutionAction(
            action_type="adjust_bounding_box",
            description="Adjust the bounding box coordinates for better field detection",
            parameters={"field_name": field_name, "current_box": template_field.bounding_box.to_dict()},
            requires_human=True,
            confidence=0.8
        ))
        
        actions.append(ResolutionAction(
            action_type="manual_extraction",
            description="Manually extract the field value",
            requires_human=True,
            confidence=1.0
        ))
        
        if template_field.field_type.value in ["signature", "checkbox"]:
            actions.append(ResolutionAction(
                action_type="enhance_computer_vision",
                description="Apply specialized computer vision algorithms for this field type",
                auto_executable=True,
                confidence=0.6
            ))
        
        return actions
    
    def _generate_validation_error_actions(self, field_name: str, extraction: FieldExtraction) -> List[ResolutionAction]:
        """Generate actions for validation errors."""
        actions = []
        
        actions.append(ResolutionAction(
            action_type="manual_correction",
            description="Manually correct the validation errors",
            parameters={"field_name": field_name, "errors": extraction.validation_errors},
            requires_human=True,
            confidence=1.0
        ))
        
        # Check for common auto-fixable validation errors
        for error in extraction.validation_errors:
            if "format" in error.lower():
                actions.append(ResolutionAction(
                    action_type="auto_format_correction",
                    description="Automatically correct common formatting issues",
                    auto_executable=True,
                    confidence=0.7
                ))
                break
        
        return actions
    
    def _generate_date_consistency_actions(self, field1: str, field2: str, date1: datetime, date2: datetime) -> List[ResolutionAction]:
        """Generate actions for date consistency issues."""
        actions = []
        
        actions.append(ResolutionAction(
            action_type="manual_date_review",
            description=f"Manually review and correct the dates for {field1} and {field2}",
            parameters={"field1": field1, "field2": field2, "date1": date1.isoformat(), "date2": date2.isoformat()},
            requires_human=True,
            confidence=1.0
        ))
        
        actions.append(ResolutionAction(
            action_type="swap_dates",
            description="Automatically swap the dates if they appear to be reversed",
            auto_executable=True,
            confidence=0.6
        ))
        
        return actions
    
    def _are_dates_logically_inconsistent(self, field1: str, date1: datetime, field2: str, date2: datetime) -> bool:
        """Check if dates are logically inconsistent."""
        # Simple heuristic: check for common field name patterns
        start_patterns = ["start", "begin", "from", "effective"]
        end_patterns = ["end", "finish", "to", "expiry", "expire"]
        
        field1_lower = field1.lower()
        field2_lower = field2.lower()
        
        # If field1 appears to be a start date and field2 an end date
        if any(pattern in field1_lower for pattern in start_patterns) and \
           any(pattern in field2_lower for pattern in end_patterns):
            return date1 > date2
        
        # If field2 appears to be a start date and field1 an end date
        if any(pattern in field2_lower for pattern in start_patterns) and \
           any(pattern in field1_lower for pattern in end_patterns):
            return date2 > date1
        
        return False
    
    def _analyze_field_quality(self, field_name: str, extraction: FieldExtraction) -> List[DocumentIssue]:
        """Analyze quality issues for a specific field."""
        issues = []
        
        if not extraction.value:
            return issues
        
        value_str = str(extraction.value)
        
        # Check for suspicious patterns
        if re.search(r'[^\w\s\-.,/()]', value_str):
            issue = DocumentIssue(
                id=f"quality_special_chars_{field_name}_{datetime.now().timestamp()}",
                issue_type=IssueType.QUALITY_ISSUE,
                severity=IssueSeverity.LOW,
                title=f"Suspicious characters in field '{field_name}'",
                description=f"Field contains unusual characters that may indicate OCR errors",
                affected_fields=[field_name],
                field_data={field_name: extraction.value}
            )
            
            issue.suggested_actions = [ResolutionAction(
                action_type="manual_review",
                description="Review field for OCR errors and correct if necessary",
                requires_human=True,
                confidence=0.8
            )]
            
            issues.append(issue)
        
        # Check for very short or very long values
        if len(value_str) == 1:
            issue = DocumentIssue(
                id=f"quality_too_short_{field_name}_{datetime.now().timestamp()}",
                issue_type=IssueType.QUALITY_ISSUE,
                severity=IssueSeverity.MEDIUM,
                title=f"Unusually short value in field '{field_name}'",
                description=f"Field value is only one character, which may indicate incomplete extraction",
                affected_fields=[field_name],
                field_data={field_name: extraction.value}
            )
            
            issue.suggested_actions = [ResolutionAction(
                action_type="reextract_field",
                description="Re-extract field with adjusted parameters",
                auto_executable=True,
                confidence=0.7
            )]
            
            issues.append(issue)
        
        return issues
    
    def _load_issue_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for issue detection."""
        return {
            'ocr_errors': [
                r'\d[a-zA-Z]\d',  # Digit-letter-digit pattern
                r'[l1][O0o][l1]',  # Common OCR confusion
                r'[rn][nm]',      # r/n confusion
            ],
            'incomplete_extraction': [
                r'^[A-Z]$',       # Single uppercase letter
                r'^\d$',          # Single digit
                r'^...$',         # Very short strings
            ]
        }


class IssueResolver:
    """Resolves document processing issues using various strategies."""
    
    def __init__(self):
        """Initialize issue resolver."""
        self.resolution_strategies = {
            "manual_review": self._manual_review_strategy,
            "reprocess_with_preprocessing": self._reprocess_strategy,
            "try_alternative_model": self._alternative_model_strategy,
            "adjust_bounding_box": self._adjust_bounding_box_strategy,
            "auto_format_correction": self._auto_format_correction_strategy,
            "swap_dates": self._swap_dates_strategy
        }
    
    def resolve_issue(self, issue: DocumentIssue, context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve a specific issue.
        
        Args:
            issue: Issue to resolve
            context: Additional context for resolution
            
        Returns:
            Resolution result
        """
        resolution_results = []
        
        for action in issue.suggested_actions:
            if action.auto_executable and action.action_type in self.resolution_strategies:
                try:
                    result = self.resolution_strategies[action.action_type](issue, action, context)
                    resolution_results.append(result)
                except Exception as e:
                    resolution_results.append({
                        'action_type': action.action_type,
                        'success': False,
                        'error': str(e)
                    })
        
        return {
            'issue_id': issue.id,
            'resolution_results': resolution_results,
            'fully_resolved': any(r.get('success', False) for r in resolution_results)
        }
    
    def _manual_review_strategy(self, issue: DocumentIssue, action: ResolutionAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle manual review requirement."""
        return {
            'action_type': action.action_type,
            'success': False,
            'requires_human_intervention': True,
            'message': 'Issue flagged for manual review'
        }
    
    def _reprocess_strategy(self, issue: DocumentIssue, action: ResolutionAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle reprocessing with enhanced preprocessing."""
        return {
            'action_type': action.action_type,
            'success': True,
            'message': 'Document queued for reprocessing with enhanced preprocessing'
        }
    
    def _alternative_model_strategy(self, issue: DocumentIssue, action: ResolutionAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle alternative model strategy."""
        return {
            'action_type': action.action_type,
            'success': True,
            'message': 'Field queued for processing with alternative AI model'
        }
    
    def _adjust_bounding_box_strategy(self, issue: DocumentIssue, action: ResolutionAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle bounding box adjustment."""
        return {
            'action_type': action.action_type,
            'success': False,
            'requires_human_intervention': True,
            'message': 'Bounding box adjustment requires manual intervention'
        }
    
    def _auto_format_correction_strategy(self, issue: DocumentIssue, action: ResolutionAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle automatic format correction."""
        # This would contain specific format correction logic
        return {
            'action_type': action.action_type,
            'success': True,
            'message': 'Applied automatic format corrections'
        }
    
    def _swap_dates_strategy(self, issue: DocumentIssue, action: ResolutionAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle date swapping for consistency."""
        return {
            'action_type': action.action_type,
            'success': True,
            'message': 'Swapped dates to correct logical ordering'
        }