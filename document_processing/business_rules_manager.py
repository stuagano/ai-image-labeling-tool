"""
Business Rules Manager

Advanced business rules management system for document processing with
interactive validation, rule modification, and violation resolution.
"""

from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import uuid
import re

from .business_rules_engine import (
    BusinessRulesEngine, BusinessRule, RuleType, RuleSeverity, 
    RuleViolation, RuleCondition
)
from .field_types import FieldExtraction, FormField
from .interactive_workflow import WorkflowEngine, WorkflowTask, TaskType, TaskPriority


class ValidationContext(Enum):
    """Context for rule validation."""
    DOCUMENT_PROCESSING = "document_processing"
    USER_VALIDATION = "user_validation"
    BATCH_PROCESSING = "batch_processing"
    QUALITY_REVIEW = "quality_review"


class RuleModificationType(Enum):
    """Types of rule modifications."""
    CONDITION_UPDATE = "condition_update"
    THRESHOLD_ADJUSTMENT = "threshold_adjustment"
    SEVERITY_CHANGE = "severity_change"
    ACTION_UPDATE = "action_update"
    RULE_DEACTIVATION = "rule_deactivation"
    RULE_CREATION = "rule_creation"


@dataclass
class SignatureValidationRule:
    """Specialized rule for signature validation."""
    field_name: str
    required_coverage: float = 0.15  # Minimum signature area coverage
    require_ink_detection: bool = True
    allow_digital_signature: bool = True
    signature_types: List[str] = field(default_factory=lambda: ["handwritten", "digital"])
    validation_message: str = "Valid signature required"
    
    def validate_signature(self, signature_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate signature data against rule criteria."""
        if not signature_data:
            return False, "No signature data found"
        
        # Check coverage
        coverage = signature_data.get("coverage", 0)
        if coverage < self.required_coverage:
            return False, f"Signature coverage {coverage:.2%} below required {self.required_coverage:.2%}"
        
        # Check ink detection if required
        if self.require_ink_detection:
            ink_detected = signature_data.get("ink_detected", False)
            if not ink_detected:
                return False, "No ink detected in signature area"
        
        # Check signature type
        signature_type = signature_data.get("type", "handwritten")
        if signature_type not in self.signature_types:
            return False, f"Signature type '{signature_type}' not allowed"
        
        return True, "Signature validation passed"


@dataclass
class FieldRequirementRule:
    """Rule for field requirement validation."""
    field_name: str
    required: bool = True
    conditional_requirements: List[Dict[str, Any]] = field(default_factory=list)
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    format_pattern: Optional[str] = None
    allowed_values: Optional[List[str]] = None
    validation_message: str = ""
    
    def validate_field(self, field_value: Any, document_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate field against requirement rules."""
        # Check if field is required
        if self.required and (field_value is None or field_value == ""):
            return False, f"Field '{self.field_name}' is required"
        
        # Check conditional requirements
        for condition in self.conditional_requirements:
            if self._evaluate_condition(condition, document_data):
                if field_value is None or field_value == "":
                    return False, f"Field '{self.field_name}' is required when {condition.get('description', 'condition met')}"
        
        # Skip further validation if field is empty and not required
        if field_value is None or field_value == "":
            return True, "Field validation passed"
        
        # Check length constraints
        if self.min_length and len(str(field_value)) < self.min_length:
            return False, f"Field '{self.field_name}' must be at least {self.min_length} characters"
        
        if self.max_length and len(str(field_value)) > self.max_length:
            return False, f"Field '{self.field_name}' must not exceed {self.max_length} characters"
        
        # Check format pattern
        if self.format_pattern and not re.match(self.format_pattern, str(field_value)):
            return False, f"Field '{self.field_name}' format is invalid"
        
        # Check allowed values
        if self.allowed_values and field_value not in self.allowed_values:
            return False, f"Field '{self.field_name}' value not in allowed list"
        
        return True, "Field validation passed"
    
    def _evaluate_condition(self, condition: Dict[str, Any], document_data: Dict[str, Any]) -> bool:
        """Evaluate a conditional requirement."""
        field = condition.get("field")
        operator = condition.get("operator")
        value = condition.get("value")
        
        if not field or field not in document_data:
            return False
        
        field_value = document_data[field]
        if isinstance(field_value, FieldExtraction):
            field_value = field_value.value
        
        if operator == "equals":
            return field_value == value
        elif operator == "not_equals":
            return field_value != value
        elif operator == "contains":
            return str(value).lower() in str(field_value).lower()
        elif operator == "in_list":
            return field_value in value if isinstance(value, list) else False
        
        return False


@dataclass
class RuleModification:
    """Represents a modification to a business rule."""
    id: str
    rule_id: str
    modification_type: RuleModificationType
    user_id: str
    timestamp: datetime
    original_value: Any
    new_value: Any
    reason: str
    approved: bool = False
    approved_by: Optional[str] = None
    approval_timestamp: Optional[datetime] = None


@dataclass
class ValidationSession:
    """User validation session for document review."""
    id: str
    user_id: str
    document_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    violations_reviewed: List[str] = field(default_factory=list)
    fields_updated: List[str] = field(default_factory=list)
    rules_modified: List[str] = field(default_factory=list)
    session_notes: str = ""
    status: str = "active"  # active, completed, abandoned


class BusinessRulesManager:
    """Advanced business rules management with interactive validation."""
    
    def __init__(self, rules_engine: BusinessRulesEngine, workflow_engine: WorkflowEngine):
        """Initialize business rules manager."""
        self.rules_engine = rules_engine
        self.workflow_engine = workflow_engine
        
        # Specialized rule types
        self.signature_rules: Dict[str, SignatureValidationRule] = {}
        self.field_requirement_rules: Dict[str, FieldRequirementRule] = {}
        
        # Rule modifications and approvals
        self.rule_modifications: Dict[str, RuleModification] = {}
        self.validation_sessions: Dict[str, ValidationSession] = {}
        
        # Load predefined rules
        self._load_predefined_rules()
    
    def _load_predefined_rules(self):
        """Load predefined business rules for common scenarios."""
        
        # Signature validation rules
        contract_signature = SignatureValidationRule(
            field_name="signature",
            required_coverage=0.15,
            require_ink_detection=True,
            allow_digital_signature=True,
            validation_message="Contract requires valid signature"
        )
        
        legal_doc_signature = SignatureValidationRule(
            field_name="client_signature",
            required_coverage=0.20,
            require_ink_detection=True,
            allow_digital_signature=False,
            signature_types=["handwritten"],
            validation_message="Legal document requires handwritten signature"
        )
        
        self.signature_rules["contract_signature"] = contract_signature
        self.signature_rules["legal_document_signature"] = legal_doc_signature
        
        # Field requirement rules
        contract_amount_rule = FieldRequirementRule(
            field_name="contract_amount",
            required=True,
            min_length=1,
            format_pattern=r'^\d+(\.\d{2})?$',
            validation_message="Contract amount is required and must be a valid monetary value"
        )
        
        client_name_rule = FieldRequirementRule(
            field_name="client_name",
            required=True,
            min_length=2,
            max_length=100,
            validation_message="Client name is required"
        )
        
        effective_date_rule = FieldRequirementRule(
            field_name="effective_date",
            required=False,
            conditional_requirements=[
                {
                    "field": "document_type",
                    "operator": "equals",
                    "value": "contract",
                    "description": "document is a contract"
                }
            ],
            format_pattern=r'^\d{4}-\d{2}-\d{2}$',
            validation_message="Effective date required for contracts in YYYY-MM-DD format"
        )
        
        self.field_requirement_rules["contract_amount"] = contract_amount_rule
        self.field_requirement_rules["client_name"] = client_name_rule
        self.field_requirement_rules["effective_date"] = effective_date_rule
        
        # Create corresponding business rules
        self._create_business_rules_from_specialized()
    
    def _create_business_rules_from_specialized(self):
        """Create business rules from specialized rule types."""
        
        # Create signature validation business rules
        for rule_id, sig_rule in self.signature_rules.items():
            business_rule = BusinessRule(
                id=f"signature_{rule_id}",
                name=f"Signature Validation: {sig_rule.field_name}",
                description=f"Validates signature requirements for {sig_rule.field_name}",
                rule_type=RuleType.VALIDATION,
                severity=RuleSeverity.CRITICAL,
                conditions=[
                    {"field": sig_rule.field_name, "operator": "is_not_empty", "value": None}
                ],
                validation_message=sig_rule.validation_message,
                suggested_action="Obtain valid signature or mark document as incomplete",
                tags=["signature", "validation", "compliance"]
            )
            self.rules_engine.add_rule(business_rule)
        
        # Create field requirement business rules
        for rule_id, field_rule in self.field_requirement_rules.items():
            business_rule = BusinessRule(
                id=f"field_req_{rule_id}",
                name=f"Field Requirement: {field_rule.field_name}",
                description=f"Validates requirements for {field_rule.field_name}",
                rule_type=RuleType.VALIDATION,
                severity=RuleSeverity.ERROR if field_rule.required else RuleSeverity.WARNING,
                conditions=[
                    {"field": field_rule.field_name, "operator": "is_empty", "value": None}
                ] if field_rule.required else [],
                validation_message=field_rule.validation_message or f"Field {field_rule.field_name} validation failed",
                suggested_action="Review and correct field value",
                tags=["field_validation", "requirements"]
            )
            self.rules_engine.add_rule(business_rule)
    
    def validate_document_with_specialized_rules(
        self,
        document_data: Dict[str, Any],
        context: ValidationContext = ValidationContext.DOCUMENT_PROCESSING
    ) -> Dict[str, Any]:
        """Validate document using both standard and specialized rules."""
        
        # Standard business rules validation
        standard_violations = self.rules_engine.evaluate_rules(document_data)
        
        # Specialized signature validation
        signature_violations = []
        for rule_id, sig_rule in self.signature_rules.items():
            if sig_rule.field_name in document_data:
                is_valid, message = sig_rule.validate_signature(
                    document_data[sig_rule.field_name]
                )
                if not is_valid:
                    violation = RuleViolation(
                        rule_id=f"signature_{rule_id}",
                        rule_name=f"Signature Validation: {sig_rule.field_name}",
                        severity=RuleSeverity.CRITICAL,
                        message=message,
                        affected_fields=[sig_rule.field_name],
                        suggested_action="Obtain valid signature",
                        auto_fixable=False
                    )
                    signature_violations.append(violation)
        
        # Specialized field requirement validation
        field_violations = []
        for rule_id, field_rule in self.field_requirement_rules.items():
            field_value = document_data.get(field_rule.field_name)
            if isinstance(field_value, FieldExtraction):
                field_value = field_value.value
            
            is_valid, message = field_rule.validate_field(field_value, document_data)
            if not is_valid:
                violation = RuleViolation(
                    rule_id=f"field_req_{rule_id}",
                    rule_name=f"Field Requirement: {field_rule.field_name}",
                    severity=RuleSeverity.ERROR if field_rule.required else RuleSeverity.WARNING,
                    message=message,
                    affected_fields=[field_rule.field_name],
                    suggested_action="Review and correct field value",
                    auto_fixable=False
                )
                field_violations.append(violation)
        
        all_violations = standard_violations + signature_violations + field_violations
        
        return {
            'violations': all_violations,
            'total_violations': len(all_violations),
            'critical_violations': len([v for v in all_violations if v.severity == RuleSeverity.CRITICAL]),
            'error_violations': len([v for v in all_violations if v.severity == RuleSeverity.ERROR]),
            'warning_violations': len([v for v in all_violations if v.severity == RuleSeverity.WARNING]),
            'validation_context': context.value,
            'timestamp': datetime.now()
        }
    
    def create_validation_session(
        self,
        user_id: str,
        document_id: str,
        violations: List[RuleViolation]
    ) -> ValidationSession:
        """Create an interactive validation session for a user."""
        
        session = ValidationSession(
            id=f"session_{uuid.uuid4().hex[:8]}",
            user_id=user_id,
            document_id=document_id,
            start_time=datetime.now()
        )
        
        self.validation_sessions[session.id] = session
        
        # Create workflow tasks for each violation
        for violation in violations:
            task = WorkflowTask(
                id=f"validation_{uuid.uuid4().hex[:8]}",
                task_type=TaskType.BUSINESS_RULE_REVIEW,
                title=f"Rule Violation: {violation.rule_name}",
                description=violation.message,
                priority=self._get_task_priority_from_severity(violation.severity),
                document_id=document_id,
                field_name=violation.affected_fields[0] if violation.affected_fields else None,
                context_data={
                    'validation_session_id': session.id,
                    'rule_id': violation.rule_id,
                    'violation_data': {
                        'severity': violation.severity.value,
                        'suggested_action': violation.suggested_action,
                        'auto_fixable': violation.auto_fixable
                    }
                },
                tags=['rule_violation', 'validation_required']
            )
            
            self.workflow_engine.add_task(task)
            self.workflow_engine.assign_task(task.id, user_id)
        
        return session
    
    def update_field_in_session(
        self,
        session_id: str,
        field_name: str,
        old_value: Any,
        new_value: Any,
        user_notes: str = ""
    ) -> bool:
        """Update a field value during validation session."""
        
        if session_id not in self.validation_sessions:
            return False
        
        session = self.validation_sessions[session_id]
        
        # Record field update
        session.fields_updated.append(field_name)
        
        # Log the change for audit trail
        change_record = {
            'session_id': session_id,
            'field_name': field_name,
            'old_value': old_value,
            'new_value': new_value,
            'user_notes': user_notes,
            'timestamp': datetime.now()
        }
        
        # TODO: Integrate with audit trail system
        
        return True
    
    def propose_rule_modification(
        self,
        user_id: str,
        rule_id: str,
        modification_type: RuleModificationType,
        new_value: Any,
        reason: str,
        session_id: Optional[str] = None
    ) -> RuleModification:
        """Propose a modification to a business rule."""
        
        if rule_id not in self.rules_engine.rules:
            raise ValueError(f"Rule {rule_id} not found")
        
        rule = self.rules_engine.rules[rule_id]
        
        # Get original value based on modification type
        original_value = None
        if modification_type == RuleModificationType.CONDITION_UPDATE:
            original_value = rule.conditions
        elif modification_type == RuleModificationType.THRESHOLD_ADJUSTMENT:
            # For specialized rules, get threshold values
            original_value = self._get_rule_thresholds(rule_id)
        elif modification_type == RuleModificationType.SEVERITY_CHANGE:
            original_value = rule.severity.value
        elif modification_type == RuleModificationType.ACTION_UPDATE:
            original_value = rule.suggested_action
        
        modification = RuleModification(
            id=f"mod_{uuid.uuid4().hex[:8]}",
            rule_id=rule_id,
            modification_type=modification_type,
            user_id=user_id,
            timestamp=datetime.now(),
            original_value=original_value,
            new_value=new_value,
            reason=reason
        )
        
        self.rule_modifications[modification.id] = modification
        
        # Update validation session if provided
        if session_id and session_id in self.validation_sessions:
            self.validation_sessions[session_id].rules_modified.append(rule_id)
        
        # Create approval task for supervisors
        approval_task = WorkflowTask(
            id=f"approval_{uuid.uuid4().hex[:8]}",
            task_type=TaskType.BUSINESS_RULE_REVIEW,
            title=f"Rule Modification Approval: {rule.name}",
            description=f"Review proposed modification: {modification_type.value}",
            priority=TaskPriority.HIGH,
            context_data={
                'modification_id': modification.id,
                'rule_id': rule_id,
                'modification_type': modification_type.value,
                'reason': reason,
                'proposed_by': user_id
            },
            tags=['rule_modification', 'approval_required']
        )
        
        self.workflow_engine.add_task(approval_task)
        
        return modification
    
    def approve_rule_modification(
        self,
        modification_id: str,
        approver_id: str,
        approved: bool,
        approval_notes: str = ""
    ) -> bool:
        """Approve or reject a rule modification."""
        
        if modification_id not in self.rule_modifications:
            return False
        
        modification = self.rule_modifications[modification_id]
        modification.approved = approved
        modification.approved_by = approver_id
        modification.approval_timestamp = datetime.now()
        
        if approved:
            # Apply the modification
            self._apply_rule_modification(modification)
        
        return True
    
    def _apply_rule_modification(self, modification: RuleModification) -> bool:
        """Apply an approved rule modification."""
        
        rule_id = modification.rule_id
        
        if rule_id not in self.rules_engine.rules:
            return False
        
        rule = self.rules_engine.rules[rule_id]
        
        try:
            if modification.modification_type == RuleModificationType.CONDITION_UPDATE:
                rule.conditions = modification.new_value
            elif modification.modification_type == RuleModificationType.SEVERITY_CHANGE:
                rule.severity = RuleSeverity(modification.new_value)
            elif modification.modification_type == RuleModificationType.ACTION_UPDATE:
                rule.suggested_action = modification.new_value
            elif modification.modification_type == RuleModificationType.THRESHOLD_ADJUSTMENT:
                self._apply_threshold_modification(rule_id, modification.new_value)
            elif modification.modification_type == RuleModificationType.RULE_DEACTIVATION:
                rule.active = False
            
            return True
        except Exception as e:
            print(f"Failed to apply rule modification: {e}")
            return False
    
    def _apply_threshold_modification(self, rule_id: str, new_thresholds: Dict[str, Any]) -> bool:
        """Apply threshold modifications to specialized rules."""
        
        # Check if it's a signature rule
        for sig_rule_id, sig_rule in self.signature_rules.items():
            if f"signature_{sig_rule_id}" == rule_id:
                if 'required_coverage' in new_thresholds:
                    sig_rule.required_coverage = new_thresholds['required_coverage']
                if 'require_ink_detection' in new_thresholds:
                    sig_rule.require_ink_detection = new_thresholds['require_ink_detection']
                return True
        
        # Check if it's a field requirement rule
        for field_rule_id, field_rule in self.field_requirement_rules.items():
            if f"field_req_{field_rule_id}" == rule_id:
                if 'min_length' in new_thresholds:
                    field_rule.min_length = new_thresholds['min_length']
                if 'max_length' in new_thresholds:
                    field_rule.max_length = new_thresholds['max_length']
                if 'required' in new_thresholds:
                    field_rule.required = new_thresholds['required']
                return True
        
        return False
    
    def _get_rule_thresholds(self, rule_id: str) -> Dict[str, Any]:
        """Get current thresholds for a rule."""
        
        # Check signature rules
        for sig_rule_id, sig_rule in self.signature_rules.items():
            if f"signature_{sig_rule_id}" == rule_id:
                return {
                    'required_coverage': sig_rule.required_coverage,
                    'require_ink_detection': sig_rule.require_ink_detection
                }
        
        # Check field requirement rules
        for field_rule_id, field_rule in self.field_requirement_rules.items():
            if f"field_req_{field_rule_id}" == rule_id:
                return {
                    'min_length': field_rule.min_length,
                    'max_length': field_rule.max_length,
                    'required': field_rule.required
                }
        
        return {}
    
    def _get_task_priority_from_severity(self, severity: RuleSeverity) -> TaskPriority:
        """Convert rule severity to task priority."""
        mapping = {
            RuleSeverity.CRITICAL: TaskPriority.CRITICAL,
            RuleSeverity.ERROR: TaskPriority.HIGH,
            RuleSeverity.WARNING: TaskPriority.MEDIUM,
            RuleSeverity.INFO: TaskPriority.LOW
        }
        return mapping.get(severity, TaskPriority.MEDIUM)
    
    def complete_validation_session(
        self,
        session_id: str,
        session_notes: str = "",
        status: str = "completed"
    ) -> bool:
        """Complete a validation session."""
        
        if session_id not in self.validation_sessions:
            return False
        
        session = self.validation_sessions[session_id]
        session.end_time = datetime.now()
        session.session_notes = session_notes
        session.status = status
        
        return True
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get statistics about validation sessions and rule modifications."""
        
        completed_sessions = [
            s for s in self.validation_sessions.values()
            if s.status == "completed"
        ]
        
        active_sessions = [
            s for s in self.validation_sessions.values()
            if s.status == "active"
        ]
        
        pending_modifications = [
            m for m in self.rule_modifications.values()
            if not m.approved and m.approved_by is None
        ]
        
        approved_modifications = [
            m for m in self.rule_modifications.values()
            if m.approved
        ]
        
        return {
            'total_sessions': len(self.validation_sessions),
            'completed_sessions': len(completed_sessions),
            'active_sessions': len(active_sessions),
            'average_session_duration': self._calculate_average_session_duration(completed_sessions),
            'total_modifications_proposed': len(self.rule_modifications),
            'pending_modifications': len(pending_modifications),
            'approved_modifications': len(approved_modifications),
            'modification_approval_rate': len(approved_modifications) / len(self.rule_modifications) if self.rule_modifications else 0,
            'most_modified_rules': self._get_most_modified_rules(),
            'signature_rules_count': len(self.signature_rules),
            'field_requirement_rules_count': len(self.field_requirement_rules)
        }
    
    def _calculate_average_session_duration(self, sessions: List[ValidationSession]) -> float:
        """Calculate average session duration in minutes."""
        if not sessions:
            return 0
        
        total_duration = 0
        valid_sessions = 0
        
        for session in sessions:
            if session.end_time:
                duration = (session.end_time - session.start_time).total_seconds() / 60
                total_duration += duration
                valid_sessions += 1
        
        return total_duration / valid_sessions if valid_sessions > 0 else 0
    
    def _get_most_modified_rules(self) -> List[Tuple[str, int]]:
        """Get rules that have been modified most frequently."""
        rule_modification_counts = {}
        
        for modification in self.rule_modifications.values():
            rule_id = modification.rule_id
            rule_modification_counts[rule_id] = rule_modification_counts.get(rule_id, 0) + 1
        
        # Sort by modification count
        sorted_rules = sorted(
            rule_modification_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_rules[:10]  # Top 10 most modified rules
    
    def export_validation_report(
        self,
        session_id: Optional[str] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """Export validation report for analysis."""
        
        sessions_to_include = []
        
        if session_id:
            if session_id in self.validation_sessions:
                sessions_to_include = [self.validation_sessions[session_id]]
        else:
            sessions_to_include = list(self.validation_sessions.values())
        
        # Filter by date range if provided
        if date_range:
            start_date, end_date = date_range
            sessions_to_include = [
                s for s in sessions_to_include
                if start_date <= s.start_time <= end_date
            ]
        
        report = {
            'report_generated': datetime.now().isoformat(),
            'session_count': len(sessions_to_include),
            'sessions': [],
            'rule_modifications': [],
            'summary_statistics': {}
        }
        
        # Include session details
        for session in sessions_to_include:
            session_data = {
                'session_id': session.id,
                'user_id': session.user_id,
                'document_id': session.document_id,
                'start_time': session.start_time.isoformat(),
                'end_time': session.end_time.isoformat() if session.end_time else None,
                'duration_minutes': (session.end_time - session.start_time).total_seconds() / 60 if session.end_time else None,
                'violations_reviewed': len(session.violations_reviewed),
                'fields_updated': len(session.fields_updated),
                'rules_modified': len(session.rules_modified),
                'status': session.status,
                'notes': session.session_notes
            }
            report['sessions'].append(session_data)
        
        # Include related rule modifications
        session_ids = {s.id for s in sessions_to_include}
        for modification in self.rule_modifications.values():
            # Check if modification is related to any of the sessions
            modification_data = {
                'modification_id': modification.id,
                'rule_id': modification.rule_id,
                'modification_type': modification.modification_type.value,
                'user_id': modification.user_id,
                'timestamp': modification.timestamp.isoformat(),
                'reason': modification.reason,
                'approved': modification.approved,
                'approved_by': modification.approved_by,
                'approval_timestamp': modification.approval_timestamp.isoformat() if modification.approval_timestamp else None
            }
            report['rule_modifications'].append(modification_data)
        
        # Add summary statistics
        report['summary_statistics'] = self.get_validation_statistics()
        
        return report