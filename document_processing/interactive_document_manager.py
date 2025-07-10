"""
Interactive Document Manager

User interface for document validation, field updates, and business rule management.
Provides real-time validation feedback and collaborative document review capabilities.
"""

from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import uuid

from .business_rules_manager import (
    BusinessRulesManager, ValidationSession, RuleModification, 
    RuleModificationType, ValidationContext
)
from .business_rules_engine import RuleViolation, RuleSeverity
from .field_types import FieldExtraction, FormField
from .interactive_workflow import WorkflowEngine, WorkflowTask, TaskStatus, TaskType, TaskPriority


class DocumentStatus(Enum):
    """Status of document validation."""
    PENDING_VALIDATION = "pending_validation"
    IN_REVIEW = "in_review"
    REQUIRES_ATTENTION = "requires_attention"
    APPROVED = "approved"
    REJECTED = "rejected"
    ON_HOLD = "on_hold"


class FieldUpdateAction(Enum):
    """Types of field update actions."""
    MANUAL_CORRECTION = "manual_correction"
    VALUE_OVERRIDE = "value_override"
    FIELD_CLEAR = "field_clear"
    CONFIDENCE_BOOST = "confidence_boost"
    REPROCESS_REQUEST = "reprocess_request"


@dataclass
class FieldUpdate:
    """Represents a field update during validation."""
    field_name: str
    old_value: Any
    new_value: Any
    action_type: FieldUpdateAction
    user_id: str
    timestamp: datetime
    confidence_before: Optional[float] = None
    confidence_after: Optional[float] = None
    validation_notes: str = ""
    auto_applied: bool = False


@dataclass
class DocumentValidationState:
    """Current validation state of a document."""
    document_id: str
    status: DocumentStatus
    current_violations: List[RuleViolation]
    resolved_violations: List[RuleViolation]
    field_updates: List[FieldUpdate]
    validation_sessions: List[str]  # session IDs
    assigned_reviewers: List[str]
    last_updated: datetime
    completion_percentage: float = 0.0
    requires_supervisor_approval: bool = False


@dataclass
class ValidationAction:
    """Action that can be taken during validation."""
    id: str
    title: str
    description: str
    action_type: str
    field_name: Optional[str] = None
    suggested_value: Optional[Any] = None
    confidence: float = 0.0
    requires_approval: bool = False
    auto_executable: bool = False


class InteractiveDocumentManager:
    """Interactive document management system for validation and rule management."""
    
    def __init__(self, rules_manager: BusinessRulesManager, workflow_engine: WorkflowEngine):
        """Initialize interactive document manager."""
        self.rules_manager = rules_manager
        self.workflow_engine = workflow_engine
        
        # Document states and validation data
        self.document_states: Dict[str, DocumentValidationState] = {}
        self.active_sessions: Dict[str, str] = {}  # user_id -> session_id
        
        # Validation action handlers
        self.action_handlers: Dict[str, Callable] = {
            'update_field': self._handle_field_update,
            'approve_field': self._handle_field_approval,
            'reject_field': self._handle_field_rejection,
            'request_reprocess': self._handle_reprocess_request,
            'modify_rule': self._handle_rule_modification,
            'escalate_to_supervisor': self._handle_escalation
        }
        
        # Real-time validation callbacks
        self.validation_callbacks: List[Callable] = []
    
    def start_document_validation(
        self,
        document_id: str,
        document_data: Dict[str, Any],
        user_id: str,
        validation_context: ValidationContext = ValidationContext.USER_VALIDATION
    ) -> Dict[str, Any]:
        """Start interactive validation session for a document."""
        
        # Validate document with business rules
        validation_result = self.rules_manager.validate_document_with_specialized_rules(
            document_data, validation_context
        )
        
        violations = validation_result['violations']
        
        # Create document validation state
        doc_state = DocumentValidationState(
            document_id=document_id,
            status=DocumentStatus.PENDING_VALIDATION if violations else DocumentStatus.APPROVED,
            current_violations=violations,
            resolved_violations=[],
            field_updates=[],
            validation_sessions=[],
            assigned_reviewers=[user_id],
            last_updated=datetime.now()
        )
        
        # Calculate completion percentage
        doc_state.completion_percentage = self._calculate_completion_percentage(doc_state)
        
        # Check if supervisor approval is required
        doc_state.requires_supervisor_approval = any(
            v.severity == RuleSeverity.CRITICAL for v in violations
        )
        
        self.document_states[document_id] = doc_state
        
        # Create validation session if there are violations
        session = None
        if violations:
            session = self.rules_manager.create_validation_session(
                user_id, document_id, violations
            )
            doc_state.validation_sessions.append(session.id)
            self.active_sessions[user_id] = session.id
            doc_state.status = DocumentStatus.IN_REVIEW
        
        # Generate suggested actions
        suggested_actions = self._generate_validation_actions(
            document_data, violations, validation_context
        )
        
        return {
            'document_id': document_id,
            'validation_state': doc_state,
            'session_id': session.id if session else None,
            'violations': violations,
            'suggested_actions': suggested_actions,
            'validation_summary': {
                'total_violations': len(violations),
                'critical_violations': validation_result['critical_violations'],
                'error_violations': validation_result['error_violations'],
                'warning_violations': validation_result['warning_violations'],
                'completion_percentage': doc_state.completion_percentage,
                'requires_supervisor_approval': doc_state.requires_supervisor_approval
            }
        }
    
    def update_document_field(
        self,
        document_id: str,
        field_name: str,
        new_value: Any,
        user_id: str,
        action_type: FieldUpdateAction = FieldUpdateAction.MANUAL_CORRECTION,
        validation_notes: str = ""
    ) -> Dict[str, Any]:
        """Update a document field and re-validate."""
        
        if document_id not in self.document_states:
            return {'success': False, 'error': 'Document not found'}
        
        doc_state = self.document_states[document_id]
        
        # Get current field value
        current_session_id = self.active_sessions.get(user_id)
        if not current_session_id:
            return {'success': False, 'error': 'No active validation session'}
        
        # Create field update record
        field_update = FieldUpdate(
            field_name=field_name,
            old_value=None,  # Will be filled from document data
            new_value=new_value,
            action_type=action_type,
            user_id=user_id,
            timestamp=datetime.now(),
            validation_notes=validation_notes
        )
        
        doc_state.field_updates.append(field_update)
        
        # Update validation session
        self.rules_manager.update_field_in_session(
            current_session_id,
            field_name,
            field_update.old_value,
            new_value,
            validation_notes
        )
        
        # Re-validate document with updated field
        updated_document_data = self._apply_field_updates_to_document(
            document_id, {field_name: new_value}
        )
        
        validation_result = self.rules_manager.validate_document_with_specialized_rules(
            updated_document_data, ValidationContext.USER_VALIDATION
        )
        
        # Update violations
        old_violation_count = len(doc_state.current_violations)
        doc_state.current_violations = validation_result['violations']
        doc_state.last_updated = datetime.now()
        
        # Update completion percentage
        doc_state.completion_percentage = self._calculate_completion_percentage(doc_state)
        
        # Check if document status should change
        if not doc_state.current_violations:
            doc_state.status = DocumentStatus.APPROVED
        elif len(doc_state.current_violations) < old_violation_count:
            doc_state.status = DocumentStatus.IN_REVIEW
        
        # Generate new suggested actions
        suggested_actions = self._generate_validation_actions(
            updated_document_data, doc_state.current_violations, ValidationContext.USER_VALIDATION
        )
        
        # Trigger validation callbacks
        self._trigger_validation_callbacks(document_id, field_name, validation_result)
        
        return {
            'success': True,
            'field_updated': field_name,
            'new_violations': validation_result['violations'],
            'violations_resolved': old_violation_count - len(doc_state.current_violations),
            'completion_percentage': doc_state.completion_percentage,
            'document_status': doc_state.status.value,
            'suggested_actions': suggested_actions,
            'validation_summary': {
                'total_violations': len(doc_state.current_violations),
                'critical_violations': validation_result['critical_violations'],
                'error_violations': validation_result['error_violations'],
                'warning_violations': validation_result['warning_violations']
            }
        }
    
    def propose_rule_modification_interactive(
        self,
        document_id: str,
        rule_id: str,
        modification_type: RuleModificationType,
        new_value: Any,
        reason: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Propose a rule modification during interactive validation."""
        
        if document_id not in self.document_states:
            return {'success': False, 'error': 'Document not found'}
        
        current_session_id = self.active_sessions.get(user_id)
        
        try:
            modification = self.rules_manager.propose_rule_modification(
                user_id=user_id,
                rule_id=rule_id,
                modification_type=modification_type,
                new_value=new_value,
                reason=reason,
                session_id=current_session_id
            )
            
            return {
                'success': True,
                'modification_id': modification.id,
                'modification_type': modification_type.value,
                'status': 'pending_approval',
                'message': 'Rule modification proposed and sent for approval'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_document_validation_interface(
        self,
        document_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Get the complete validation interface data for a document."""
        
        if document_id not in self.document_states:
            return {'error': 'Document not found'}
        
        doc_state = self.document_states[document_id]
        current_session_id = self.active_sessions.get(user_id)
        
        # Get current violations grouped by severity
        violations_by_severity = {
            'critical': [v for v in doc_state.current_violations if v.severity == RuleSeverity.CRITICAL],
            'error': [v for v in doc_state.current_violations if v.severity == RuleSeverity.ERROR],
            'warning': [v for v in doc_state.current_violations if v.severity == RuleSeverity.WARNING],
            'info': [v for v in doc_state.current_violations if v.severity == RuleSeverity.INFO]
        }
        
        # Get field-specific violations
        field_violations = {}
        for violation in doc_state.current_violations:
            for field_name in violation.affected_fields:
                if field_name not in field_violations:
                    field_violations[field_name] = []
                field_violations[field_name].append(violation)
        
        # Get recent field updates
        recent_updates = sorted(
            doc_state.field_updates,
            key=lambda x: x.timestamp,
            reverse=True
        )[:10]
        
        # Get available actions for each violation
        violation_actions = {}
        for violation in doc_state.current_violations:
            violation_actions[violation.rule_id] = self._get_violation_specific_actions(violation)
        
        # Get workflow tasks for this document
        user_tasks = self.workflow_engine.get_user_tasks(user_id)
        document_tasks = [
            task for task in user_tasks
            if task.document_id == document_id and task.status in [TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]
        ]
        
        return {
            'document_id': document_id,
            'validation_state': doc_state,
            'session_id': current_session_id,
            'violations_by_severity': violations_by_severity,
            'field_violations': field_violations,
            'recent_field_updates': [
                {
                    'field_name': update.field_name,
                    'old_value': update.old_value,
                    'new_value': update.new_value,
                    'action_type': update.action_type.value,
                    'timestamp': update.timestamp.isoformat(),
                    'user_id': update.user_id,
                    'notes': update.validation_notes
                }
                for update in recent_updates
            ],
            'violation_actions': violation_actions,
            'document_tasks': [task.to_dict() for task in document_tasks],
            'validation_progress': {
                'completion_percentage': doc_state.completion_percentage,
                'total_violations': len(doc_state.current_violations),
                'resolved_violations': len(doc_state.resolved_violations),
                'fields_updated': len(doc_state.field_updates),
                'requires_supervisor_approval': doc_state.requires_supervisor_approval
            },
            'available_actions': self._get_document_level_actions(doc_state)
        }
    
    def execute_validation_action(
        self,
        document_id: str,
        action_id: str,
        parameters: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
        """Execute a validation action."""
        
        if document_id not in self.document_states:
            return {'success': False, 'error': 'Document not found'}
        
        # Get action handler
        action_type = parameters.get('action_type')
        if action_type not in self.action_handlers:
            return {'success': False, 'error': f'Unknown action type: {action_type}'}
        
        handler = self.action_handlers[action_type]
        
        try:
            result = handler(document_id, parameters, user_id)
            
            # Update document state timestamp
            self.document_states[document_id].last_updated = datetime.now()
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _handle_field_update(
        self,
        document_id: str,
        parameters: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
                 """Handle field update action."""
         
         field_name = parameters.get('field_name')
         new_value = parameters.get('new_value')
         action_type = FieldUpdateAction(parameters.get('field_action_type', 'manual_correction'))
         notes = parameters.get('notes', '')
         
         if not field_name:
             return {'success': False, 'error': 'Field name is required'}
         
         return self.update_document_field(
             document_id, field_name, new_value, user_id, action_type, notes
         )
    
    def _handle_field_approval(
        self,
        document_id: str,
        parameters: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
        """Handle field approval action."""
        
        field_name = parameters.get('field_name')
        doc_state = self.document_states[document_id]
        
        # Mark field as approved (remove related violations)
        remaining_violations = []
        approved_violations = []
        
        for violation in doc_state.current_violations:
            if field_name in violation.affected_fields:
                approved_violations.append(violation)
            else:
                remaining_violations.append(violation)
        
        doc_state.current_violations = remaining_violations
        doc_state.resolved_violations.extend(approved_violations)
        doc_state.completion_percentage = self._calculate_completion_percentage(doc_state)
        
        return {
            'success': True,
            'action': 'field_approved',
            'field_name': field_name,
            'violations_resolved': len(approved_violations),
            'completion_percentage': doc_state.completion_percentage
        }
    
    def _handle_field_rejection(
        self,
        document_id: str,
        parameters: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
        """Handle field rejection action."""
        
        field_name = parameters.get('field_name')
        reason = parameters.get('reason', '')
        
        # Create workflow task for field reprocessing
        task = WorkflowTask(
            id=f"reprocess_{uuid.uuid4().hex[:8]}",
            task_type=TaskType.MANUAL_EXTRACTION,
            title=f"Field Reprocessing Required: {field_name}",
            description=f"Field '{field_name}' rejected and requires reprocessing. Reason: {reason}",
            priority=TaskPriority.HIGH,
            document_id=document_id,
            field_name=field_name,
            context_data={
                'rejection_reason': reason,
                'rejected_by': user_id,
                'rejection_timestamp': datetime.now().isoformat()
            },
            tags=['field_rejection', 'reprocessing_required']
        )
        
        self.workflow_engine.add_task(task)
        
        return {
            'success': True,
            'action': 'field_rejected',
            'field_name': field_name,
            'task_created': task.id,
            'message': 'Field marked for reprocessing'
        }
    
    def _handle_reprocess_request(
        self,
        document_id: str,
        parameters: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
        """Handle document reprocessing request."""
        
        reprocess_type = parameters.get('reprocess_type', 'full_document')
        reason = parameters.get('reason', '')
        
        # Create reprocessing task
        task = WorkflowTask(
            id=f"reprocess_doc_{uuid.uuid4().hex[:8]}",
            task_type=TaskType.DOCUMENT_CLASSIFICATION,
            title=f"Document Reprocessing: {document_id}",
            description=f"Document requires reprocessing ({reprocess_type}). Reason: {reason}",
            priority=TaskPriority.HIGH,
            document_id=document_id,
            context_data={
                'reprocess_type': reprocess_type,
                'reason': reason,
                'requested_by': user_id,
                'request_timestamp': datetime.now().isoformat()
            },
            tags=['document_reprocessing', 'quality_issue']
        )
        
        self.workflow_engine.add_task(task)
        
        # Update document status
        doc_state = self.document_states[document_id]
        doc_state.status = DocumentStatus.ON_HOLD
        
        return {
            'success': True,
            'action': 'reprocess_requested',
            'reprocess_type': reprocess_type,
            'task_created': task.id,
            'document_status': doc_state.status.value
        }
    
    def _handle_rule_modification(
        self,
        document_id: str,
        parameters: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
        """Handle rule modification request."""
        
                 rule_id = parameters.get('rule_id')
         modification_type = RuleModificationType(parameters.get('modification_type'))
         new_value = parameters.get('new_value')
         reason = parameters.get('reason', '')
         
         if not rule_id:
             return {'success': False, 'error': 'Rule ID is required'}
         
         return self.propose_rule_modification_interactive(
             document_id, rule_id, modification_type, new_value, reason, user_id
         )
    
    def _handle_escalation(
        self,
        document_id: str,
        parameters: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
        """Handle escalation to supervisor."""
        
        escalation_reason = parameters.get('reason', '')
        urgency = parameters.get('urgency', 'normal')
        
        # Create escalation task
        priority = TaskPriority.CRITICAL if urgency == 'urgent' else TaskPriority.HIGH
        
        task = WorkflowTask(
            id=f"escalation_{uuid.uuid4().hex[:8]}",
            task_type=TaskType.QUALITY_REVIEW,
            title=f"Supervisor Review Required: {document_id}",
            description=f"Document escalated for supervisor review. Reason: {escalation_reason}",
            priority=priority,
            document_id=document_id,
            context_data={
                'escalation_reason': escalation_reason,
                'escalated_by': user_id,
                'urgency': urgency,
                'escalation_timestamp': datetime.now().isoformat()
            },
            tags=['escalation', 'supervisor_review']
        )
        
        self.workflow_engine.add_task(task)
        
        # Update document status
        doc_state = self.document_states[document_id]
        doc_state.status = DocumentStatus.REQUIRES_ATTENTION
        doc_state.requires_supervisor_approval = True
        
        return {
            'success': True,
            'action': 'escalated_to_supervisor',
            'task_created': task.id,
            'urgency': urgency,
            'document_status': doc_state.status.value
        }
    
    def _generate_validation_actions(
        self,
        document_data: Dict[str, Any],
        violations: List[RuleViolation],
        context: ValidationContext
    ) -> List[ValidationAction]:
        """Generate suggested validation actions."""
        
        actions = []
        
        for violation in violations:
            # Field-specific actions
            for field_name in violation.affected_fields:
                field_value = document_data.get(field_name)
                
                # Suggest field update
                actions.append(ValidationAction(
                    id=f"update_{field_name}_{uuid.uuid4().hex[:8]}",
                    title=f"Update {field_name}",
                    description=f"Manually correct the value for {field_name}",
                    action_type="update_field",
                    field_name=field_name,
                    confidence=0.9,
                    auto_executable=False
                ))
                
                # Suggest field approval if confidence is reasonable
                if isinstance(field_value, FieldExtraction) and field_value.confidence > 0.7:
                    actions.append(ValidationAction(
                        id=f"approve_{field_name}_{uuid.uuid4().hex[:8]}",
                        title=f"Approve {field_name}",
                        description=f"Accept the current value for {field_name}",
                        action_type="approve_field",
                        field_name=field_name,
                        confidence=field_value.confidence,
                        auto_executable=True
                    ))
            
            # Rule-specific actions
            if violation.severity == RuleSeverity.WARNING:
                actions.append(ValidationAction(
                    id=f"modify_rule_{violation.rule_id}_{uuid.uuid4().hex[:8]}",
                    title=f"Adjust Rule: {violation.rule_name}",
                    description="Propose modification to this business rule",
                    action_type="modify_rule",
                    confidence=0.6,
                    requires_approval=True
                ))
        
        # Document-level actions
        if violations:
            actions.append(ValidationAction(
                id=f"escalate_{uuid.uuid4().hex[:8]}",
                title="Escalate to Supervisor",
                description="Request supervisor review for complex issues",
                action_type="escalate_to_supervisor",
                confidence=0.8,
                requires_approval=False
            ))
            
            actions.append(ValidationAction(
                id=f"reprocess_{uuid.uuid4().hex[:8]}",
                title="Request Reprocessing",
                description="Reprocess document with enhanced algorithms",
                action_type="request_reprocess",
                confidence=0.7,
                requires_approval=False
            ))
        
        return actions
    
    def _get_violation_specific_actions(self, violation: RuleViolation) -> List[Dict[str, Any]]:
        """Get actions specific to a violation."""
        
        actions = []
        
        # Always allow manual field updates
        for field_name in violation.affected_fields:
            actions.append({
                'type': 'update_field',
                'title': f'Update {field_name}',
                'description': violation.suggested_action or f'Manually update {field_name}',
                'field_name': field_name,
                'auto_executable': False
            })
        
        # Rule modification for non-critical violations
        if violation.severity != RuleSeverity.CRITICAL:
            actions.append({
                'type': 'modify_rule',
                'title': 'Modify Rule',
                'description': f'Propose changes to rule: {violation.rule_name}',
                'rule_id': violation.rule_id,
                'requires_approval': True
            })
        
        # Approval option for low-severity violations
        if violation.severity in [RuleSeverity.WARNING, RuleSeverity.INFO]:
            actions.append({
                'type': 'approve_violation',
                'title': 'Accept as Exception',
                'description': 'Accept this violation as a business exception',
                'requires_approval': violation.severity == RuleSeverity.WARNING
            })
        
        return actions
    
    def _get_document_level_actions(self, doc_state: DocumentValidationState) -> List[Dict[str, Any]]:
        """Get document-level actions."""
        
        actions = []
        
        if doc_state.current_violations:
            actions.extend([
                {
                    'type': 'bulk_approve_warnings',
                    'title': 'Approve All Warnings',
                    'description': 'Approve all warning-level violations',
                    'enabled': any(v.severity == RuleSeverity.WARNING for v in doc_state.current_violations)
                },
                {
                    'type': 'request_reprocess',
                    'title': 'Reprocess Document',
                    'description': 'Request full document reprocessing'
                },
                {
                    'type': 'escalate_to_supervisor',
                    'title': 'Escalate to Supervisor',
                    'description': 'Request supervisor review'
                }
            ])
        
        # Final approval action
        if not doc_state.current_violations or all(v.severity == RuleSeverity.WARNING for v in doc_state.current_violations):
            actions.append({
                'type': 'final_approval',
                'title': 'Approve Document',
                'description': 'Mark document as approved and complete validation',
                'requires_supervisor': doc_state.requires_supervisor_approval
            })
        
        return actions
    
    def _calculate_completion_percentage(self, doc_state: DocumentValidationState) -> float:
        """Calculate validation completion percentage."""
        
        total_issues = len(doc_state.current_violations) + len(doc_state.resolved_violations)
        if total_issues == 0:
            return 100.0
        
        resolved_issues = len(doc_state.resolved_violations)
        
        # Weight critical issues more heavily
        critical_current = sum(1 for v in doc_state.current_violations if v.severity == RuleSeverity.CRITICAL)
        critical_resolved = sum(1 for v in doc_state.resolved_violations if v.severity == RuleSeverity.CRITICAL)
        
        # Apply weighting: critical issues count as 3, errors as 2, warnings as 1
        weighted_total = (
            critical_current * 3 + critical_resolved * 3 +
            sum(2 if v.severity == RuleSeverity.ERROR else 1 
                for v in doc_state.current_violations if v.severity != RuleSeverity.CRITICAL) +
            sum(2 if v.severity == RuleSeverity.ERROR else 1 
                for v in doc_state.resolved_violations if v.severity != RuleSeverity.CRITICAL)
        )
        
        weighted_resolved = (
            critical_resolved * 3 +
            sum(2 if v.severity == RuleSeverity.ERROR else 1 
                for v in doc_state.resolved_violations if v.severity != RuleSeverity.CRITICAL)
        )
        
        if weighted_total == 0:
            return 100.0
        
        return min(100.0, (weighted_resolved / weighted_total) * 100)
    
    def _apply_field_updates_to_document(
        self,
        document_id: str,
        field_updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply field updates to document data for re-validation."""
        
        # This would integrate with the actual document storage system
        # For now, return the updates as a new document state
        return field_updates
    
    def _trigger_validation_callbacks(
        self,
        document_id: str,
        field_name: str,
        validation_result: Dict[str, Any]
    ) -> None:
        """Trigger registered validation callbacks."""
        
        for callback in self.validation_callbacks:
            try:
                callback(document_id, field_name, validation_result)
            except Exception as e:
                print(f"Validation callback error: {e}")
    
    def register_validation_callback(self, callback: Callable) -> None:
        """Register a callback for validation events."""
        self.validation_callbacks.append(callback)
    
    def get_validation_dashboard_data(self, user_id: str) -> Dict[str, Any]:
        """Get dashboard data for validation overview."""
        
        # Get user's active documents
        user_documents = {
            doc_id: state for doc_id, state in self.document_states.items()
            if user_id in state.assigned_reviewers
        }
        
        # Calculate statistics
        total_documents = len(user_documents)
        pending_validation = len([s for s in user_documents.values() if s.status == DocumentStatus.PENDING_VALIDATION])
        in_review = len([s for s in user_documents.values() if s.status == DocumentStatus.IN_REVIEW])
        approved = len([s for s in user_documents.values() if s.status == DocumentStatus.APPROVED])
        requires_attention = len([s for s in user_documents.values() if s.status == DocumentStatus.REQUIRES_ATTENTION])
        
        # Get active tasks
        user_tasks = self.workflow_engine.get_user_tasks(user_id, [TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS])
        
        # Get recent activity
        recent_activity = []
        for doc_id, state in user_documents.items():
            for update in state.field_updates[-5:]:  # Last 5 updates per document
                recent_activity.append({
                    'document_id': doc_id,
                    'type': 'field_update',
                    'field_name': update.field_name,
                    'timestamp': update.timestamp.isoformat(),
                    'action': update.action_type.value
                })
        
        # Sort by timestamp
        recent_activity.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return {
            'user_id': user_id,
            'summary': {
                'total_documents': total_documents,
                'pending_validation': pending_validation,
                'in_review': in_review,
                'approved': approved,
                'requires_attention': requires_attention,
                'active_tasks': len(user_tasks)
            },
            'documents': [
                {
                    'document_id': doc_id,
                    'status': state.status.value,
                    'completion_percentage': state.completion_percentage,
                    'violations_count': len(state.current_violations),
                    'last_updated': state.last_updated.isoformat(),
                    'requires_supervisor_approval': state.requires_supervisor_approval
                }
                for doc_id, state in user_documents.items()
            ],
            'active_tasks': [task.to_dict() for task in user_tasks],
            'recent_activity': recent_activity[:20],  # Last 20 activities
            'validation_statistics': self.rules_manager.get_validation_statistics()
        }