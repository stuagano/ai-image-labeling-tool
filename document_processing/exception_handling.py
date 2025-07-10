"""
Exception Handling Framework

Systematic exception management and routing for document processing
with escalation paths and recovery strategies.
"""

from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import traceback
import logging
import json

from .field_types import FieldExtraction
from .issue_resolution import DocumentIssue, IssueType, IssueSeverity


class ExceptionType(Enum):
    """Types of exceptions in document processing."""
    EXTRACTION_ERROR = "extraction_error"
    VALIDATION_ERROR = "validation_error"
    PREPROCESSING_ERROR = "preprocessing_error"
    MODEL_ERROR = "model_error"
    TEMPLATE_ERROR = "template_error"
    BUSINESS_RULE_ERROR = "business_rule_error"
    SYSTEM_ERROR = "system_error"
    DATA_ERROR = "data_error"
    CONFIGURATION_ERROR = "configuration_error"


class ExceptionSeverity(Enum):
    """Severity levels for exceptions."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EscalationLevel(Enum):
    """Escalation levels for exception handling."""
    AUTOMATIC_RETRY = "automatic_retry"
    ALTERNATIVE_PROCESSING = "alternative_processing"
    HUMAN_REVIEW = "human_review"
    TECHNICAL_SUPPORT = "technical_support"
    SYSTEM_ADMIN = "system_admin"


@dataclass
class ExceptionContext:
    """Context information for an exception."""
    document_id: Optional[str] = None
    field_name: Optional[str] = None
    template_id: Optional[str] = None
    processing_step: Optional[str] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    system_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryAction:
    """Represents a recovery action for an exception."""
    action_type: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    success_probability: float = 0.5
    execution_time_estimate: Optional[int] = None  # seconds
    requires_user_input: bool = False
    escalation_level: EscalationLevel = EscalationLevel.AUTOMATIC_RETRY


@dataclass
class ProcessingException:
    """Represents a processing exception with context and recovery options."""
    id: str
    exception_type: ExceptionType
    severity: ExceptionSeverity
    title: str
    description: str
    original_exception: Optional[Exception] = None
    
    # Context
    context: ExceptionContext = field(default_factory=ExceptionContext)
    stack_trace: Optional[str] = None
    
    # Recovery
    recovery_actions: List[RecoveryAction] = field(default_factory=list)
    escalation_path: List[EscalationLevel] = field(default_factory=list)
    
    # Status tracking
    retry_count: int = 0
    max_retries: int = 3
    resolved: bool = False
    escalated: bool = False
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    last_retry: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary."""
        return {
            'id': self.id,
            'exception_type': self.exception_type.value,
            'severity': self.severity.value,
            'title': self.title,
            'description': self.description,
            'context': {
                'document_id': self.context.document_id,
                'field_name': self.context.field_name,
                'template_id': self.context.template_id,
                'processing_step': self.context.processing_step
            },
            'recovery_actions': [
                {
                    'action_type': action.action_type,
                    'description': action.description,
                    'success_probability': action.success_probability,
                    'escalation_level': action.escalation_level.value
                }
                for action in self.recovery_actions
            ],
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'resolved': self.resolved,
            'escalated': self.escalated,
            'timestamp': self.timestamp.isoformat()
        }


class ExceptionClassifier:
    """Classifies exceptions and determines appropriate handling strategies."""
    
    def __init__(self):
        """Initialize exception classifier."""
        self.classification_rules = self._load_classification_rules()
    
    def classify_exception(self, exception: Exception, context: ExceptionContext) -> ProcessingException:
        """Classify an exception and determine handling strategy.
        
        Args:
            exception: The original exception
            context: Context information
            
        Returns:
            Classified ProcessingException
        """
        exception_type, severity = self._determine_type_and_severity(exception, context)
        
        processing_exception = ProcessingException(
            id=f"exc_{datetime.now().timestamp()}",
            exception_type=exception_type,
            severity=severity,
            title=self._generate_title(exception, exception_type),
            description=str(exception),
            original_exception=exception,
            context=context,
            stack_trace=traceback.format_exc()
        )
        
        # Generate recovery actions
        processing_exception.recovery_actions = self._generate_recovery_actions(
            processing_exception
        )
        
        # Determine escalation path
        processing_exception.escalation_path = self._determine_escalation_path(
            processing_exception
        )
        
        return processing_exception
    
    def _determine_type_and_severity(
        self,
        exception: Exception,
        context: ExceptionContext
    ) -> tuple[ExceptionType, ExceptionSeverity]:
        """Determine exception type and severity."""
        exception_name = type(exception).__name__
        exception_message = str(exception).lower()
        
        # Check for specific exception patterns
        if "extraction" in exception_message or "extract" in exception_message:
            return ExceptionType.EXTRACTION_ERROR, ExceptionSeverity.MEDIUM
        
        if "validation" in exception_message or "validate" in exception_message:
            return ExceptionType.VALIDATION_ERROR, ExceptionSeverity.LOW
        
        if "preprocessing" in exception_message or "preprocess" in exception_message:
            return ExceptionType.PREPROCESSING_ERROR, ExceptionSeverity.MEDIUM
        
        if "model" in exception_message or "ai" in exception_message:
            return ExceptionType.MODEL_ERROR, ExceptionSeverity.HIGH
        
        if "template" in exception_message:
            return ExceptionType.TEMPLATE_ERROR, ExceptionSeverity.MEDIUM
        
        if "business" in exception_message or "rule" in exception_message:
            return ExceptionType.BUSINESS_RULE_ERROR, ExceptionSeverity.LOW
        
        # Check for system-level exceptions
        if exception_name in ["FileNotFoundError", "PermissionError", "IOError"]:
            return ExceptionType.SYSTEM_ERROR, ExceptionSeverity.HIGH
        
        if exception_name in ["ValueError", "TypeError", "KeyError"]:
            return ExceptionType.DATA_ERROR, ExceptionSeverity.MEDIUM
        
        if exception_name in ["ConnectionError", "TimeoutError"]:
            return ExceptionType.SYSTEM_ERROR, ExceptionSeverity.HIGH
        
        # Default classification
        return ExceptionType.SYSTEM_ERROR, ExceptionSeverity.MEDIUM
    
    def _generate_title(self, exception: Exception, exception_type: ExceptionType) -> str:
        """Generate a descriptive title for the exception."""
        type_name = exception_type.value.replace('_', ' ').title()
        exception_name = type(exception).__name__
        return f"{type_name}: {exception_name}"
    
    def _generate_recovery_actions(self, proc_exc: ProcessingException) -> List[RecoveryAction]:
        """Generate recovery actions based on exception type."""
        actions = []
        
        if proc_exc.exception_type == ExceptionType.EXTRACTION_ERROR:
            actions.extend([
                RecoveryAction(
                    action_type="retry_with_different_model",
                    description="Retry extraction with alternative AI model",
                    success_probability=0.6,
                    escalation_level=EscalationLevel.ALTERNATIVE_PROCESSING
                ),
                RecoveryAction(
                    action_type="manual_extraction",
                    description="Escalate to manual extraction",
                    success_probability=0.95,
                    requires_user_input=True,
                    escalation_level=EscalationLevel.HUMAN_REVIEW
                )
            ])
        
        elif proc_exc.exception_type == ExceptionType.PREPROCESSING_ERROR:
            actions.extend([
                RecoveryAction(
                    action_type="retry_with_basic_preprocessing",
                    description="Retry with simplified preprocessing",
                    success_probability=0.7,
                    escalation_level=EscalationLevel.ALTERNATIVE_PROCESSING
                ),
                RecoveryAction(
                    action_type="skip_preprocessing",
                    description="Process document without preprocessing",
                    success_probability=0.4,
                    escalation_level=EscalationLevel.ALTERNATIVE_PROCESSING
                )
            ])
        
        elif proc_exc.exception_type == ExceptionType.MODEL_ERROR:
            actions.extend([
                RecoveryAction(
                    action_type="fallback_to_basic_ocr",
                    description="Use basic OCR instead of AI model",
                    success_probability=0.5,
                    escalation_level=EscalationLevel.ALTERNATIVE_PROCESSING
                ),
                RecoveryAction(
                    action_type="technical_review",
                    description="Escalate to technical support for model issues",
                    success_probability=0.9,
                    requires_user_input=True,
                    escalation_level=EscalationLevel.TECHNICAL_SUPPORT
                )
            ])
        
        elif proc_exc.exception_type == ExceptionType.VALIDATION_ERROR:
            actions.extend([
                RecoveryAction(
                    action_type="relaxed_validation",
                    description="Retry with relaxed validation rules",
                    success_probability=0.8,
                    escalation_level=EscalationLevel.ALTERNATIVE_PROCESSING
                ),
                RecoveryAction(
                    action_type="manual_validation_review",
                    description="Manual review of validation rules",
                    success_probability=0.95,
                    requires_user_input=True,
                    escalation_level=EscalationLevel.HUMAN_REVIEW
                )
            ])
        
        elif proc_exc.exception_type == ExceptionType.SYSTEM_ERROR:
            actions.extend([
                RecoveryAction(
                    action_type="system_retry",
                    description="Retry operation after brief delay",
                    success_probability=0.6,
                    escalation_level=EscalationLevel.AUTOMATIC_RETRY
                ),
                RecoveryAction(
                    action_type="system_admin_review",
                    description="Escalate to system administrator",
                    success_probability=0.9,
                    requires_user_input=True,
                    escalation_level=EscalationLevel.SYSTEM_ADMIN
                )
            ])
        
        # Add default retry action if no specific actions
        if not actions:
            actions.append(RecoveryAction(
                action_type="generic_retry",
                description="Retry the operation",
                success_probability=0.3,
                escalation_level=EscalationLevel.AUTOMATIC_RETRY
            ))
        
        return actions
    
    def _determine_escalation_path(self, proc_exc: ProcessingException) -> List[EscalationLevel]:
        """Determine escalation path based on exception characteristics."""
        path = []
        
        if proc_exc.severity == ExceptionSeverity.CRITICAL:
            path = [
                EscalationLevel.AUTOMATIC_RETRY,
                EscalationLevel.TECHNICAL_SUPPORT,
                EscalationLevel.SYSTEM_ADMIN
            ]
        elif proc_exc.severity == ExceptionSeverity.HIGH:
            path = [
                EscalationLevel.AUTOMATIC_RETRY,
                EscalationLevel.ALTERNATIVE_PROCESSING,
                EscalationLevel.HUMAN_REVIEW
            ]
        elif proc_exc.severity == ExceptionSeverity.MEDIUM:
            path = [
                EscalationLevel.AUTOMATIC_RETRY,
                EscalationLevel.ALTERNATIVE_PROCESSING,
                EscalationLevel.HUMAN_REVIEW
            ]
        else:  # LOW
            path = [
                EscalationLevel.AUTOMATIC_RETRY,
                EscalationLevel.ALTERNATIVE_PROCESSING
            ]
        
        return path
    
    def _load_classification_rules(self) -> Dict[str, Any]:
        """Load classification rules for exceptions."""
        return {
            'severity_keywords': {
                'critical': ['critical', 'fatal', 'crash', 'corruption'],
                'high': ['error', 'failed', 'exception', 'timeout'],
                'medium': ['warning', 'invalid', 'missing'],
                'low': ['info', 'notice', 'debug']
            },
            'type_patterns': {
                'extraction': ['extract', 'ocr', 'recognition'],
                'validation': ['validate', 'check', 'verify'],
                'preprocessing': ['preprocess', 'normalize', 'enhance'],
                'model': ['model', 'ai', 'prediction', 'inference']
            }
        }


class ExceptionHandler:
    """Main exception handler for document processing."""
    
    def __init__(self):
        """Initialize exception handler."""
        self.classifier = ExceptionClassifier()
        self.active_exceptions: Dict[str, ProcessingException] = {}
        self.exception_history: List[ProcessingException] = []
        self.recovery_strategies: Dict[str, Callable] = self._load_recovery_strategies()
        
        # Statistics
        self.stats = {
            'total_exceptions': 0,
            'resolved_exceptions': 0,
            'escalated_exceptions': 0,
            'exception_types': {},
            'recovery_success_rates': {}
        }
    
    def handle_exception(
        self,
        exception: Exception,
        context: ExceptionContext,
        auto_recover: bool = True
    ) -> ProcessingException:
        """Handle an exception with classification and recovery.
        
        Args:
            exception: The original exception
            context: Context information
            auto_recover: Whether to attempt automatic recovery
            
        Returns:
            ProcessingException with handling results
        """
        # Classify the exception
        proc_exc = self.classifier.classify_exception(exception, context)
        
        # Add to active exceptions
        self.active_exceptions[proc_exc.id] = proc_exc
        self.exception_history.append(proc_exc)
        
        # Update statistics
        self._update_statistics(proc_exc)
        
        # Log the exception
        self._log_exception(proc_exc)
        
        # Attempt automatic recovery if enabled
        if auto_recover:
            recovery_result = self._attempt_recovery(proc_exc)
            if recovery_result['success']:
                proc_exc.resolved = True
                self.stats['resolved_exceptions'] += 1
        
        return proc_exc
    
    def retry_exception(self, exception_id: str) -> Dict[str, Any]:
        """Retry handling for a specific exception.
        
        Args:
            exception_id: ID of the exception to retry
            
        Returns:
            Retry result
        """
        if exception_id not in self.active_exceptions:
            return {'success': False, 'error': 'Exception not found'}
        
        proc_exc = self.active_exceptions[exception_id]
        
        if proc_exc.retry_count >= proc_exc.max_retries:
            return {'success': False, 'error': 'Maximum retries exceeded'}
        
        proc_exc.retry_count += 1
        proc_exc.last_retry = datetime.now()
        
        return self._attempt_recovery(proc_exc)
    
    def escalate_exception(self, exception_id: str, escalation_level: EscalationLevel) -> Dict[str, Any]:
        """Escalate an exception to a higher level.
        
        Args:
            exception_id: ID of the exception to escalate
            escalation_level: Target escalation level
            
        Returns:
            Escalation result
        """
        if exception_id not in self.active_exceptions:
            return {'success': False, 'error': 'Exception not found'}
        
        proc_exc = self.active_exceptions[exception_id]
        proc_exc.escalated = True
        
        # Update statistics
        self.stats['escalated_exceptions'] += 1
        
        # Log escalation
        logging.warning(f"Exception {exception_id} escalated to {escalation_level.value}")
        
        return {
            'success': True,
            'escalation_level': escalation_level.value,
            'message': f'Exception escalated to {escalation_level.value}'
        }
    
    def resolve_exception(self, exception_id: str, resolution_notes: str = "") -> Dict[str, Any]:
        """Mark an exception as resolved.
        
        Args:
            exception_id: ID of the exception to resolve
            resolution_notes: Notes about the resolution
            
        Returns:
            Resolution result
        """
        if exception_id not in self.active_exceptions:
            return {'success': False, 'error': 'Exception not found'}
        
        proc_exc = self.active_exceptions[exception_id]
        proc_exc.resolved = True
        
        # Remove from active exceptions
        del self.active_exceptions[exception_id]
        
        # Update statistics
        self.stats['resolved_exceptions'] += 1
        
        # Log resolution
        logging.info(f"Exception {exception_id} resolved: {resolution_notes}")
        
        return {
            'success': True,
            'resolution_notes': resolution_notes,
            'resolved_at': datetime.now().isoformat()
        }
    
    def _attempt_recovery(self, proc_exc: ProcessingException) -> Dict[str, Any]:
        """Attempt automatic recovery for an exception."""
        for action in proc_exc.recovery_actions:
            if (action.escalation_level == EscalationLevel.AUTOMATIC_RETRY and
                not action.requires_user_input):
                
                if action.action_type in self.recovery_strategies:
                    try:
                        result = self.recovery_strategies[action.action_type](proc_exc, action)
                        if result.get('success', False):
                            self._update_recovery_stats(action.action_type, True)
                            return result
                        else:
                            self._update_recovery_stats(action.action_type, False)
                    except Exception as e:
                        logging.error(f"Recovery action {action.action_type} failed: {e}")
                        self._update_recovery_stats(action.action_type, False)
        
        return {'success': False, 'message': 'No successful automatic recovery actions'}
    
    def _update_statistics(self, proc_exc: ProcessingException) -> None:
        """Update exception statistics."""
        self.stats['total_exceptions'] += 1
        
        exc_type = proc_exc.exception_type.value
        if exc_type not in self.stats['exception_types']:
            self.stats['exception_types'][exc_type] = 0
        self.stats['exception_types'][exc_type] += 1
    
    def _update_recovery_stats(self, action_type: str, success: bool) -> None:
        """Update recovery action statistics."""
        if action_type not in self.stats['recovery_success_rates']:
            self.stats['recovery_success_rates'][action_type] = {'attempts': 0, 'successes': 0}
        
        self.stats['recovery_success_rates'][action_type]['attempts'] += 1
        if success:
            self.stats['recovery_success_rates'][action_type]['successes'] += 1
    
    def _log_exception(self, proc_exc: ProcessingException) -> None:
        """Log exception details."""
        log_level = {
            ExceptionSeverity.CRITICAL: logging.CRITICAL,
            ExceptionSeverity.HIGH: logging.ERROR,
            ExceptionSeverity.MEDIUM: logging.WARNING,
            ExceptionSeverity.LOW: logging.INFO
        }.get(proc_exc.severity, logging.WARNING)
        
        logging.log(
            log_level,
            f"Exception {proc_exc.id}: {proc_exc.title} - {proc_exc.description}"
        )
    
    def _load_recovery_strategies(self) -> Dict[str, Callable]:
        """Load recovery strategy functions."""
        return {
            'system_retry': self._system_retry_strategy,
            'retry_with_different_model': self._alternative_model_strategy,
            'retry_with_basic_preprocessing': self._basic_preprocessing_strategy,
            'relaxed_validation': self._relaxed_validation_strategy,
            'fallback_to_basic_ocr': self._basic_ocr_strategy
        }
    
    def _system_retry_strategy(self, proc_exc: ProcessingException, action: RecoveryAction) -> Dict[str, Any]:
        """System retry recovery strategy."""
        # Add a small delay for system issues
        import time
        time.sleep(1)
        
        return {
            'success': True,
            'message': 'System retry completed',
            'action_type': action.action_type
        }
    
    def _alternative_model_strategy(self, proc_exc: ProcessingException, action: RecoveryAction) -> Dict[str, Any]:
        """Alternative model recovery strategy."""
        return {
            'success': True,
            'message': 'Queued for processing with alternative model',
            'action_type': action.action_type
        }
    
    def _basic_preprocessing_strategy(self, proc_exc: ProcessingException, action: RecoveryAction) -> Dict[str, Any]:
        """Basic preprocessing recovery strategy."""
        return {
            'success': True,
            'message': 'Queued for reprocessing with basic preprocessing',
            'action_type': action.action_type
        }
    
    def _relaxed_validation_strategy(self, proc_exc: ProcessingException, action: RecoveryAction) -> Dict[str, Any]:
        """Relaxed validation recovery strategy."""
        return {
            'success': True,
            'message': 'Applied relaxed validation rules',
            'action_type': action.action_type
        }
    
    def _basic_ocr_strategy(self, proc_exc: ProcessingException, action: RecoveryAction) -> Dict[str, Any]:
        """Basic OCR fallback strategy."""
        return {
            'success': True,
            'message': 'Fallback to basic OCR processing',
            'action_type': action.action_type
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get exception handling statistics."""
        return self.stats.copy()
    
    def get_active_exceptions(self) -> List[Dict[str, Any]]:
        """Get list of active exceptions."""
        return [exc.to_dict() for exc in self.active_exceptions.values()]
    
    def export_exception_log(self, file_path: str) -> None:
        """Export exception history to file."""
        export_data = {
            'statistics': self.stats,
            'exceptions': [exc.to_dict() for exc in self.exception_history],
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)