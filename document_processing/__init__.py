"""
Document Processing Module

Enhanced document processing system with robustness features
including business rules, issue resolution, exception handling,
workflow management, analytics, and audit trail.
"""

# Core field types - no heavy dependencies
from .field_types import (
    FieldType, ValidationLevel, BoundingBox, FieldExtraction, BaseField,
    TextField, NumberField, DateField, EmailField, PhoneField,
    SignatureField, CheckboxField, RadioButtonField, DropdownField, TableField
)

# Business rules engine
from .business_rules_engine import (
    BusinessRulesEngine, BusinessRule, RuleType, RuleSeverity, RuleViolation
)

# Issue detection and resolution
from .issue_resolution import (
    IssueDetector, IssueResolver, DocumentIssue, IssueType, IssueSeverity, ResolutionAction
)

# Exception handling
from .exception_handling import (
    ExceptionHandler, ExceptionContext, ProcessingException, 
    ExceptionType, ExceptionSeverity, EscalationLevel
)

# Interactive workflow
from .interactive_workflow import (
    WorkflowEngine, WorkflowUser, WorkflowTask, TaskType, TaskPriority, 
    TaskStatus, UserRole
)

# Analytics and learning
from .analytics_learning import (
    LearningEngine, FieldPerformanceAnalyzer, ErrorPatternDetector,
    LearningInsight, LearningCategory
)

# Audit trail
from .audit_trail import (
    AuditLogger, AuditEvent, AuditContext, AuditEventType, AuditLevel
)

# Template management - basic features only
try:
    from .template_manager import DocumentTemplate, DocumentTemplateManager
except ImportError:
    # Skip if dependencies not available
    pass

# Validators
try:
    from .validators import ValidationReport, DocumentValidator, QualityAssurance
except ImportError:
    # Skip if dependencies not available
    pass

__version__ = "2.0.0"
__all__ = [
    # Field types
    "FieldType", "ValidationLevel", "BoundingBox", "FieldExtraction", "BaseField",
    "TextField", "NumberField", "DateField", "EmailField", "PhoneField",
    "SignatureField", "CheckboxField", "RadioButtonField", "DropdownField", "TableField",
    
    # Business rules
    "BusinessRulesEngine", "BusinessRule", "RuleType", "RuleSeverity", "RuleViolation",
    
    # Issue resolution
    "IssueDetector", "IssueResolver", "DocumentIssue", "IssueType", "IssueSeverity", "ResolutionAction",
    
    # Exception handling
    "ExceptionHandler", "ExceptionContext", "ProcessingException", 
    "ExceptionType", "ExceptionSeverity", "EscalationLevel",
    
    # Workflow
    "WorkflowEngine", "WorkflowUser", "WorkflowTask", "TaskType", "TaskPriority", 
    "TaskStatus", "UserRole",
    
    # Analytics
    "LearningEngine", "FieldPerformanceAnalyzer", "ErrorPatternDetector",
    "LearningInsight", "LearningCategory",
    
    # Audit
    "AuditLogger", "AuditEvent", "AuditContext", "AuditEventType", "AuditLevel"
]

def get_version():
    """Get the version of the document processing system."""
    return __version__

def get_system_info():
    """Get information about the robustness enhancement system."""
    return {
        "version": __version__,
        "features": [
            "Business Rules Engine",
            "Issue Detection & Resolution", 
            "Exception Handling Framework",
            "Interactive Workflow System",
            "Analytics & Learning Engine",
            "Complete Audit Trail",
            "Strong Field Typing",
            "Template Management",
            "Quality Assurance"
        ],
        "components": len(__all__)
    }

def initialize_system():
    """Initialize the robustness enhancement system."""
    print("🚀 Initializing Document Processing Robustness System...")
    print(f"   Version: {__version__}")
    print(f"   Components: {len(__all__)}")
    print("✅ System initialized successfully!")
    return True