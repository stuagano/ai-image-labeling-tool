"""
Business Rules Engine

Comprehensive business rules engine for capturing and enforcing complex
business logic beyond basic field validation.
"""

from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import re
import json

from .field_types import FieldExtraction, FormField


class RuleType(Enum):
    """Types of business rules."""
    VALIDATION = "validation"
    CONDITIONAL = "conditional"
    CALCULATION = "calculation"
    WORKFLOW = "workflow"
    COMPLIANCE = "compliance"
    BUSINESS_LOGIC = "business_logic"


class RuleSeverity(Enum):
    """Severity levels for rule violations."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    CRITICAL = "critical"


class RuleCondition(Enum):
    """Condition operators for rules."""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    REGEX_MATCH = "regex_match"
    IS_EMPTY = "is_empty"
    IS_NOT_EMPTY = "is_not_empty"
    IN_LIST = "in_list"
    NOT_IN_LIST = "not_in_list"


@dataclass
class RuleViolation:
    """Represents a business rule violation."""
    rule_id: str
    rule_name: str
    severity: RuleSeverity
    message: str
    affected_fields: List[str]
    suggested_action: Optional[str] = None
    auto_fixable: bool = False
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BusinessRule:
    """Defines a business rule with conditions and actions."""
    id: str
    name: str
    description: str
    rule_type: RuleType
    severity: RuleSeverity = RuleSeverity.ERROR
    
    # Conditions
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    logical_operator: str = "AND"  # AND, OR
    
    # Actions
    validation_message: str = ""
    suggested_action: str = ""
    auto_fix_action: Optional[str] = None
    
    # Metadata
    priority: int = 1  # Higher number = higher priority
    active: bool = True
    tags: List[str] = field(default_factory=list)
    created_date: datetime = field(default_factory=datetime.now)
    
    def evaluate_conditions(self, field_data: Dict[str, Any]) -> bool:
        """Evaluate rule conditions against field data."""
        if not self.conditions:
            return True
        
        results = []
        for condition in self.conditions:
            result = self._evaluate_single_condition(condition, field_data)
            results.append(result)
        
        if self.logical_operator == "OR":
            return any(results)
        else:  # AND
            return all(results)
    
    def _evaluate_single_condition(self, condition: Dict[str, Any], field_data: Dict[str, Any]) -> bool:
        """Evaluate a single condition."""
        field_name = condition.get("field")
        operator = condition.get("operator")
        expected_value = condition.get("value")
        
        if not operator:
            return False
        
        if field_name not in field_data:
            return operator == RuleCondition.IS_EMPTY.value
        
        actual_value = field_data[field_name]
        
        # Handle FieldExtraction objects
        if isinstance(actual_value, FieldExtraction):
            actual_value = actual_value.value
        
        return self._apply_condition_operator(operator, actual_value, expected_value)
    
    def _apply_condition_operator(self, operator: str, actual: Any, expected: Any) -> bool:
        """Apply condition operator."""
        try:
            if operator == RuleCondition.EQUALS.value:
                return actual == expected
            elif operator == RuleCondition.NOT_EQUALS.value:
                return actual != expected
            elif operator == RuleCondition.GREATER_THAN.value:
                return float(actual) > float(expected)
            elif operator == RuleCondition.LESS_THAN.value:
                return float(actual) < float(expected)
            elif operator == RuleCondition.GREATER_EQUAL.value:
                return float(actual) >= float(expected)
            elif operator == RuleCondition.LESS_EQUAL.value:
                return float(actual) <= float(expected)
            elif operator == RuleCondition.CONTAINS.value:
                return str(expected).lower() in str(actual).lower()
            elif operator == RuleCondition.NOT_CONTAINS.value:
                return str(expected).lower() not in str(actual).lower()
            elif operator == RuleCondition.REGEX_MATCH.value:
                return bool(re.match(str(expected), str(actual)))
            elif operator == RuleCondition.IS_EMPTY.value:
                return not actual or actual == ""
            elif operator == RuleCondition.IS_NOT_EMPTY.value:
                return actual and actual != ""
            elif operator == RuleCondition.IN_LIST.value:
                return actual in expected if isinstance(expected, list) else False
            elif operator == RuleCondition.NOT_IN_LIST.value:
                return actual not in expected if isinstance(expected, list) else True
            else:
                return False
        except (ValueError, TypeError):
            return False


class BusinessRulesEngine:
    """Main business rules engine for document processing."""
    
    def __init__(self):
        """Initialize the business rules engine."""
        self.rules: Dict[str, BusinessRule] = {}
        self.rule_categories: Dict[str, List[str]] = {}
        self.execution_log: List[Dict[str, Any]] = []
        
        # Load built-in rules
        self._load_builtin_rules()
    
    def add_rule(self, rule: BusinessRule) -> None:
        """Add a business rule to the engine."""
        self.rules[rule.id] = rule
        
        # Add to categories
        for tag in rule.tags:
            if tag not in self.rule_categories:
                self.rule_categories[tag] = []
            if rule.id not in self.rule_categories[tag]:
                self.rule_categories[tag].append(rule.id)
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a business rule."""
        if rule_id in self.rules:
            rule = self.rules[rule_id]
            del self.rules[rule_id]
            
            # Remove from categories
            for tag in rule.tags:
                if tag in self.rule_categories and rule_id in self.rule_categories[tag]:
                    self.rule_categories[tag].remove(rule_id)
            
            return True
        return False
    
    def evaluate_rules(
        self,
        field_data: Dict[str, Any],
        rule_categories: Optional[List[str]] = None
    ) -> List[RuleViolation]:
        """Evaluate business rules against field data."""
        violations = []
        
        # Determine which rules to evaluate
        rules_to_evaluate = []
        if rule_categories:
            for category in rule_categories:
                if category in self.rule_categories:
                    rules_to_evaluate.extend(self.rule_categories[category])
        else:
            rules_to_evaluate = list(self.rules.keys())
        
        # Remove duplicates and sort by priority
        rules_to_evaluate = list(set(rules_to_evaluate))
        rules_to_evaluate.sort(key=lambda r_id: self.rules[r_id].priority, reverse=True)
        
        # Evaluate each rule
        for rule_id in rules_to_evaluate:
            rule = self.rules[rule_id]
            
            if not rule.active:
                continue
            
            try:
                # Check if rule conditions are met
                if not rule.evaluate_conditions(field_data):
                    violation = RuleViolation(
                        rule_id=rule.id,
                        rule_name=rule.name,
                        severity=rule.severity,
                        message=rule.validation_message or f"Business rule '{rule.name}' violated",
                        affected_fields=self._extract_affected_fields(rule, field_data),
                        suggested_action=rule.suggested_action,
                        auto_fixable=rule.auto_fix_action is not None
                    )
                    violations.append(violation)
                
                # Log execution
                self.execution_log.append({
                    'rule_id': rule.id,
                    'timestamp': datetime.now(),
                    'result': 'violated' if violations else 'passed',
                    'field_count': len(field_data)
                })
                
            except Exception as e:
                # Log rule execution error
                self.execution_log.append({
                    'rule_id': rule.id,
                    'timestamp': datetime.now(),
                    'result': 'error',
                    'error': str(e)
                })
        
        return violations
    
    def _extract_affected_fields(self, rule: BusinessRule, field_data: Dict[str, Any]) -> List[str]:
        """Extract field names affected by a rule."""
        affected_fields = []
        for condition in rule.conditions:
            field_name = condition.get("field")
            if field_name and field_name in field_data:
                affected_fields.append(field_name)
        return affected_fields
    
    def auto_fix_violations(
        self,
        violations: List[RuleViolation],
        field_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Attempt to automatically fix rule violations."""
        fixed_data = field_data.copy()
        fix_log = []
        
        for violation in violations:
            if violation.auto_fixable and violation.rule_id in self.rules:
                rule = self.rules[violation.rule_id]
                
                try:
                    if rule.auto_fix_action:
                        # Apply auto-fix logic (this would be expanded based on specific rules)
                        fix_result = self._apply_auto_fix(rule, fixed_data)
                        if fix_result['success']:
                            fixed_data.update(fix_result['data'])
                            fix_log.append({
                                'rule_id': rule.id,
                                'action': rule.auto_fix_action,
                                'status': 'success'
                            })
                        else:
                            fix_log.append({
                                'rule_id': rule.id,
                                'action': rule.auto_fix_action,
                                'status': 'failed',
                                'error': fix_result.get('error', 'Unknown error')
                            })
                except Exception as e:
                    fix_log.append({
                        'rule_id': rule.id,
                        'status': 'error',
                        'error': str(e)
                    })
        
        return {
            'fixed_data': fixed_data,
            'fix_log': fix_log,
            'fixes_applied': len([log for log in fix_log if log.get('status') == 'success'])
        }
    
    def _apply_auto_fix(self, rule: BusinessRule, field_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply auto-fix action for a rule."""
        # This is a placeholder for auto-fix logic
        # In practice, this would contain specific fix implementations
        return {'success': False, 'error': 'Auto-fix not implemented for this rule'}
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get statistics about rule execution."""
        total_rules = len(self.rules)
        active_rules = sum(1 for rule in self.rules.values() if rule.active)
        
        # Analyze execution log
        recent_executions = [
            log for log in self.execution_log
            if log['timestamp'] > datetime.now() - timedelta(hours=24)
        ]
        
        rule_performance = {}
        for log in recent_executions:
            rule_id = log['rule_id']
            if rule_id not in rule_performance:
                rule_performance[rule_id] = {'executions': 0, 'violations': 0, 'errors': 0}
            
            rule_performance[rule_id]['executions'] += 1
            if log['result'] == 'violated':
                rule_performance[rule_id]['violations'] += 1
            elif log['result'] == 'error':
                rule_performance[rule_id]['errors'] += 1
        
        return {
            'total_rules': total_rules,
            'active_rules': active_rules,
            'rule_categories': len(self.rule_categories),
            'recent_executions': len(recent_executions),
            'rule_performance': rule_performance
        }
    
    def _load_builtin_rules(self) -> None:
        """Load built-in business rules."""
        
        # Example: Date consistency rule
        date_consistency_rule = BusinessRule(
            id="date_consistency_001",
            name="Date Logical Order",
            description="Ensure dates follow logical order (start < end)",
            rule_type=RuleType.BUSINESS_LOGIC,
            severity=RuleSeverity.ERROR,
            conditions=[
                {"field": "start_date", "operator": "is_not_empty", "value": None},
                {"field": "end_date", "operator": "is_not_empty", "value": None}
            ],
            validation_message="End date must be after start date",
            suggested_action="Check date fields and ensure proper chronological order",
            tags=["dates", "consistency"]
        )
        
        # Example: Signature requirement rule
        signature_required_rule = BusinessRule(
            id="signature_required_001",
            name="Signature Required for Contracts",
            description="Legal documents must have valid signatures",
            rule_type=RuleType.COMPLIANCE,
            severity=RuleSeverity.CRITICAL,
            conditions=[
                {"field": "document_type", "operator": "in_list", "value": ["contract", "agreement", "legal"]},
                {"field": "signature", "operator": "is_empty", "value": None}
            ],
            validation_message="Signature is required for legal documents",
            suggested_action="Obtain signature or mark document as incomplete",
            tags=["legal", "signatures", "compliance"]
        )
        
        # Example: Amount validation rule
        amount_validation_rule = BusinessRule(
            id="amount_validation_001",
            name="Amount Field Validation",
            description="Monetary amounts must be positive and reasonable",
            rule_type=RuleType.VALIDATION,
            severity=RuleSeverity.WARNING,
            conditions=[
                {"field": "amount", "operator": "less_than", "value": 0}
            ],
            logical_operator="OR",
            validation_message="Amount cannot be negative",
            suggested_action="Verify amount field and correct if necessary",
            tags=["financial", "validation"]
        )
        
        # Add built-in rules
        for rule in [date_consistency_rule, signature_required_rule, amount_validation_rule]:
            self.add_rule(rule)
    
    def export_rules(self, file_path: str) -> None:
        """Export rules to JSON file."""
        rules_data = {}
        for rule_id, rule in self.rules.items():
            rules_data[rule_id] = {
                'id': rule.id,
                'name': rule.name,
                'description': rule.description,
                'rule_type': rule.rule_type.value,
                'severity': rule.severity.value,
                'conditions': rule.conditions,
                'logical_operator': rule.logical_operator,
                'validation_message': rule.validation_message,
                'suggested_action': rule.suggested_action,
                'auto_fix_action': rule.auto_fix_action,
                'priority': rule.priority,
                'active': rule.active,
                'tags': rule.tags,
                'created_date': rule.created_date.isoformat()
            }
        
        with open(file_path, 'w') as f:
            json.dump(rules_data, f, indent=2)
    
    def import_rules(self, file_path: str) -> int:
        """Import rules from JSON file."""
        with open(file_path, 'r') as f:
            rules_data = json.load(f)
        
        imported_count = 0
        for rule_data in rules_data.values():
            try:
                rule = BusinessRule(
                    id=rule_data['id'],
                    name=rule_data['name'],
                    description=rule_data['description'],
                    rule_type=RuleType(rule_data['rule_type']),
                    severity=RuleSeverity(rule_data['severity']),
                    conditions=rule_data['conditions'],
                    logical_operator=rule_data['logical_operator'],
                    validation_message=rule_data['validation_message'],
                    suggested_action=rule_data['suggested_action'],
                    auto_fix_action=rule_data.get('auto_fix_action'),
                    priority=rule_data['priority'],
                    active=rule_data['active'],
                    tags=rule_data['tags'],
                    created_date=datetime.fromisoformat(rule_data['created_date'])
                )
                self.add_rule(rule)
                imported_count += 1
            except Exception as e:
                print(f"Failed to import rule {rule_data.get('id', 'unknown')}: {e}")
        
        return imported_count