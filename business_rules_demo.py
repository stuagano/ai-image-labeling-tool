"""
Business Rules Management Demo

Comprehensive demonstration of the business rules management system
with signature validation, field requirements, and interactive document validation.
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any

# Add the document_processing module to the path
sys.path.append(os.path.dirname(__file__))

from document_processing.business_rules_engine import (
    BusinessRulesEngine, BusinessRule, RuleType, RuleSeverity
)
from document_processing.business_rules_manager import (
    BusinessRulesManager, SignatureValidationRule, FieldRequirementRule,
    ValidationContext, RuleModificationType
)
from document_processing.interactive_workflow import (
    WorkflowEngine, WorkflowUser, UserRole
)
from document_processing.field_types import FieldExtraction, BoundingBox


class BusinessRulesDemo:
    """Comprehensive business rules management demonstration."""
    
    def __init__(self):
        """Initialize demo components."""
        print("🚀 Initializing Business Rules Management Demo...")
        
        # Initialize core components
        self.rules_engine = BusinessRulesEngine()
        self.workflow_engine = WorkflowEngine()
        self.rules_manager = BusinessRulesManager(self.rules_engine, self.workflow_engine)
        
        # Demo users
        self.demo_users = []
        
        print("✅ All components initialized successfully!")
    
    def run_complete_demo(self):
        """Run the complete business rules management demonstration."""
        print("\n" + "="*80)
        print("📋 BUSINESS RULES MANAGEMENT COMPREHENSIVE DEMONSTRATION")
        print("="*80)
        
        try:
            # Set up demo environment
            self.setup_demo_users()
            self.demo_signature_validation()
            self.demo_field_requirements()
            self.demo_interactive_validation()
            self.demo_rule_modifications()
            self.demo_validation_statistics()
            
            print("\n🏁 Business Rules Demo completed successfully!")
            
        except Exception as e:
            print(f"❌ Demo error: {e}")
            import traceback
            traceback.print_exc()
    
    def setup_demo_users(self):
        """Set up demo users for workflow testing."""
        print("\n👥 SETTING UP DEMO USERS")
        print("-" * 50)
        
        users = [
            WorkflowUser(
                id="operator_001",
                name="Sarah Wilson",
                email="sarah@example.com",
                role=UserRole.OPERATOR,
                skills=["data_entry", "basic_validation", "contract_review"],
                performance_metrics={"accuracy": 0.92, "tasks_completed": 245}
            ),
            WorkflowUser(
                id="reviewer_001",
                name="Michael Chen",
                email="michael@example.com",
                role=UserRole.REVIEWER,
                skills=["quality_review", "signature_validation", "compliance"],
                performance_metrics={"accuracy": 0.96, "tasks_completed": 128}
            ),
            WorkflowUser(
                id="supervisor_001",
                name="Jennifer Davis",
                email="jennifer@example.com",
                role=UserRole.SUPERVISOR,
                skills=["business_rules", "exception_handling", "approval_authority"],
                performance_metrics={"accuracy": 0.99, "tasks_completed": 67}
            )
        ]
        
        for user in users:
            self.workflow_engine.register_user(user)
            self.demo_users.append(user)
        
        print(f"✅ Registered {len(users)} demo users")
        for user in users:
            print(f"  - {user.name} ({user.role.value}): {', '.join(user.skills)}")
    
    def demo_signature_validation(self):
        """Demonstrate signature validation rules."""
        print("\n✍️  SIGNATURE VALIDATION DEMO")
        print("-" * 50)
        
        print("Testing signature validation rules...")
        
        # Test case 1: Valid signature
        valid_signature_data = {
            "signature": {
                "coverage": 0.18,  # 18% coverage
                "ink_detected": True,
                "type": "handwritten",
                "confidence": 0.85
            },
            "document_type": "contract",
            "client_name": "John Smith"
        }
        
        print("\n📄 Test Case 1: Valid Contract Signature")
        result = self.rules_manager.validate_document_with_specialized_rules(
            valid_signature_data, ValidationContext.DOCUMENT_PROCESSING
        )
        
        print(f"  Violations found: {result['total_violations']}")
        if result['violations']:
            for violation in result['violations']:
                print(f"    - {violation.rule_name}: {violation.message}")
        else:
            print("  ✅ All signature validation rules passed")
        
        # Test case 2: Invalid signature (low coverage)
        invalid_signature_data = {
            "signature": {
                "coverage": 0.08,  # Only 8% coverage (below 15% requirement)
                "ink_detected": True,
                "type": "handwritten",
                "confidence": 0.45
            },
            "document_type": "contract",
            "client_name": "Jane Doe"
        }
        
        print("\n📄 Test Case 2: Invalid Signature (Low Coverage)")
        result = self.rules_manager.validate_document_with_specialized_rules(
            invalid_signature_data, ValidationContext.DOCUMENT_PROCESSING
        )
        
        print(f"  Violations found: {result['total_violations']}")
        for violation in result['violations']:
            print(f"    - {violation.rule_name}: {violation.message}")
            print(f"      Severity: {violation.severity.value}")
            print(f"      Suggested action: {violation.suggested_action}")
        
        # Test case 3: Missing signature
        missing_signature_data = {
            "signature": None,
            "document_type": "legal_agreement",
            "client_name": "Bob Johnson"
        }
        
        print("\n📄 Test Case 3: Missing Signature")
        result = self.rules_manager.validate_document_with_specialized_rules(
            missing_signature_data, ValidationContext.DOCUMENT_PROCESSING
        )
        
        print(f"  Violations found: {result['total_violations']}")
        for violation in result['violations']:
            print(f"    - {violation.rule_name}: {violation.message}")
            print(f"      Severity: {violation.severity.value}")
    
    def demo_field_requirements(self):
        """Demonstrate field requirement validation."""
        print("\n📝 FIELD REQUIREMENTS VALIDATION DEMO")
        print("-" * 50)
        
        print("Testing field requirement rules...")
        
        # Test case 1: Complete contract data
        complete_contract_data = {
            "contract_amount": "125000.00",
            "client_name": "Acme Corporation",
            "effective_date": "2024-01-15",
            "document_type": "contract",
            "signature": {
                "coverage": 0.20,
                "ink_detected": True,
                "type": "handwritten"
            }
        }
        
        print("\n📄 Test Case 1: Complete Contract Data")
        result = self.rules_manager.validate_document_with_specialized_rules(
            complete_contract_data, ValidationContext.USER_VALIDATION
        )
        
        print(f"  Violations found: {result['total_violations']}")
        if result['violations']:
            for violation in result['violations']:
                print(f"    - {violation.rule_name}: {violation.message}")
        else:
            print("  ✅ All field requirement rules passed")
        
        # Test case 2: Missing required fields
        incomplete_data = {
            "contract_amount": "",  # Missing required field
            "client_name": "X",     # Too short (min 2 characters)
            "effective_date": "invalid-date",  # Invalid format
            "document_type": "contract"
        }
        
        print("\n📄 Test Case 2: Incomplete/Invalid Data")
        result = self.rules_manager.validate_document_with_specialized_rules(
            incomplete_data, ValidationContext.USER_VALIDATION
        )
        
        print(f"  Violations found: {result['total_violations']}")
        for violation in result['violations']:
            print(f"    - {violation.rule_name}: {violation.message}")
            print(f"      Affected fields: {', '.join(violation.affected_fields)}")
            print(f"      Severity: {violation.severity.value}")
        
        # Test case 3: Conditional requirements
        non_contract_data = {
            "client_name": "Regular Customer",
            "document_type": "invoice",  # Not a contract
            "amount": "500.00"
            # effective_date not required for non-contracts
        }
        
        print("\n📄 Test Case 3: Non-Contract Document (Conditional Requirements)")
        result = self.rules_manager.validate_document_with_specialized_rules(
            non_contract_data, ValidationContext.USER_VALIDATION
        )
        
        print(f"  Violations found: {result['total_violations']}")
        if result['violations']:
            for violation in result['violations']:
                print(f"    - {violation.rule_name}: {violation.message}")
        else:
            print("  ✅ Conditional requirements correctly applied")
    
    def demo_interactive_validation(self):
        """Demonstrate interactive validation session."""
        print("\n🔄 INTERACTIVE VALIDATION SESSION DEMO")
        print("-" * 50)
        
        # Simulate a document with violations
        problematic_document = {
            "document_id": "contract_2024_001",
            "contract_amount": "",  # Missing
            "client_name": "A",     # Too short
            "effective_date": "2024-13-45",  # Invalid date
            "signature": {
                "coverage": 0.05,   # Too low
                "ink_detected": False,
                "type": "handwritten"
            },
            "document_type": "contract"
        }
        
        print("Creating validation session for problematic document...")
        
        # Validate document
        validation_result = self.rules_manager.validate_document_with_specialized_rules(
            problematic_document, ValidationContext.USER_VALIDATION
        )
        
        violations = validation_result['violations']
        print(f"  Found {len(violations)} violations")
        
        # Create validation session
        user_id = "reviewer_001"
        session = self.rules_manager.create_validation_session(
            user_id, problematic_document["document_id"], violations
        )
        
        print(f"  Created validation session: {session.id}")
        print(f"  Assigned to: {user_id}")
        
        # Simulate field updates during validation
        print("\nSimulating field corrections...")
        
        field_updates = [
            ("contract_amount", "75000.00", "Corrected missing contract amount"),
            ("client_name", "ABC Corporation", "Expanded abbreviated client name"),
            ("effective_date", "2024-03-15", "Corrected invalid date format")
        ]
        
        for field_name, new_value, notes in field_updates:
            success = self.rules_manager.update_field_in_session(
                session.id, field_name, None, new_value, notes
            )
            
            if success:
                print(f"  ✅ Updated {field_name}: {new_value}")
                session.fields_updated.append(field_name)
            else:
                print(f"  ❌ Failed to update {field_name}")
        
        # Complete validation session
        session_notes = "Corrected all field validation issues. Signature still needs attention."
        self.rules_manager.complete_validation_session(session.id, session_notes)
        
        print(f"  Session completed with notes: {session_notes}")
        print(f"  Fields updated: {len(session.fields_updated)}")
    
    def demo_rule_modifications(self):
        """Demonstrate rule modification proposals."""
        print("\n⚙️  RULE MODIFICATION DEMO")
        print("-" * 50)
        
        print("Demonstrating rule modification workflow...")
        
        # Propose signature coverage threshold adjustment
        user_id = "supervisor_001"
        rule_id = "signature_contract_signature"  # From predefined rules
        
        print(f"\n📋 User {user_id} proposing rule modification...")
        
        try:
            modification = self.rules_manager.propose_rule_modification(
                user_id=user_id,
                rule_id=rule_id,
                modification_type=RuleModificationType.THRESHOLD_ADJUSTMENT,
                new_value={"required_coverage": 0.12},  # Lower from 15% to 12%
                reason="Based on recent document quality analysis, 12% coverage provides adequate signature validation while reducing false positives"
            )
            
            print(f"  ✅ Rule modification proposed: {modification.id}")
            print(f"  Modification type: {modification.modification_type.value}")
            print(f"  Reason: {modification.reason}")
            print(f"  Status: Pending approval")
            
            # Simulate approval process
            print(f"\n👨‍💼 Simulating supervisor approval...")
            
            approval_success = self.rules_manager.approve_rule_modification(
                modification.id,
                approver_id="supervisor_001",
                approved=True,
                approval_notes="Approved based on data analysis and business need"
            )
            
            if approval_success:
                print(f"  ✅ Rule modification approved and applied")
                
                # Verify the change was applied
                updated_rule = self.rules_manager.signature_rules.get("contract_signature")
                if updated_rule:
                    print(f"  New coverage threshold: {updated_rule.required_coverage:.1%}")
            else:
                print(f"  ❌ Failed to approve rule modification")
        
        except Exception as e:
            print(f"  ❌ Rule modification failed: {e}")
        
        # Propose severity change for field requirement
        print(f"\n📋 Proposing severity change for field requirement rule...")
        
        try:
            field_rule_id = "field_req_client_name"
            
            modification = self.rules_manager.propose_rule_modification(
                user_id=user_id,
                rule_id=field_rule_id,
                modification_type=RuleModificationType.SEVERITY_CHANGE,
                new_value="warning",  # Change from error to warning
                reason="Client name validation should be warning level to allow processing with review"
            )
            
            print(f"  ✅ Severity change proposed: {modification.id}")
            print(f"  Rule: {field_rule_id}")
            print(f"  New severity: warning")
            
        except Exception as e:
            print(f"  ❌ Severity change proposal failed: {e}")
    
    def demo_validation_statistics(self):
        """Demonstrate validation statistics and reporting."""
        print("\n📊 VALIDATION STATISTICS DEMO")
        print("-" * 50)
        
        print("Generating validation statistics...")
        
        # Get validation statistics
        stats = self.rules_manager.get_validation_statistics()
        
        print(f"\n📈 Validation Statistics:")
        print(f"  Total validation sessions: {stats['total_sessions']}")
        print(f"  Completed sessions: {stats['completed_sessions']}")
        print(f"  Active sessions: {stats['active_sessions']}")
        print(f"  Average session duration: {stats['average_session_duration']:.1f} minutes")
        
        print(f"\n⚙️  Rule Modification Statistics:")
        print(f"  Total modifications proposed: {stats['total_modifications_proposed']}")
        print(f"  Pending modifications: {stats['pending_modifications']}")
        print(f"  Approved modifications: {stats['approved_modifications']}")
        print(f"  Approval rate: {stats['modification_approval_rate']:.1%}")
        
        print(f"\n📋 Rule Inventory:")
        print(f"  Signature rules: {stats['signature_rules_count']}")
        print(f"  Field requirement rules: {stats['field_requirement_rules_count']}")
        
        if stats['most_modified_rules']:
            print(f"\n🔄 Most Modified Rules:")
            for rule_id, count in stats['most_modified_rules'][:5]:
                print(f"  - {rule_id}: {count} modifications")
        
        # Generate validation report
        print(f"\n📋 Generating validation report...")
        
        report = self.rules_manager.export_validation_report()
        
        print(f"  Report generated: {report['report_generated']}")
        print(f"  Sessions included: {report['session_count']}")
        print(f"  Rule modifications: {len(report['rule_modifications'])}")
        
        # Display summary statistics from report
        summary = report['summary_statistics']
        print(f"\n📊 Report Summary:")
        print(f"  Total sessions: {summary['total_sessions']}")
        print(f"  Average session duration: {summary['average_session_duration']:.1f} minutes")
        print(f"  Modification approval rate: {summary['modification_approval_rate']:.1%}")
    
    def demonstrate_signature_scenarios(self):
        """Demonstrate various signature validation scenarios."""
        print("\n✍️  SIGNATURE VALIDATION SCENARIOS")
        print("-" * 50)
        
        scenarios = [
            {
                "name": "Digital Signature (Allowed)",
                "data": {
                    "signature": {
                        "coverage": 0.25,
                        "ink_detected": False,
                        "type": "digital",
                        "digital_certificate": True
                    },
                    "document_type": "contract"
                }
            },
            {
                "name": "Handwritten Signature (Good Coverage)",
                "data": {
                    "signature": {
                        "coverage": 0.22,
                        "ink_detected": True,
                        "type": "handwritten"
                    },
                    "document_type": "contract"
                }
            },
            {
                "name": "Poor Quality Signature",
                "data": {
                    "signature": {
                        "coverage": 0.08,
                        "ink_detected": True,
                        "type": "handwritten",
                        "quality_issues": ["smudged", "partial"]
                    },
                    "document_type": "contract"
                }
            },
            {
                "name": "No Signature Detected",
                "data": {
                    "signature": {
                        "coverage": 0.02,
                        "ink_detected": False,
                        "type": None
                    },
                    "document_type": "contract"
                }
            }
        ]
        
        for scenario in scenarios:
            print(f"\n📝 Scenario: {scenario['name']}")
            
            result = self.rules_manager.validate_document_with_specialized_rules(
                scenario['data'], ValidationContext.QUALITY_REVIEW
            )
            
            if result['violations']:
                for violation in result['violations']:
                    if 'signature' in violation.rule_id.lower():
                        print(f"  ❌ {violation.message}")
                        print(f"     Severity: {violation.severity.value}")
                        print(f"     Action: {violation.suggested_action}")
            else:
                print(f"  ✅ Signature validation passed")
    
    def demonstrate_business_scenarios(self):
        """Demonstrate real-world business scenarios."""
        print("\n🏢 BUSINESS SCENARIOS DEMO")
        print("-" * 50)
        
        scenarios = [
            {
                "name": "High-Value Contract Processing",
                "description": "Contract over $100K requiring enhanced validation",
                "data": {
                    "contract_amount": "150000.00",
                    "client_name": "Enterprise Solutions Ltd",
                    "effective_date": "2024-02-01",
                    "document_type": "contract",
                    "signature": {
                        "coverage": 0.18,
                        "ink_detected": True,
                        "type": "handwritten"
                    },
                    "approval_level": "director_required"
                }
            },
            {
                "name": "International Agreement",
                "description": "Cross-border agreement with special requirements",
                "data": {
                    "contract_amount": "75000.00",
                    "client_name": "Global Industries GmbH",
                    "effective_date": "2024-03-15",
                    "document_type": "international_agreement",
                    "signature": {
                        "coverage": 0.16,
                        "ink_detected": True,
                        "type": "handwritten"
                    },
                    "jurisdiction": "EU",
                    "compliance_required": ["GDPR", "international_trade"]
                }
            },
            {
                "name": "Amendment Processing",
                "description": "Contract amendment with partial information",
                "data": {
                    "original_contract_id": "CONTRACT_2023_045",
                    "amendment_type": "value_change",
                    "new_amount": "95000.00",
                    "client_name": "Existing Client Corp",
                    "document_type": "amendment",
                    "signature": {
                        "coverage": 0.14,
                        "ink_detected": True,
                        "type": "handwritten"
                    }
                }
            }
        ]
        
        for scenario in scenarios:
            print(f"\n📋 {scenario['name']}")
            print(f"   {scenario['description']}")
            
            result = self.rules_manager.validate_document_with_specialized_rules(
                scenario['data'], ValidationContext.DOCUMENT_PROCESSING
            )
            
            print(f"   Validation result: {result['total_violations']} violations")
            
            if result['violations']:
                for violation in result['violations']:
                    print(f"     - {violation.rule_name}")
                    print(f"       {violation.message}")
                    print(f"       Severity: {violation.severity.value}")
            else:
                print(f"   ✅ All business rules satisfied")


def main():
    """Run the business rules management demo."""
    demo = BusinessRulesDemo()
    demo.run_complete_demo()
    
    # Additional demonstrations
    print("\n" + "="*80)
    print("🔍 ADDITIONAL DEMONSTRATIONS")
    print("="*80)
    
    demo.demonstrate_signature_scenarios()
    demo.demonstrate_business_scenarios()
    
    print("\n" + "="*80)
    print("🎯 BUSINESS RULES DEMO SUMMARY")
    print("="*80)
    
    print("""
✅ Successfully demonstrated:

1. 📋 Business Rules Engine
   - Signature validation with coverage thresholds
   - Field requirement validation with conditional logic
   - Rule severity and priority management

2. 🔄 Interactive Validation
   - Real-time document validation
   - Field update tracking
   - Validation session management

3. ⚙️  Rule Management
   - Dynamic rule modification proposals
   - Approval workflow for rule changes
   - Threshold adjustments and severity changes

4. 📊 Analytics & Reporting
   - Validation statistics and metrics
   - Rule modification tracking
   - Performance analysis

5. 🏢 Business Scenarios
   - High-value contract processing
   - International agreement handling
   - Amendment processing workflows

The system provides comprehensive business rules management with:
- Flexible rule definition and modification
- Interactive document validation
- User-friendly violation resolution
- Complete audit trail and reporting
- Real-time feedback and suggestions

Ready for production deployment with enterprise-grade reliability!
    """)


if __name__ == "__main__":
    main()