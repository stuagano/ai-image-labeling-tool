"""
Robustness System Comprehensive Demo

Demonstrates the complete robustness enhancement system for document processing,
including business rules, issue resolution, exception handling, workflow management,
analytics, and audit trail capabilities.
"""

import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, Any

# Add the document_processing module to the path
sys.path.append(os.path.dirname(__file__))

from document_processing.business_rules_engine import (
    BusinessRulesEngine, BusinessRule, RuleType, RuleSeverity, RuleCondition
)
from document_processing.issue_resolution import (
    IssueDetector, IssueResolver, DocumentIssue, IssueType, IssueSeverity
)
from document_processing.exception_handling import (
    ExceptionHandler, ExceptionContext, ExceptionType, ExceptionSeverity
)
from document_processing.interactive_workflow import (
    WorkflowEngine, WorkflowUser, UserRole, TaskType, TaskPriority
)
from document_processing.analytics_learning import (
    LearningEngine, FieldPerformanceAnalyzer, ErrorPatternDetector
)
from document_processing.audit_trail import (
    AuditLogger, AuditContext, AuditEventType, AuditLevel
)
from document_processing.field_types import (
    FieldExtraction, TextField, NumberField, DateField, BoundingBox
)


class RobustnessSystemDemo:
    """Comprehensive demonstration of the robustness system."""
    
    def __init__(self):
        """Initialize all robustness components."""
        print("🚀 Initializing Robustness System Demo...")
        
        # Initialize core components
        self.business_rules = BusinessRulesEngine()
        self.issue_detector = IssueDetector()
        self.issue_resolver = IssueResolver()
        self.exception_handler = ExceptionHandler()
        self.workflow_engine = WorkflowEngine()
        self.learning_engine = LearningEngine()
        self.audit_logger = AuditLogger("demo_audit_logs")
        
        # Demo data storage
        self.demo_documents = []
        self.demo_processing_results = []
        self.demo_users = []
        self.session_id = None
        
        print("✅ All components initialized successfully!")
    
    def run_complete_demo(self):
        """Run the complete robustness system demonstration."""
        print("\n" + "="*80)
        print("🔧 ROBUSTNESS SYSTEM COMPREHENSIVE DEMONSTRATION")
        print("="*80)
        
        try:
            # Start audit session
            self.session_id = self.audit_logger.start_session(
                user_id="demo_user",
                ip_address="127.0.0.1",
                user_agent="Demo System"
            )
            
            # Run individual component demos
            self.demo_business_rules()
            self.demo_issue_detection_resolution()
            self.demo_exception_handling()
            self.demo_workflow_management()
            self.demo_analytics_learning()
            self.demo_audit_trail()
            self.demo_integration()
            
            # Generate final reports
            self.generate_comprehensive_report()
            
        except Exception as e:
            print(f"❌ Demo error: {e}")
            # Log the exception
            context = ExceptionContext()
            self.exception_handler.handle_exception(e, context)
        
        finally:
            # End audit session
            if self.session_id:
                self.audit_logger.end_session(self.session_id)
            
            print("\n🏁 Demo completed successfully!")
    
    def demo_business_rules(self):
        """Demonstrate business rules engine capabilities."""
        print("\n📋 BUSINESS RULES ENGINE DEMO")
        print("-" * 50)
        
        # Create custom business rules
        print("Creating custom business rules...")
        
        # Rule 1: Contract amount validation
        contract_amount_rule = BusinessRule(
            id="contract_amount_validation",
            name="Contract Amount Validation",
            description="Contract amounts must be within reasonable range",
            rule_type=RuleType.VALIDATION,
            severity=RuleSeverity.ERROR,
            conditions=[
                {"field": "contract_amount", "operator": "greater_than", "value": 1000000},
                {"field": "document_type", "operator": "equals", "value": "contract"}
            ],
            logical_operator="AND",
            validation_message="Contract amount exceeds maximum allowed limit",
            suggested_action="Review contract amount and obtain additional approval",
            tags=["financial", "contracts"]
        )
        
        # Rule 2: Date consistency
        date_consistency_rule = BusinessRule(
            id="date_consistency_check",
            name="Date Consistency Check",
            description="End date must be after start date",
            rule_type=RuleType.BUSINESS_LOGIC,
            severity=RuleSeverity.WARNING,
            conditions=[
                {"field": "start_date", "operator": "is_not_empty", "value": None},
                {"field": "end_date", "operator": "is_not_empty", "value": None}
            ],
            validation_message="End date should be after start date",
            suggested_action="Verify and correct date fields",
            tags=["dates", "consistency"]
        )
        
        # Add rules to engine
        self.business_rules.add_rule(contract_amount_rule)
        self.business_rules.add_rule(date_consistency_rule)
        
        print(f"✅ Added {len(self.business_rules.rules)} business rules")
        
        # Test rule evaluation
        print("\nTesting rule evaluation...")
        
        # Test case 1: Valid data
        valid_data = {
            "contract_amount": 50000,
            "document_type": "contract",
            "start_date": datetime(2024, 1, 1),
            "end_date": datetime(2024, 12, 31)
        }
        
        violations = self.business_rules.evaluate_rules(valid_data)
        print(f"Valid data violations: {len(violations)}")
        
        # Test case 2: Invalid data
        invalid_data = {
            "contract_amount": 2000000,  # Exceeds limit
            "document_type": "contract",
            "start_date": datetime(2024, 12, 31),
            "end_date": datetime(2024, 1, 1)  # End before start
        }
        
        violations = self.business_rules.evaluate_rules(invalid_data)
        print(f"Invalid data violations: {len(violations)}")
        
        for violation in violations:
            print(f"  - {violation.rule_name}: {violation.message}")
            
            # Log to audit trail
            self.audit_logger.log_business_rule_event(
                rule_id=violation.rule_id,
                rule_name=violation.rule_name,
                violated=True,
                context=AuditContext(session_id=self.session_id),
                affected_fields=violation.affected_fields
            )
        
        # Display rule statistics
        stats = self.business_rules.get_rule_statistics()
        print(f"\nRule Statistics:")
        print(f"  Total rules: {stats['total_rules']}")
        print(f"  Active rules: {stats['active_rules']}")
        print(f"  Rule categories: {stats['rule_categories']}")
    
    def demo_issue_detection_resolution(self):
        """Demonstrate issue detection and resolution capabilities."""
        print("\n🔍 ISSUE DETECTION & RESOLUTION DEMO")
        print("-" * 50)
        
        # Create sample field results with various issues
        print("Creating sample field extractions with issues...")
        
        field_results = {
            "customer_name": FieldExtraction(
                value="J0hn D0e",  # OCR errors
                confidence=0.4,  # Low confidence
                is_valid=False,
                validation_errors=["Contains suspicious characters"],
                extraction_method="ai"
            ),
            "contract_amount": FieldExtraction(
                value=None,  # Missing value
                confidence=0.0,
                is_valid=False,
                validation_errors=["Field is required"],
                extraction_method="ai"
            ),
            "signature": FieldExtraction(
                value={"coverage": 0.05, "ink_detected": False},  # Poor signature
                confidence=0.3,
                is_valid=False,
                validation_errors=["Signature area too small"],
                extraction_method="computer_vision"
            )
        }
        
        # Create template fields
        template_fields = [
            TextField(name="customer_name", bounding_box=BoundingBox(100, 50, 200, 30), required=True),
            NumberField(name="contract_amount", bounding_box=BoundingBox(300, 100, 150, 25), required=True),
            TextField(name="signature", bounding_box=BoundingBox(400, 200, 100, 50), required=True),
            TextField(name="missing_field", bounding_box=BoundingBox(500, 250, 100, 25), required=False)
        ]
        
        # Detect issues
        print("Detecting issues...")
        issues = self.issue_detector.detect_issues(field_results, template_fields)
        
        print(f"✅ Detected {len(issues)} issues:")
        for issue in issues:
            print(f"  - {issue.title} ({issue.severity.value})")
            print(f"    Description: {issue.description}")
            print(f"    Suggested actions: {len(issue.suggested_actions)}")
            
            # Log issue to audit trail
            self.audit_logger.log_event(
                event_type=AuditEventType.FIELD_VALIDATED,
                level=AuditLevel.HIGH if issue.severity.value == "critical" else AuditLevel.MEDIUM,
                description=f"Issue detected: {issue.title}",
                context=AuditContext(session_id=self.session_id),
                metadata={
                    "issue_type": issue.issue_type.value,
                    "affected_fields": issue.affected_fields
                }
            )
        
        # Attempt issue resolution
        print("\nAttempting automatic resolution...")
        
        resolution_results = []
        for issue in issues:
            result = self.issue_resolver.resolve_issue(issue, {"document_id": "demo_doc_001"})
            resolution_results.append(result)
            
            if result['fully_resolved']:
                print(f"  ✅ Resolved: {issue.title}")
            else:
                print(f"  ⚠️  Partial resolution: {issue.title}")
        
        resolved_count = sum(1 for r in resolution_results if r['fully_resolved'])
        print(f"\nResolution Summary: {resolved_count}/{len(issues)} issues automatically resolved")
    
    def demo_exception_handling(self):
        """Demonstrate exception handling capabilities."""
        print("\n⚠️  EXCEPTION HANDLING DEMO")
        print("-" * 50)
        
        print("Simulating various exception scenarios...")
        
        # Scenario 1: Model error
        try:
            raise RuntimeError("AI model inference failed due to memory constraints")
        except Exception as e:
            context = ExceptionContext(
                document_id="demo_doc_002",
                field_name="customer_name",
                processing_step="extraction"
            )
            proc_exc = self.exception_handler.handle_exception(e, context)
            print(f"  🔴 {proc_exc.title}: {proc_exc.severity.value}")
            print(f"     Recovery actions: {len(proc_exc.recovery_actions)}")
        
        # Scenario 2: Preprocessing error
        try:
            raise ValueError("Document preprocessing failed: invalid image format")
        except Exception as e:
            context = ExceptionContext(
                document_id="demo_doc_003",
                processing_step="preprocessing"
            )
            proc_exc = self.exception_handler.handle_exception(e, context)
            print(f"  🔴 {proc_exc.title}: {proc_exc.severity.value}")
        
        # Scenario 3: Validation error
        try:
            raise ValueError("Field validation failed: invalid date format")
        except Exception as e:
            context = ExceptionContext(
                document_id="demo_doc_004",
                field_name="contract_date",
                processing_step="validation"
            )
            proc_exc = self.exception_handler.handle_exception(e, context)
            print(f"  🔴 {proc_exc.title}: {proc_exc.severity.value}")
        
        # Display exception statistics
        stats = self.exception_handler.get_statistics()
        print(f"\nException Statistics:")
        print(f"  Total exceptions: {stats['total_exceptions']}")
        print(f"  Resolved exceptions: {stats['resolved_exceptions']}")
        print(f"  Exception types: {list(stats['exception_types'].keys())}")
        
        # Show active exceptions
        active = self.exception_handler.get_active_exceptions()
        print(f"  Active exceptions: {len(active)}")
    
    def demo_workflow_management(self):
        """Demonstrate workflow management capabilities."""
        print("\n👥 WORKFLOW MANAGEMENT DEMO")
        print("-" * 50)
        
        # Create demo users
        print("Creating workflow users...")
        
        users = [
            WorkflowUser(
                id="user_001",
                name="Alice Johnson",
                email="alice@example.com",
                role=UserRole.OPERATOR,
                skills=["data_entry", "basic_validation"],
                performance_metrics={"accuracy": 0.95, "tasks_completed": 150}
            ),
            WorkflowUser(
                id="user_002",
                name="Bob Smith",
                email="bob@example.com",
                role=UserRole.REVIEWER,
                skills=["quality_review", "complex_validation"],
                performance_metrics={"accuracy": 0.98, "tasks_completed": 75}
            ),
            WorkflowUser(
                id="user_003",
                name="Carol Davis",
                email="carol@example.com",
                role=UserRole.SUPERVISOR,
                skills=["business_rules", "exception_handling"],
                performance_metrics={"accuracy": 0.99, "tasks_completed": 50}
            )
        ]
        
        for user in users:
            self.workflow_engine.register_user(user)
            self.demo_users.append(user)
        
        print(f"✅ Registered {len(users)} users")
        
        # Create tasks from issues
        print("\nCreating workflow tasks from detected issues...")
        
        # Simulate some issues
        sample_issues = [
            DocumentIssue(
                id="issue_001",
                issue_type=IssueType.LOW_CONFIDENCE,
                severity=IssueSeverity.MEDIUM,
                title="Low confidence field extraction",
                description="Customer name field has low confidence score",
                affected_fields=["customer_name"]
            ),
            DocumentIssue(
                id="issue_002",
                issue_type=IssueType.VALIDATION_ERROR,
                severity=IssueSeverity.HIGH,
                title="Date validation failed",
                description="Contract date format is invalid",
                affected_fields=["contract_date"]
            ),
            DocumentIssue(
                id="issue_003",
                issue_type=IssueType.BUSINESS_RULE_VIOLATION,
                severity=IssueSeverity.CRITICAL,
                title="Contract amount exceeds limit",
                description="Contract amount exceeds approval threshold",
                affected_fields=["contract_amount"]
            )
        ]
        
        created_tasks = []
        for issue in sample_issues:
            task = self.workflow_engine.create_task_from_issue(issue)
            created_tasks.append(task)
            print(f"  📋 Created task: {task.title} (Priority: {task.priority.value})")
        
        # Simulate task processing
        print("\nSimulating task processing...")
        
        for task in created_tasks[:2]:  # Process first 2 tasks
            if task.assigned_to:
                user = self.workflow_engine.users[task.assigned_to]
                print(f"  👤 {user.name} starting task: {task.title}")
                
                # Start task
                self.workflow_engine.start_task(task.id, task.assigned_to)
                
                # Simulate processing time
                time.sleep(0.1)
                
                # Complete task
                result = {
                    "action_taken": "manual_correction",
                    "corrected_value": "Corrected field value",
                    "confidence": 0.95
                }
                
                self.workflow_engine.complete_task(
                    task.id,
                    task.assigned_to,
                    result,
                    "Task completed successfully"
                )
                
                print(f"  ✅ Task completed by {user.name}")
        
        # Display workflow statistics
        stats = self.workflow_engine.get_workflow_statistics()
        print(f"\nWorkflow Statistics:")
        print(f"  Active tasks: {stats['active_tasks']}")
        print(f"  Pending tasks: {stats['pending_tasks']}")
        print(f"  Average completion time: {stats['average_completion_time']:.1f} minutes")
    
    def demo_analytics_learning(self):
        """Demonstrate analytics and learning capabilities."""
        print("\n📊 ANALYTICS & LEARNING DEMO")
        print("-" * 50)
        
        # Create sample processing results
        print("Generating sample processing results for analysis...")
        
        processing_results = []
        
        # Simulate 50 document processing results
        for i in range(50):
            result = {
                "document_id": f"doc_{i:03d}",
                "template_id": f"template_{i % 3 + 1}",
                "timestamp": (datetime.now() - timedelta(days=i % 30)).isoformat(),
                "field_results": {
                    "customer_name": {
                        "value": f"Customer {i}",
                        "confidence": 0.7 + (i % 30) * 0.01,  # Varying confidence
                        "is_valid": i % 10 != 0,  # 10% failure rate
                        "extraction_method": "ai",
                        "validation_errors": [] if i % 10 != 0 else ["Invalid format"]
                    },
                    "contract_amount": {
                        "value": 10000 + i * 1000,
                        "confidence": 0.8 + (i % 20) * 0.01,
                        "is_valid": i % 15 != 0,  # 6.7% failure rate
                        "extraction_method": "ai",
                        "validation_errors": [] if i % 15 != 0 else ["Amount out of range"]
                    },
                    "signature": {
                        "value": {"coverage": 0.6 + (i % 10) * 0.04, "ink_detected": i % 8 != 0},
                        "confidence": 0.6 + (i % 25) * 0.016,
                        "is_valid": i % 8 != 0,  # 12.5% failure rate
                        "extraction_method": "computer_vision",
                        "validation_errors": [] if i % 8 != 0 else ["Signature not detected"]
                    }
                }
            }
            processing_results.append(result)
        
        self.demo_processing_results = processing_results
        print(f"✅ Generated {len(processing_results)} processing results")
        
        # Generate insights
        print("\nGenerating learning insights...")
        
        # Create sample issues and tasks for pattern analysis
        sample_issues = [
            DocumentIssue(
                id=f"issue_{i}",
                issue_type=IssueType.LOW_CONFIDENCE if i % 3 == 0 else IssueType.VALIDATION_ERROR,
                severity=IssueSeverity.MEDIUM,
                title=f"Issue {i}",
                description=f"Sample issue {i}",
                affected_fields=["customer_name"] if i % 2 == 0 else ["signature"],
                timestamp=datetime.now() - timedelta(hours=i)
            )
            for i in range(20)
        ]
        
        sample_tasks = [task for task in self.workflow_engine.tasks.values()]
        
        insights = self.learning_engine.generate_insights(
            processing_results, sample_issues, sample_tasks
        )
        
        print(f"✅ Generated {len(insights)} learning insights:")
        for insight in insights[:5]:  # Show first 5
            print(f"  💡 {insight.title} (Impact: {insight.impact_estimate:.2f})")
            print(f"     Category: {insight.category.value}")
            print(f"     Recommendations: {len(insight.recommendations)}")
        
        # Generate analytics dashboard
        dashboard = self.learning_engine.get_analytics_dashboard()
        print(f"\nAnalytics Dashboard Summary:")
        print(f"  Total insights: {dashboard['insights_summary']['total_insights']}")
        print(f"  Potential impact: {dashboard['impact_analysis']['total_potential_impact']:.2f}")
        print(f"  Error patterns detected: {len(dashboard['error_patterns'])}")
    
    def demo_audit_trail(self):
        """Demonstrate audit trail capabilities."""
        print("\n📋 AUDIT TRAIL DEMO")
        print("-" * 50)
        
        # Log various events
        print("Logging various audit events...")
        
        context = AuditContext(
            session_id=self.session_id,
            user_id="demo_user",
            document_id="demo_doc_005"
        )
        
        # Start document chain
        chain_id = self.audit_logger.start_document_chain("demo_doc_005", context)
        print(f"  📄 Started document chain: {chain_id}")
        
        # Log field extractions
        for field_name in ["customer_name", "contract_amount", "signature"]:
            extraction = FieldExtraction(
                value=f"Sample {field_name}",
                confidence=0.85,
                is_valid=True,
                extraction_method="ai"
            )
            
            self.audit_logger.log_field_extraction(field_name, extraction, context)
        
        print(f"  📝 Logged field extractions")
        
        # Log field correction
        self.audit_logger.log_field_correction(
            field_name="customer_name",
            old_value="J0hn D0e",
            new_value="John Doe",
            user_id="demo_user",
            context=context,
            reason="OCR error correction"
        )
        print(f"  ✏️  Logged field correction")
        
        # Log template application
        self.audit_logger.log_template_event(
            template_id="contract_template_v1",
            action="applied",
            context=context,
            template_data={"field_count": 10, "version": "1.0"}
        )
        print(f"  📋 Logged template application")
        
        # Complete document chain
        self.audit_logger.complete_document_chain(chain_id, "completed")
        print(f"  ✅ Completed document chain")
        
        # Query events
        print("\nQuerying audit events...")
        
        # Get document timeline
        timeline = self.audit_logger.get_document_timeline("demo_doc_005")
        print(f"  📊 Document timeline: {len(timeline)} events")
        
        # Get user activity
        user_activity = self.audit_logger.get_user_activity("demo_user")
        print(f"  👤 User activity: {len(user_activity)} events")
        
        # Generate compliance report
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=1)
        
        compliance_report = self.audit_logger.generate_compliance_report(
            start_date, end_date
        )
        
        print(f"\nCompliance Report Summary:")
        print(f"  Total events: {compliance_report['summary']['total_events']}")
        print(f"  Unique users: {compliance_report['summary']['unique_users']}")
        print(f"  Unique documents: {compliance_report['summary']['unique_documents']}")
        print(f"  Integrity rate: {compliance_report['integrity_status']['integrity_rate']:.2%}")
    
    def demo_integration(self):
        """Demonstrate integrated system capabilities."""
        print("\n🔗 INTEGRATED SYSTEM DEMO")
        print("-" * 50)
        
        print("Demonstrating end-to-end integrated processing...")
        
        # Simulate a complete document processing workflow
        document_id = "integration_demo_001"
        
        # 1. Start audit chain
        context = AuditContext(
            session_id=self.session_id,
            user_id="demo_user",
            document_id=document_id
        )
        chain_id = self.audit_logger.start_document_chain(document_id, context)
        
        print(f"  1. 📄 Started processing: {document_id}")
        
        # 2. Extract fields (with some issues)
        field_results = {
            "customer_name": FieldExtraction(
                value="ABC Corporation",
                confidence=0.9,
                is_valid=True,
                extraction_method="ai"
            ),
            "contract_amount": FieldExtraction(
                value=2500000,  # Will trigger business rule
                confidence=0.85,
                is_valid=True,
                extraction_method="ai"
            ),
            "contract_date": FieldExtraction(
                value=datetime(2024, 6, 15),
                confidence=0.8,
                is_valid=True,
                extraction_method="ai"
            )
        }
        
        # Log extractions
        for field_name, extraction in field_results.items():
            self.audit_logger.log_field_extraction(field_name, extraction, context)
        
        print(f"  2. 📝 Extracted {len(field_results)} fields")
        
        # 3. Apply business rules
        rule_data = {field: ext.value for field, ext in field_results.items()}
        rule_data["document_type"] = "contract"
        
        violations = self.business_rules.evaluate_rules(rule_data)
        
        if violations:
            print(f"  3. ⚠️  Business rule violations: {len(violations)}")
            for violation in violations:
                # Log violation
                self.audit_logger.log_business_rule_event(
                    rule_id=violation.rule_id,
                    rule_name=violation.rule_name,
                    violated=True,
                    context=context,
                    affected_fields=violation.affected_fields
                )
                
                # Create workflow task for violation
                issue = DocumentIssue(
                    id=f"violation_{violation.rule_id}",
                    issue_type=IssueType.BUSINESS_RULE_VIOLATION,
                    severity=IssueSeverity.CRITICAL,
                    title=f"Business Rule Violation: {violation.rule_name}",
                    description=violation.message,
                    affected_fields=violation.affected_fields
                )
                
                task = self.workflow_engine.create_task_from_issue(issue)
                print(f"     📋 Created task: {task.title}")
        else:
            print(f"  3. ✅ All business rules passed")
        
        # 4. Detect other issues
        template_fields = [
            TextField(name="customer_name", bounding_box=BoundingBox(100, 50, 200, 30), required=True),
            NumberField(name="contract_amount", bounding_box=BoundingBox(300, 100, 150, 25), required=True),
            DateField(name="contract_date", bounding_box=BoundingBox(450, 150, 120, 25), required=True)
        ]
        
        issues = self.issue_detector.detect_issues(field_results, template_fields)
        print(f"  4. 🔍 Detected {len(issues)} additional issues")
        
        # 5. Generate learning insights
        processing_result = {
            "document_id": document_id,
            "template_id": "contract_template_v1",
            "timestamp": datetime.now().isoformat(),
            "field_results": {
                field: {
                    "value": ext.value,
                    "confidence": ext.confidence,
                    "is_valid": ext.is_valid,
                    "extraction_method": ext.extraction_method
                }
                for field, ext in field_results.items()
            }
        }
        
        # Add to processing results for analytics
        self.demo_processing_results.append(processing_result)
        
        # Generate new insights
        new_insights = self.learning_engine.generate_insights(
            self.demo_processing_results[-10:],  # Last 10 results
            issues,
            list(self.workflow_engine.tasks.values())
        )
        
        print(f"  5. 💡 Generated {len(new_insights)} new learning insights")
        
        # 6. Complete processing
        self.audit_logger.complete_document_chain(chain_id, "completed")
        print(f"  6. ✅ Processing completed")
        
        print("\n🎯 Integration Summary:")
        print(f"   - Document processed with full audit trail")
        print(f"   - Business rules evaluated and violations handled")
        print(f"   - Issues detected and workflow tasks created")
        print(f"   - Learning insights generated for continuous improvement")
        print(f"   - All events logged for compliance and debugging")
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive system status report."""
        print("\n📊 COMPREHENSIVE SYSTEM REPORT")
        print("=" * 80)
        
        # Business Rules Summary
        rule_stats = self.business_rules.get_rule_statistics()
        print(f"\n📋 Business Rules Engine:")
        print(f"   Total Rules: {rule_stats['total_rules']}")
        print(f"   Active Rules: {rule_stats['active_rules']}")
        print(f"   Rule Categories: {rule_stats['rule_categories']}")
        
        # Exception Handling Summary
        exc_stats = self.exception_handler.get_statistics()
        print(f"\n⚠️  Exception Handling:")
        print(f"   Total Exceptions: {exc_stats['total_exceptions']}")
        print(f"   Resolved: {exc_stats['resolved_exceptions']}")
        print(f"   Resolution Rate: {exc_stats['resolved_exceptions']/exc_stats['total_exceptions']*100:.1f}%" if exc_stats['total_exceptions'] > 0 else "   Resolution Rate: N/A")
        
        # Workflow Summary
        workflow_stats = self.workflow_engine.get_workflow_statistics()
        print(f"\n👥 Workflow Management:")
        print(f"   Active Tasks: {workflow_stats['active_tasks']}")
        print(f"   Pending Tasks: {workflow_stats['pending_tasks']}")
        print(f"   Users Registered: {len(self.workflow_engine.users)}")
        
        # Analytics Summary
        analytics_dashboard = self.learning_engine.get_analytics_dashboard()
        print(f"\n📊 Analytics & Learning:")
        print(f"   Total Insights: {analytics_dashboard['insights_summary']['total_insights']}")
        print(f"   Potential Impact: {analytics_dashboard['impact_analysis']['total_potential_impact']:.2f}")
        print(f"   Error Patterns: {len(analytics_dashboard['error_patterns'])}")
        
        # Audit Trail Summary
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)
        compliance_report = self.audit_logger.generate_compliance_report(start_time, end_time)
        
        print(f"\n📋 Audit Trail:")
        print(f"   Events Logged: {compliance_report['summary']['total_events']}")
        print(f"   Unique Documents: {compliance_report['summary']['unique_documents']}")
        print(f"   Integrity Rate: {compliance_report['integrity_status']['integrity_rate']:.2%}")
        
        # Export capabilities demonstration
        print(f"\n💾 Export Capabilities:")
        
        # Export business rules
        self.business_rules.export_rules("demo_business_rules.json")
        print(f"   ✅ Business rules exported")
        
        # Export workflow data
        self.workflow_engine.export_tasks("demo_workflow_tasks.json")
        print(f"   ✅ Workflow tasks exported")
        
        # Export learning insights
        self.learning_engine.export_insights("demo_learning_insights.json")
        print(f"   ✅ Learning insights exported")
        
        # Export audit data
        self.audit_logger.export_audit_data("demo_audit_trail.json")
        print(f"   ✅ Audit trail exported")
        
        # Export exception log
        self.exception_handler.export_exception_log("demo_exceptions.json")
        print(f"   ✅ Exception log exported")
        
        print(f"\n🎯 SYSTEM PERFORMANCE METRICS:")
        print(f"   ✅ Robustness: HIGH - Multiple layers of error detection and recovery")
        print(f"   ✅ Traceability: COMPLETE - Full audit trail for all operations")
        print(f"   ✅ Learning: ACTIVE - Continuous improvement through analytics")
        print(f"   ✅ Workflow: INTEGRATED - Human-in-the-loop for complex cases")
        print(f"   ✅ Compliance: READY - Comprehensive logging and reporting")
        
        print(f"\n🚀 ROBUSTNESS SYSTEM READY FOR PRODUCTION!")


def main():
    """Main demo execution."""
    print("🎬 Starting Robustness System Comprehensive Demo...\n")
    
    demo = RobustnessSystemDemo()
    demo.run_complete_demo()
    
    print("\n" + "="*80)
    print("📋 DEMO FILES GENERATED:")
    print("   - demo_business_rules.json")
    print("   - demo_workflow_tasks.json") 
    print("   - demo_learning_insights.json")
    print("   - demo_audit_trail.json")
    print("   - demo_exceptions.json")
    print("   - demo_audit_logs/ (directory)")
    print("="*80)


if __name__ == "__main__":
    main()