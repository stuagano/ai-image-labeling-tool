"""
Advanced Analytics and Learning System

Comprehensive analytics and machine learning system for continuous
improvement of document processing accuracy and efficiency.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import statistics
from collections import defaultdict, Counter

from .field_types import FieldExtraction, FormField
from .issue_resolution import DocumentIssue
from .interactive_workflow import WorkflowTask


class AnalyticsType(Enum):
    """Types of analytics."""
    ACCURACY_TRENDS = "accuracy_trends"
    FIELD_PERFORMANCE = "field_performance"
    TEMPLATE_EFFECTIVENESS = "template_effectiveness"
    USER_PRODUCTIVITY = "user_productivity"
    ERROR_PATTERNS = "error_patterns"
    PROCESSING_EFFICIENCY = "processing_efficiency"


class LearningCategory(Enum):
    """Categories of learning insights."""
    FIELD_EXTRACTION = "field_extraction"
    VALIDATION_RULES = "validation_rules"
    BUSINESS_RULES = "business_rules"
    TEMPLATE_OPTIMIZATION = "template_optimization"
    WORKFLOW_OPTIMIZATION = "workflow_optimization"


@dataclass
class ProcessingMetric:
    """Represents a processing metric over time."""
    timestamp: datetime
    metric_name: str
    value: float
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningInsight:
    """Represents a learning insight from data analysis."""
    id: str
    category: LearningCategory
    title: str
    description: str
    confidence: float
    impact_estimate: float  # 0-1 scale
    
    # Supporting data
    evidence: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    # Implementation
    auto_implementable: bool = False
    implementation_complexity: str = "medium"  # low, medium, high
    
    # Metadata
    created_date: datetime = field(default_factory=datetime.now)
    status: str = "new"  # new, reviewed, implemented, rejected


@dataclass
class PatternAnalysis:
    """Represents analysis of patterns in processing data."""
    pattern_type: str
    pattern_description: str
    frequency: int
    examples: List[Dict[str, Any]] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)


class FieldPerformanceAnalyzer:
    """Analyzes performance of individual fields across documents."""
    
    def __init__(self):
        """Initialize field performance analyzer."""
        self.field_metrics: Dict[str, List[ProcessingMetric]] = defaultdict(list)
    
    def analyze_field_performance(
        self,
        field_results: List[Dict[str, Any]],
        time_window: timedelta = timedelta(days=30)
    ) -> Dict[str, Any]:
        """Analyze performance of fields over time.
        
        Args:
            field_results: List of field extraction results
            time_window: Time window for analysis
            
        Returns:
            Field performance analysis
        """
        cutoff_date = datetime.now() - time_window
        
        field_stats = {}
        
        for result in field_results:
            timestamp = datetime.fromisoformat(result.get('timestamp', datetime.now().isoformat()))
            if timestamp < cutoff_date:
                continue
            
            field_extractions = result.get('field_results', {})
            
            for field_name, extraction_data in field_extractions.items():
                if field_name not in field_stats:
                    field_stats[field_name] = {
                        'total_extractions': 0,
                        'successful_extractions': 0,
                        'confidence_scores': [],
                        'validation_failures': 0,
                        'extraction_methods': Counter(),
                        'common_errors': Counter()
                    }
                
                stats = field_stats[field_name]
                stats['total_extractions'] += 1
                
                confidence = extraction_data.get('confidence', 0)
                stats['confidence_scores'].append(confidence)
                
                if extraction_data.get('is_valid', False):
                    stats['successful_extractions'] += 1
                else:
                    stats['validation_failures'] += 1
                    errors = extraction_data.get('validation_errors', [])
                    for error in errors:
                        stats['common_errors'][error] += 1
                
                method = extraction_data.get('extraction_method', 'unknown')
                stats['extraction_methods'][method] += 1
        
        # Calculate derived metrics
        performance_analysis = {}
        
        for field_name, stats in field_stats.items():
            total = stats['total_extractions']
            if total == 0:
                continue
            
            success_rate = stats['successful_extractions'] / total
            avg_confidence = statistics.mean(stats['confidence_scores']) if stats['confidence_scores'] else 0
            confidence_std = statistics.stdev(stats['confidence_scores']) if len(stats['confidence_scores']) > 1 else 0
            
            performance_analysis[field_name] = {
                'success_rate': success_rate,
                'average_confidence': avg_confidence,
                'confidence_variance': confidence_std,
                'total_extractions': total,
                'validation_failure_rate': stats['validation_failures'] / total,
                'most_common_method': stats['extraction_methods'].most_common(1)[0] if stats['extraction_methods'] else None,
                'top_errors': stats['common_errors'].most_common(3),
                'performance_grade': self._calculate_performance_grade(success_rate, avg_confidence),
                'improvement_potential': self._calculate_improvement_potential(stats)
            }
        
        return performance_analysis
    
    def _calculate_performance_grade(self, success_rate: float, avg_confidence: float) -> str:
        """Calculate performance grade for a field."""
        combined_score = (success_rate + avg_confidence) / 2
        
        if combined_score >= 0.9:
            return "A"
        elif combined_score >= 0.8:
            return "B"
        elif combined_score >= 0.7:
            return "C"
        elif combined_score >= 0.6:
            return "D"
        else:
            return "F"
    
    def _calculate_improvement_potential(self, stats: Dict[str, Any]) -> str:
        """Calculate improvement potential for a field."""
        success_rate = stats['successful_extractions'] / stats['total_extractions']
        avg_confidence = statistics.mean(stats['confidence_scores']) if stats['confidence_scores'] else 0
        
        if success_rate < 0.7 or avg_confidence < 0.7:
            return "High"
        elif success_rate < 0.85 or avg_confidence < 0.85:
            return "Medium"
        else:
            return "Low"


class ErrorPatternDetector:
    """Detects patterns in errors and suggests improvements."""
    
    def __init__(self):
        """Initialize error pattern detector."""
        self.error_patterns: List[PatternAnalysis] = []
    
    def detect_error_patterns(
        self,
        issues: List[DocumentIssue],
        tasks: List[WorkflowTask]
    ) -> List[PatternAnalysis]:
        """Detect patterns in errors and issues.
        
        Args:
            issues: List of document issues
            tasks: List of workflow tasks
            
        Returns:
            List of detected error patterns
        """
        patterns = []
        
        # Analyze issue patterns
        patterns.extend(self._analyze_issue_patterns(issues))
        
        # Analyze task patterns
        patterns.extend(self._analyze_task_patterns(tasks))
        
        # Analyze temporal patterns
        patterns.extend(self._analyze_temporal_patterns(issues + [self._issue_from_task(task) for task in tasks]))
        
        self.error_patterns = patterns
        return patterns
    
    def _analyze_issue_patterns(self, issues: List[DocumentIssue]) -> List[PatternAnalysis]:
        """Analyze patterns in document issues."""
        patterns = []
        
        # Group issues by type
        issue_by_type = defaultdict(list)
        for issue in issues:
            issue_by_type[issue.issue_type.value].append(issue)
        
        # Analyze each type
        for issue_type, type_issues in issue_by_type.items():
            if len(type_issues) < 3:  # Need minimum occurrences
                continue
            
            # Find common fields
            field_counter = Counter()
            for issue in type_issues:
                for field in issue.affected_fields:
                    field_counter[field] += 1
            
            if field_counter:
                most_common_field = field_counter.most_common(1)[0]
                if most_common_field[1] >= len(type_issues) * 0.3:  # At least 30% of issues
                    pattern = PatternAnalysis(
                        pattern_type=f"{issue_type}_field_pattern",
                        pattern_description=f"Field '{most_common_field[0]}' frequently has {issue_type} issues",
                        frequency=most_common_field[1],
                        examples=[
                            {'issue_id': issue.id, 'fields': issue.affected_fields}
                            for issue in type_issues[:5]
                        ],
                        suggested_actions=[
                            f"Review template definition for field '{most_common_field[0]}'",
                            f"Adjust validation rules for {issue_type} issues",
                            "Consider specialized extraction method for this field"
                        ]
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _analyze_task_patterns(self, tasks: List[WorkflowTask]) -> List[PatternAnalysis]:
        """Analyze patterns in workflow tasks."""
        patterns = []
        
        # Group tasks by type
        task_by_type = defaultdict(list)
        for task in tasks:
            task_by_type[task.task_type.value].append(task)
        
        # Analyze completion times
        for task_type, type_tasks in task_by_type.items():
            completion_times = [
                task.actual_time for task in type_tasks 
                if task.actual_time and task.estimated_time
            ]
            
            if len(completion_times) >= 5:
                avg_actual = statistics.mean(completion_times)
                estimated_times = [
                    task.estimated_time for task in type_tasks 
                    if task.estimated_time and task.actual_time
                ]
                avg_estimated = statistics.mean(estimated_times) if estimated_times else 0
                
                if avg_actual > avg_estimated * 1.5:  # Significantly over estimate
                    pattern = PatternAnalysis(
                        pattern_type=f"{task_type}_time_overrun",
                        pattern_description=f"Tasks of type '{task_type}' consistently take longer than estimated",
                        frequency=len(completion_times),
                        examples=[
                            {
                                'task_id': task.id,
                                'estimated': task.estimated_time,
                                'actual': task.actual_time
                            }
                            for task in type_tasks[:3] if task.actual_time and task.estimated_time
                        ],
                        suggested_actions=[
                            f"Adjust time estimates for {task_type} tasks",
                            "Provide additional training for this task type",
                            "Review task complexity and break down if needed"
                        ]
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _analyze_temporal_patterns(self, items: List[Any]) -> List[PatternAnalysis]:
        """Analyze temporal patterns in issues and tasks."""
        patterns = []
        
        # Group by hour of day
        hour_counts = defaultdict(int)
        for item in items:
            if hasattr(item, 'timestamp'):
                hour = item.timestamp.hour
                hour_counts[hour] += 1
            elif hasattr(item, 'created_date'):
                hour = item.created_date.hour
                hour_counts[hour] += 1
        
        if hour_counts:
            # Find peak error hours
            max_count = max(hour_counts.values())
            avg_count = sum(hour_counts.values()) / len(hour_counts)
            
            peak_hours = [hour for hour, count in hour_counts.items() if count > avg_count * 1.5]
            
            if peak_hours:
                pattern = PatternAnalysis(
                    pattern_type="temporal_error_pattern",
                    pattern_description=f"Higher error rates during hours: {peak_hours}",
                    frequency=sum(hour_counts[hour] for hour in peak_hours),
                    examples=[
                        {'hour': hour, 'count': hour_counts[hour]}
                        for hour in peak_hours
                    ],
                    suggested_actions=[
                        "Review system performance during peak hours",
                        "Consider load balancing or resource scaling",
                        "Investigate if specific processes run during these hours"
                    ]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _issue_from_task(self, task: WorkflowTask) -> Any:
        """Convert task to issue-like object for analysis."""
        # Create a simple object with timestamp for temporal analysis
        class TaskLikeIssue:
            def __init__(self, timestamp):
                self.timestamp = timestamp
        
        return TaskLikeIssue(task.created_date)


class LearningEngine:
    """Main learning engine that generates insights and recommendations."""
    
    def __init__(self):
        """Initialize learning engine."""
        self.field_analyzer = FieldPerformanceAnalyzer()
        self.pattern_detector = ErrorPatternDetector()
        self.insights: List[LearningInsight] = []
        self.implementation_history: List[Dict[str, Any]] = []
    
    def generate_insights(
        self,
        processing_results: List[Dict[str, Any]],
        issues: List[DocumentIssue],
        tasks: List[WorkflowTask]
    ) -> List[LearningInsight]:
        """Generate learning insights from processing data.
        
        Args:
            processing_results: Historical processing results
            issues: Document issues
            tasks: Workflow tasks
            
        Returns:
            List of learning insights
        """
        insights = []
        
        # Analyze field performance
        field_performance = self.field_analyzer.analyze_field_performance(processing_results)
        insights.extend(self._generate_field_insights(field_performance))
        
        # Detect error patterns
        error_patterns = self.pattern_detector.detect_error_patterns(issues, tasks)
        insights.extend(self._generate_pattern_insights(error_patterns))
        
        # Analyze workflow efficiency
        insights.extend(self._generate_workflow_insights(tasks))
        
        # Generate template optimization insights
        insights.extend(self._generate_template_insights(processing_results))
        
        # Store insights
        self.insights.extend(insights)
        
        return insights
    
    def _generate_field_insights(self, field_performance: Dict[str, Any]) -> List[LearningInsight]:
        """Generate insights from field performance analysis."""
        insights = []
        
        for field_name, performance in field_performance.items():
            # Low performing fields
            if performance['performance_grade'] in ['D', 'F']:
                insight = LearningInsight(
                    id=f"field_performance_{field_name}_{datetime.now().timestamp()}",
                    category=LearningCategory.FIELD_EXTRACTION,
                    title=f"Poor Performance: Field '{field_name}'",
                    description=f"Field '{field_name}' has low performance (Grade: {performance['performance_grade']}, Success Rate: {performance['success_rate']:.2f})",
                    confidence=0.9,
                    impact_estimate=0.8,
                    evidence={
                        'success_rate': performance['success_rate'],
                        'average_confidence': performance['average_confidence'],
                        'total_extractions': performance['total_extractions'],
                        'top_errors': performance['top_errors']
                    },
                    recommendations=[
                        f"Review and adjust bounding box for field '{field_name}'",
                        "Consider using alternative extraction method",
                        "Add preprocessing steps to improve field visibility",
                        "Review validation rules for potential issues"
                    ],
                    auto_implementable=False,
                    implementation_complexity="medium"
                )
                insights.append(insight)
            
            # High variance in confidence
            if performance['confidence_variance'] > 0.3:
                insight = LearningInsight(
                    id=f"confidence_variance_{field_name}_{datetime.now().timestamp()}",
                    category=LearningCategory.FIELD_EXTRACTION,
                    title=f"Inconsistent Confidence: Field '{field_name}'",
                    description=f"Field '{field_name}' shows high variance in confidence scores",
                    confidence=0.7,
                    impact_estimate=0.6,
                    evidence={
                        'confidence_variance': performance['confidence_variance'],
                        'average_confidence': performance['average_confidence']
                    },
                    recommendations=[
                        "Investigate document quality variations",
                        "Consider confidence threshold adjustments",
                        "Review extraction method consistency"
                    ],
                    auto_implementable=True,
                    implementation_complexity="low"
                )
                insights.append(insight)
        
        return insights
    
    def _generate_pattern_insights(self, patterns: List[PatternAnalysis]) -> List[LearningInsight]:
        """Generate insights from error patterns."""
        insights = []
        
        for pattern in patterns:
            insight = LearningInsight(
                id=f"pattern_{pattern.pattern_type}_{datetime.now().timestamp()}",
                category=LearningCategory.TEMPLATE_OPTIMIZATION,
                title=f"Pattern Detected: {pattern.pattern_type}",
                description=pattern.pattern_description,
                confidence=0.8,
                impact_estimate=min(pattern.frequency / 100, 1.0),  # Scale by frequency
                evidence={
                    'pattern_type': pattern.pattern_type,
                    'frequency': pattern.frequency,
                    'examples': pattern.examples
                },
                recommendations=pattern.suggested_actions,
                auto_implementable=False,
                implementation_complexity="medium"
            )
            insights.append(insight)
        
        return insights
    
    def _generate_workflow_insights(self, tasks: List[WorkflowTask]) -> List[LearningInsight]:
        """Generate insights from workflow analysis."""
        insights = []
        
        if not tasks:
            return insights
        
        # Analyze task completion rates
        completed_tasks = [task for task in tasks if task.status.value == "completed"]
        completion_rate = len(completed_tasks) / len(tasks)
        
        if completion_rate < 0.8:
            insight = LearningInsight(
                id=f"workflow_completion_{datetime.now().timestamp()}",
                category=LearningCategory.WORKFLOW_OPTIMIZATION,
                title="Low Task Completion Rate",
                description=f"Task completion rate is {completion_rate:.2f}, below optimal threshold",
                confidence=0.9,
                impact_estimate=0.7,
                evidence={
                    'completion_rate': completion_rate,
                    'total_tasks': len(tasks),
                    'completed_tasks': len(completed_tasks)
                },
                recommendations=[
                    "Review task assignment algorithm",
                    "Investigate task complexity and user workload",
                    "Consider providing additional user training",
                    "Review task prioritization strategy"
                ],
                auto_implementable=False,
                implementation_complexity="high"
            )
            insights.append(insight)
        
        # Analyze overdue tasks
        overdue_tasks = [
            task for task in tasks 
            if task.due_date and task.due_date < datetime.now() and task.status.value != "completed"
        ]
        
        if overdue_tasks:
            overdue_rate = len(overdue_tasks) / len(tasks)
            if overdue_rate > 0.2:
                insight = LearningInsight(
                    id=f"workflow_overdue_{datetime.now().timestamp()}",
                    category=LearningCategory.WORKFLOW_OPTIMIZATION,
                    title="High Overdue Task Rate",
                    description=f"Overdue task rate is {overdue_rate:.2f}, indicating scheduling issues",
                    confidence=0.8,
                    impact_estimate=0.6,
                    evidence={
                        'overdue_rate': overdue_rate,
                        'overdue_count': len(overdue_tasks)
                    },
                    recommendations=[
                        "Review task time estimation accuracy",
                        "Adjust due date calculation algorithm",
                        "Consider workload rebalancing",
                        "Implement early warning system for at-risk tasks"
                    ],
                    auto_implementable=True,
                    implementation_complexity="medium"
                )
                insights.append(insight)
        
        return insights
    
    def _generate_template_insights(self, processing_results: List[Dict[str, Any]]) -> List[LearningInsight]:
        """Generate template optimization insights."""
        insights = []
        
        # Analyze template usage patterns
        template_usage = defaultdict(int)
        template_success = defaultdict(int)
        
        for result in processing_results:
            template_id = result.get('template_id')
            if template_id:
                template_usage[template_id] += 1
                
                # Check overall success
                field_results = result.get('field_results', {})
                successful_fields = sum(
                    1 for field_data in field_results.values()
                    if field_data.get('is_valid', False)
                )
                
                if field_results and successful_fields / len(field_results) > 0.8:
                    template_success[template_id] += 1
        
        # Generate insights for templates with low success rates
        for template_id, usage_count in template_usage.items():
            if usage_count < 5:  # Skip rarely used templates
                continue
            
            success_count = template_success.get(template_id, 0)
            success_rate = success_count / usage_count
            
            if success_rate < 0.7:
                insight = LearningInsight(
                    id=f"template_performance_{template_id}_{datetime.now().timestamp()}",
                    category=LearningCategory.TEMPLATE_OPTIMIZATION,
                    title=f"Template Performance Issue: {template_id}",
                    description=f"Template '{template_id}' has low success rate: {success_rate:.2f}",
                    confidence=0.8,
                    impact_estimate=min(usage_count / 100, 1.0),
                    evidence={
                        'template_id': template_id,
                        'success_rate': success_rate,
                        'usage_count': usage_count
                    },
                    recommendations=[
                        f"Review field definitions in template '{template_id}'",
                        "Verify bounding box accuracy",
                        "Consider template versioning and A/B testing",
                        "Gather user feedback on template usability"
                    ],
                    auto_implementable=False,
                    implementation_complexity="medium"
                )
                insights.append(insight)
        
        return insights
    
    def implement_insight(self, insight_id: str, implementation_notes: str = "") -> bool:
        """Mark an insight as implemented.
        
        Args:
            insight_id: ID of insight to implement
            implementation_notes: Notes about implementation
            
        Returns:
            True if successful
        """
        insight = next((ins for ins in self.insights if ins.id == insight_id), None)
        if not insight:
            return False
        
        insight.status = "implemented"
        
        # Record implementation
        self.implementation_history.append({
            'insight_id': insight_id,
            'implementation_date': datetime.now(),
            'notes': implementation_notes,
            'category': insight.category.value,
            'expected_impact': insight.impact_estimate
        })
        
        return True
    
    def get_analytics_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive analytics dashboard data."""
        # Categorize insights
        insights_by_category = defaultdict(list)
        for insight in self.insights:
            insights_by_category[insight.category.value].append(insight)
        
        # Calculate impact metrics
        total_potential_impact = sum(insight.impact_estimate for insight in self.insights)
        implemented_impact = sum(
            insight.impact_estimate for insight in self.insights 
            if insight.status == "implemented"
        )
        
        return {
            'insights_summary': {
                'total_insights': len(self.insights),
                'by_category': {
                    category: len(insights) 
                    for category, insights in insights_by_category.items()
                },
                'by_status': {
                    status: len([ins for ins in self.insights if ins.status == status])
                    for status in ['new', 'reviewed', 'implemented', 'rejected']
                }
            },
            'impact_analysis': {
                'total_potential_impact': total_potential_impact,
                'implemented_impact': implemented_impact,
                'impact_realization_rate': implemented_impact / total_potential_impact if total_potential_impact > 0 else 0
            },
            'recent_insights': [
                {
                    'id': insight.id,
                    'title': insight.title,
                    'category': insight.category.value,
                    'confidence': insight.confidence,
                    'impact_estimate': insight.impact_estimate,
                    'created_date': insight.created_date.isoformat()
                }
                for insight in sorted(self.insights, key=lambda x: x.created_date, reverse=True)[:10]
            ],
            'implementation_history': self.implementation_history[-20:],  # Last 20 implementations
            'error_patterns': [
                {
                    'pattern_type': pattern.pattern_type,
                    'description': pattern.pattern_description,
                    'frequency': pattern.frequency
                }
                for pattern in self.pattern_detector.error_patterns[-10:]  # Last 10 patterns
            ]
        }
    
    def export_insights(self, file_path: str) -> None:
        """Export insights and analytics to file."""
        export_data = {
            'insights': [
                {
                    'id': insight.id,
                    'category': insight.category.value,
                    'title': insight.title,
                    'description': insight.description,
                    'confidence': insight.confidence,
                    'impact_estimate': insight.impact_estimate,
                    'evidence': insight.evidence,
                    'recommendations': insight.recommendations,
                    'auto_implementable': insight.auto_implementable,
                    'implementation_complexity': insight.implementation_complexity,
                    'status': insight.status,
                    'created_date': insight.created_date.isoformat()
                }
                for insight in self.insights
            ],
            'implementation_history': self.implementation_history,
            'error_patterns': [
                {
                    'pattern_type': pattern.pattern_type,
                    'pattern_description': pattern.pattern_description,
                    'frequency': pattern.frequency,
                    'examples': pattern.examples,
                    'suggested_actions': pattern.suggested_actions
                }
                for pattern in self.pattern_detector.error_patterns
            ],
            'analytics_dashboard': self.get_analytics_dashboard(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)