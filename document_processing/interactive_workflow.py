"""
Interactive Workflow System

Human-in-the-loop processing system with task management,
approval workflows, and collaborative features.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import uuid

from .field_types import FieldExtraction, FormField
from .issue_resolution import DocumentIssue, ResolutionAction
from .exception_handling import ProcessingException


class TaskType(Enum):
    """Types of workflow tasks."""
    FIELD_VALIDATION = "field_validation"
    MANUAL_EXTRACTION = "manual_extraction"
    QUALITY_REVIEW = "quality_review"
    BUSINESS_RULE_REVIEW = "business_rule_review"
    EXCEPTION_RESOLUTION = "exception_resolution"
    TEMPLATE_ADJUSTMENT = "template_adjustment"
    DOCUMENT_CLASSIFICATION = "document_classification"
    BATCH_APPROVAL = "batch_approval"


class TaskPriority(Enum):
    """Priority levels for tasks."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskStatus(Enum):
    """Status of workflow tasks."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    COMPLETED = "completed"
    REJECTED = "rejected"
    ESCALATED = "escalated"


class UserRole(Enum):
    """User roles in the workflow system."""
    OPERATOR = "operator"
    REVIEWER = "reviewer"
    SUPERVISOR = "supervisor"
    ADMIN = "admin"
    SPECIALIST = "specialist"


@dataclass
class WorkflowUser:
    """User in the workflow system."""
    id: str
    name: str
    email: str
    role: UserRole
    skills: List[str] = field(default_factory=list)
    active: bool = True
    max_concurrent_tasks: int = 5
    current_task_count: int = 0
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class TaskAction:
    """Action taken on a task."""
    action_type: str
    description: str
    timestamp: datetime
    user_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None


@dataclass
class WorkflowTask:
    """Represents a task in the interactive workflow."""
    id: str
    task_type: TaskType
    title: str
    description: str
    priority: TaskPriority
    
    # Assignment
    assigned_to: Optional[str] = None
    assigned_by: Optional[str] = None
    assignment_date: Optional[datetime] = None
    
    # Status and timing
    status: TaskStatus = TaskStatus.PENDING
    created_date: datetime = field(default_factory=datetime.now)
    due_date: Optional[datetime] = None
    completed_date: Optional[datetime] = None
    
    # Task data
    document_id: Optional[str] = None
    field_name: Optional[str] = None
    current_value: Any = None
    expected_value: Any = None
    context_data: Dict[str, Any] = field(default_factory=dict)
    
    # Actions and history
    actions: List[TaskAction] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    estimated_time: Optional[int] = None  # minutes
    actual_time: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            'id': self.id,
            'task_type': self.task_type.value,
            'title': self.title,
            'description': self.description,
            'priority': self.priority.value,
            'assigned_to': self.assigned_to,
            'status': self.status.value,
            'created_date': self.created_date.isoformat(),
            'due_date': self.due_date.isoformat() if self.due_date else None,
            'document_id': self.document_id,
            'field_name': self.field_name,
            'current_value': self.current_value,
            'context_data': self.context_data,
            'estimated_time': self.estimated_time,
            'actual_time': self.actual_time,
            'tags': self.tags,
            'actions': [
                {
                    'action_type': action.action_type,
                    'description': action.description,
                    'timestamp': action.timestamp.isoformat(),
                    'user_id': action.user_id,
                    'notes': action.notes
                }
                for action in self.actions
            ],
            'result': self.result
        }


class TaskAssigner:
    """Intelligent task assignment system."""
    
    def __init__(self):
        """Initialize task assigner."""
        self.assignment_rules = self._load_assignment_rules()
    
    def assign_task(
        self,
        task: WorkflowTask,
        users: List[WorkflowUser],
        workload_balance: bool = True
    ) -> Optional[str]:
        """Assign a task to the most suitable user.
        
        Args:
            task: Task to assign
            users: Available users
            workload_balance: Whether to consider workload balancing
            
        Returns:
            User ID of assigned user, or None if no suitable user found
        """
        # Filter users by role requirements
        suitable_users = self._filter_users_by_role(task, users)
        
        if not suitable_users:
            return None
        
        # Filter by skills if needed
        if task.tags:
            suitable_users = self._filter_users_by_skills(task, suitable_users)
        
        # Filter by availability
        available_users = [
            user for user in suitable_users
            if user.active and user.current_task_count < user.max_concurrent_tasks
        ]
        
        if not available_users:
            return None
        
        # Choose best user based on criteria
        best_user = self._select_best_user(task, available_users, workload_balance)
        
        return best_user.id if best_user else None
    
    def _filter_users_by_role(self, task: WorkflowTask, users: List[WorkflowUser]) -> List[WorkflowUser]:
        """Filter users by role requirements."""
        role_requirements = {
            TaskType.FIELD_VALIDATION: [UserRole.OPERATOR, UserRole.REVIEWER],
            TaskType.MANUAL_EXTRACTION: [UserRole.OPERATOR, UserRole.SPECIALIST],
            TaskType.QUALITY_REVIEW: [UserRole.REVIEWER, UserRole.SUPERVISOR],
            TaskType.BUSINESS_RULE_REVIEW: [UserRole.SUPERVISOR, UserRole.ADMIN],
            TaskType.EXCEPTION_RESOLUTION: [UserRole.SPECIALIST, UserRole.ADMIN],
            TaskType.TEMPLATE_ADJUSTMENT: [UserRole.SPECIALIST, UserRole.ADMIN],
            TaskType.DOCUMENT_CLASSIFICATION: [UserRole.OPERATOR, UserRole.SPECIALIST],
            TaskType.BATCH_APPROVAL: [UserRole.SUPERVISOR, UserRole.ADMIN]
        }
        
        required_roles = role_requirements.get(task.task_type, [UserRole.OPERATOR])
        return [user for user in users if user.role in required_roles]
    
    def _filter_users_by_skills(self, task: WorkflowTask, users: List[WorkflowUser]) -> List[WorkflowUser]:
        """Filter users by required skills."""
        required_skills = set(task.tags)
        return [
            user for user in users
            if not required_skills or required_skills.intersection(set(user.skills))
        ]
    
    def _select_best_user(
        self,
        task: WorkflowTask,
        users: List[WorkflowUser],
        workload_balance: bool
    ) -> Optional[WorkflowUser]:
        """Select the best user for the task."""
        if not users:
            return None
        
        # Score users based on multiple criteria
        user_scores = []
        
        for user in users:
            score = 0
            
            # Performance score
            if 'accuracy' in user.performance_metrics:
                score += user.performance_metrics['accuracy'] * 40
            
            # Workload balance
            if workload_balance:
                workload_factor = 1 - (user.current_task_count / user.max_concurrent_tasks)
                score += workload_factor * 30
            
            # Skill match
            if task.tags:
                skill_match = len(set(task.tags).intersection(set(user.skills))) / len(task.tags)
                score += skill_match * 20
            
            # Priority experience (specialists handle high priority)
            if task.priority in [TaskPriority.CRITICAL, TaskPriority.HIGH]:
                if user.role in [UserRole.SPECIALIST, UserRole.SUPERVISOR, UserRole.ADMIN]:
                    score += 10
            
            user_scores.append((user, score))
        
        # Return user with highest score
        user_scores.sort(key=lambda x: x[1], reverse=True)
        return user_scores[0][0]
    
    def _load_assignment_rules(self) -> Dict[str, Any]:
        """Load assignment rules configuration."""
        return {
            'max_task_load_factor': 0.8,
            'skill_match_weight': 0.3,
            'performance_weight': 0.4,
            'workload_weight': 0.3
        }


class WorkflowEngine:
    """Main workflow engine for managing interactive tasks."""
    
    def __init__(self):
        """Initialize workflow engine."""
        self.tasks: Dict[str, WorkflowTask] = {}
        self.users: Dict[str, WorkflowUser] = {}
        self.task_assigner = TaskAssigner()
        self.workflows: Dict[str, List[str]] = {}  # workflow_id -> task_ids
        
        # Statistics
        self.stats = {
            'tasks_created': 0,
            'tasks_completed': 0,
            'average_completion_time': 0,
            'user_performance': {},
            'task_type_metrics': {}
        }
    
    def register_user(self, user: WorkflowUser) -> None:
        """Register a user in the workflow system."""
        self.users[user.id] = user
    
    def create_task_from_issue(self, issue: DocumentIssue) -> WorkflowTask:
        """Create a workflow task from a document issue.
        
        Args:
            issue: Document issue to create task from
            
        Returns:
            Created workflow task
        """
        # Determine task type based on issue type
        task_type_mapping = {
            'low_confidence': TaskType.FIELD_VALIDATION,
            'extraction_failure': TaskType.MANUAL_EXTRACTION,
            'validation_error': TaskType.QUALITY_REVIEW,
            'business_rule_violation': TaskType.BUSINESS_RULE_REVIEW,
            'quality_issue': TaskType.QUALITY_REVIEW
        }
        
        task_type = task_type_mapping.get(
            issue.issue_type.value,
            TaskType.QUALITY_REVIEW
        )
        
        # Determine priority
        priority_mapping = {
            'critical': TaskPriority.CRITICAL,
            'high': TaskPriority.HIGH,
            'medium': TaskPriority.MEDIUM,
            'low': TaskPriority.LOW
        }
        
        priority = priority_mapping.get(
            issue.severity.value,
            TaskPriority.MEDIUM
        )
        
        task = WorkflowTask(
            id=f"task_{uuid.uuid4().hex[:8]}",
            task_type=task_type,
            title=issue.title,
            description=issue.description,
            priority=priority,
            context_data={
                'issue_id': issue.id,
                'affected_fields': issue.affected_fields,
                'field_data': issue.field_data,
                'suggested_actions': [action.description for action in issue.suggested_actions]
            }
        )
        
        # Set field-specific data if available
        if issue.affected_fields:
            task.field_name = issue.affected_fields[0]
            if issue.field_data and task.field_name in issue.field_data:
                task.current_value = issue.field_data[task.field_name]
        
        # Set estimated completion time
        task.estimated_time = self._estimate_task_time(task)
        
        # Set due date based on priority
        task.due_date = self._calculate_due_date(task)
        
        return self.add_task(task)
    
    def create_task_from_exception(self, exception: ProcessingException) -> WorkflowTask:
        """Create a workflow task from a processing exception.
        
        Args:
            exception: Processing exception to create task from
            
        Returns:
            Created workflow task
        """
        task = WorkflowTask(
            id=f"task_{uuid.uuid4().hex[:8]}",
            task_type=TaskType.EXCEPTION_RESOLUTION,
            title=f"Exception Resolution: {exception.title}",
            description=exception.description,
            priority=TaskPriority.HIGH,  # Exceptions are typically high priority
            document_id=exception.context.document_id,
            field_name=exception.context.field_name,
            context_data={
                'exception_id': exception.id,
                'exception_type': exception.exception_type.value,
                'recovery_actions': [
                    action.description for action in exception.recovery_actions
                ],
                'context': {
                    'template_id': exception.context.template_id,
                    'processing_step': exception.context.processing_step
                }
            }
        )
        
        task.estimated_time = 30  # Exceptions typically take longer
        task.due_date = datetime.now() + timedelta(hours=4)  # Urgent resolution
        
        return self.add_task(task)
    
    def add_task(self, task: WorkflowTask) -> WorkflowTask:
        """Add a task to the workflow system."""
        self.tasks[task.id] = task
        self.stats['tasks_created'] += 1
        
        # Auto-assign if possible
        if self.users:
            assigned_user_id = self.task_assigner.assign_task(
                task, list(self.users.values())
            )
            if assigned_user_id:
                self.assign_task(task.id, assigned_user_id)
        
        return task
    
    def assign_task(self, task_id: str, user_id: str, assigned_by: Optional[str] = None) -> bool:
        """Assign a task to a user.
        
        Args:
            task_id: ID of task to assign
            user_id: ID of user to assign to
            assigned_by: ID of user making the assignment
            
        Returns:
            True if assignment successful
        """
        if task_id not in self.tasks or user_id not in self.users:
            return False
        
        task = self.tasks[task_id]
        user = self.users[user_id]
        
        # Check if user can take more tasks
        if user.current_task_count >= user.max_concurrent_tasks:
            return False
        
        # Update task
        task.assigned_to = user_id
        task.assigned_by = assigned_by
        task.assignment_date = datetime.now()
        task.status = TaskStatus.ASSIGNED
        
        # Update user workload
        user.current_task_count += 1
        
        # Add action
        action = TaskAction(
            action_type="assigned",
            description=f"Task assigned to {user.name}",
            timestamp=datetime.now(),
            user_id=assigned_by or "system"
        )
        task.actions.append(action)
        
        return True
    
    def start_task(self, task_id: str, user_id: str) -> bool:
        """Start working on a task.
        
        Args:
            task_id: ID of task to start
            user_id: ID of user starting the task
            
        Returns:
            True if task started successfully
        """
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        if task.assigned_to != user_id:
            return False
        
        task.status = TaskStatus.IN_PROGRESS
        
        action = TaskAction(
            action_type="started",
            description="Task started",
            timestamp=datetime.now(),
            user_id=user_id
        )
        task.actions.append(action)
        
        return True
    
    def complete_task(
        self,
        task_id: str,
        user_id: str,
        result: Dict[str, Any],
        notes: Optional[str] = None
    ) -> bool:
        """Complete a task with results.
        
        Args:
            task_id: ID of task to complete
            user_id: ID of user completing the task
            result: Task completion result
            notes: Optional completion notes
            
        Returns:
            True if task completed successfully
        """
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        if task.assigned_to != user_id:
            return False
        
        task.status = TaskStatus.COMPLETED
        task.completed_date = datetime.now()
        task.result = result
        
        # Calculate actual time
        if task.assignment_date:
            task.actual_time = int(
                (task.completed_date - task.assignment_date).total_seconds() / 60
            )
        
        # Update user workload
        if user_id in self.users:
            self.users[user_id].current_task_count -= 1
        
        # Add action
        action = TaskAction(
            action_type="completed",
            description="Task completed",
            timestamp=datetime.now(),
            user_id=user_id,
            data=result,
            notes=notes
        )
        task.actions.append(action)
        
        # Update statistics
        self.stats['tasks_completed'] += 1
        self._update_performance_metrics(user_id, task)
        
        return True
    
    def reject_task(
        self,
        task_id: str,
        user_id: str,
        reason: str,
        reassign: bool = True
    ) -> bool:
        """Reject a task and optionally reassign.
        
        Args:
            task_id: ID of task to reject
            user_id: ID of user rejecting the task
            reason: Reason for rejection
            reassign: Whether to automatically reassign
            
        Returns:
            True if task rejected successfully
        """
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        if task.assigned_to != user_id:
            return False
        
        task.status = TaskStatus.REJECTED
        task.assigned_to = None
        task.assignment_date = None
        
        # Update user workload
        if user_id in self.users:
            self.users[user_id].current_task_count -= 1
        
        # Add action
        action = TaskAction(
            action_type="rejected",
            description=f"Task rejected: {reason}",
            timestamp=datetime.now(),
            user_id=user_id,
            notes=reason
        )
        task.actions.append(action)
        
        # Try to reassign if requested
        if reassign:
            available_users = [
                user for user in self.users.values()
                if user.id != user_id  # Exclude the user who rejected it
            ]
            assigned_user_id = self.task_assigner.assign_task(task, available_users)
            if assigned_user_id:
                self.assign_task(task_id, assigned_user_id)
            else:
                task.status = TaskStatus.ESCALATED
        
        return True
    
    def get_user_tasks(self, user_id: str, status_filter: Optional[List[TaskStatus]] = None) -> List[WorkflowTask]:
        """Get tasks for a specific user.
        
        Args:
            user_id: ID of user
            status_filter: Optional status filter
            
        Returns:
            List of tasks for the user
        """
        user_tasks = [
            task for task in self.tasks.values()
            if task.assigned_to == user_id
        ]
        
        if status_filter:
            user_tasks = [
                task for task in user_tasks
                if task.status in status_filter
            ]
        
        return user_tasks
    
    def get_pending_tasks(self, priority_filter: Optional[List[TaskPriority]] = None) -> List[WorkflowTask]:
        """Get all pending tasks.
        
        Args:
            priority_filter: Optional priority filter
            
        Returns:
            List of pending tasks
        """
        pending_tasks = [
            task for task in self.tasks.values()
            if task.status == TaskStatus.PENDING
        ]
        
        if priority_filter:
            pending_tasks = [
                task for task in pending_tasks
                if task.priority in priority_filter
            ]
        
        # Sort by priority and creation date
        priority_order = {
            TaskPriority.CRITICAL: 0,
            TaskPriority.HIGH: 1,
            TaskPriority.MEDIUM: 2,
            TaskPriority.LOW: 3
        }
        
        pending_tasks.sort(key=lambda t: (priority_order[t.priority], t.created_date))
        
        return pending_tasks
    
    def _estimate_task_time(self, task: WorkflowTask) -> int:
        """Estimate completion time for a task in minutes."""
        base_times = {
            TaskType.FIELD_VALIDATION: 5,
            TaskType.MANUAL_EXTRACTION: 10,
            TaskType.QUALITY_REVIEW: 15,
            TaskType.BUSINESS_RULE_REVIEW: 20,
            TaskType.EXCEPTION_RESOLUTION: 30,
            TaskType.TEMPLATE_ADJUSTMENT: 45,
            TaskType.DOCUMENT_CLASSIFICATION: 8,
            TaskType.BATCH_APPROVAL: 25
        }
        
        base_time = base_times.get(task.task_type, 15)
        
        # Adjust for priority
        if task.priority == TaskPriority.CRITICAL:
            base_time *= 1.5  # More careful review needed
        elif task.priority == TaskPriority.LOW:
            base_time *= 0.8  # Simpler cases
        
        return int(base_time)
    
    def _calculate_due_date(self, task: WorkflowTask) -> datetime:
        """Calculate due date based on task priority."""
        now = datetime.now()
        
        if task.priority == TaskPriority.CRITICAL:
            return now + timedelta(hours=2)
        elif task.priority == TaskPriority.HIGH:
            return now + timedelta(hours=8)
        elif task.priority == TaskPriority.MEDIUM:
            return now + timedelta(days=1)
        else:  # LOW
            return now + timedelta(days=3)
    
    def _update_performance_metrics(self, user_id: str, task: WorkflowTask) -> None:
        """Update user performance metrics."""
        if user_id not in self.users:
            return
        
        user = self.users[user_id]
        
        # Initialize metrics if needed
        if 'tasks_completed' not in user.performance_metrics:
            user.performance_metrics['tasks_completed'] = 0
            user.performance_metrics['average_time'] = 0
            user.performance_metrics['on_time_completion'] = 1.0
        
        # Update completion count
        user.performance_metrics['tasks_completed'] += 1
        
        # Update average time
        if task.actual_time:
            prev_avg = user.performance_metrics['average_time']
            count = user.performance_metrics['tasks_completed']
            new_avg = (prev_avg * (count - 1) + task.actual_time) / count
            user.performance_metrics['average_time'] = new_avg
        
        # Update on-time completion rate
        if task.due_date and task.completed_date:
            on_time = task.completed_date <= task.due_date
            prev_rate = user.performance_metrics['on_time_completion']
            count = user.performance_metrics['tasks_completed']
            new_rate = (prev_rate * (count - 1) + (1 if on_time else 0)) / count
            user.performance_metrics['on_time_completion'] = new_rate
    
    def get_workflow_statistics(self) -> Dict[str, Any]:
        """Get comprehensive workflow statistics."""
        # Calculate completion rates by task type
        task_type_stats = {}
        for task in self.tasks.values():
            task_type = task.task_type.value
            if task_type not in task_type_stats:
                task_type_stats[task_type] = {'total': 0, 'completed': 0}
            
            task_type_stats[task_type]['total'] += 1
            if task.status == TaskStatus.COMPLETED:
                task_type_stats[task_type]['completed'] += 1
        
        # Calculate average completion times
        completed_tasks = [t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED and t.actual_time]
        avg_completion_time = sum(t.actual_time for t in completed_tasks if t.actual_time is not None) / len(completed_tasks) if completed_tasks else 0
        
        return {
            'task_statistics': self.stats,
            'task_type_performance': task_type_stats,
            'average_completion_time': avg_completion_time,
            'active_tasks': len([t for t in self.tasks.values() if t.status in [TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]]),
            'pending_tasks': len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING]),
            'overdue_tasks': len([
                t for t in self.tasks.values()
                if t.due_date and t.due_date < datetime.now() and t.status != TaskStatus.COMPLETED
            ]),
            'user_performance': {
                user_id: user.performance_metrics
                for user_id, user in self.users.items()
            }
        }
    
    def export_tasks(self, file_path: str, date_range: Optional[tuple] = None) -> None:
        """Export tasks to JSON file.
        
        Args:
            file_path: Path to export file
            date_range: Optional date range (start_date, end_date)
        """
        tasks_to_export = list(self.tasks.values())
        
        if date_range:
            start_date, end_date = date_range
            tasks_to_export = [
                task for task in tasks_to_export
                if start_date <= task.created_date <= end_date
            ]
        
        export_data = {
            'tasks': [task.to_dict() for task in tasks_to_export],
            'users': {user_id: {
                'name': user.name,
                'role': user.role.value,
                'performance_metrics': user.performance_metrics
            } for user_id, user in self.users.items()},
            'statistics': self.get_workflow_statistics(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)