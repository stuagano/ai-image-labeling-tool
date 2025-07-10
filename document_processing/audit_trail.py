"""
Audit Trail System

Comprehensive audit trail system for complete traceability of all
document processing decisions, changes, and system interactions.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import hashlib
import uuid
from pathlib import Path

from .field_types import FieldExtraction


class AuditEventType(Enum):
    """Types of audit events."""
    DOCUMENT_UPLOADED = "document_uploaded"
    DOCUMENT_PROCESSED = "document_processed"
    FIELD_EXTRACTED = "field_extracted"
    FIELD_VALIDATED = "field_validated"
    FIELD_CORRECTED = "field_corrected"
    BUSINESS_RULE_APPLIED = "business_rule_applied"
    BUSINESS_RULE_VIOLATED = "business_rule_violated"
    EXCEPTION_OCCURRED = "exception_occurred"
    EXCEPTION_RESOLVED = "exception_resolved"
    TASK_CREATED = "task_created"
    TASK_ASSIGNED = "task_assigned"
    TASK_COMPLETED = "task_completed"
    TEMPLATE_APPLIED = "template_applied"
    TEMPLATE_MODIFIED = "template_modified"
    USER_LOGIN = "user_login"
    USER_ACTION = "user_action"
    SYSTEM_CONFIGURATION_CHANGED = "system_configuration_changed"
    DATA_EXPORTED = "data_exported"
    BACKUP_CREATED = "backup_created"


class AuditLevel(Enum):
    """Audit logging levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    DEBUG = "debug"


@dataclass
class AuditContext:
    """Context information for audit events."""
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    document_id: Optional[str] = None
    template_id: Optional[str] = None
    batch_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    system_version: Optional[str] = None


@dataclass
class AuditEvent:
    """Represents a single audit event."""
    id: str
    timestamp: datetime
    event_type: AuditEventType
    level: AuditLevel
    description: str
    
    # Context
    context: AuditContext = field(default_factory=AuditContext)
    
    # Data
    before_state: Optional[Dict[str, Any]] = None
    after_state: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Integrity
    checksum: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'level': self.level.value,
            'description': self.description,
            'context': {
                'session_id': self.context.session_id,
                'user_id': self.context.user_id,
                'document_id': self.context.document_id,
                'template_id': self.context.template_id,
                'batch_id': self.context.batch_id,
                'ip_address': self.context.ip_address,
                'user_agent': self.context.user_agent,
                'system_version': self.context.system_version
            },
            'before_state': self.before_state,
            'after_state': self.after_state,
            'metadata': self.metadata,
            'checksum': self.checksum
        }
    
    def calculate_checksum(self) -> str:
        """Calculate checksum for event integrity."""
        event_data = {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'description': self.description,
            'before_state': self.before_state,
            'after_state': self.after_state
        }
        
        event_string = json.dumps(event_data, sort_keys=True)
        return hashlib.sha256(event_string.encode()).hexdigest()


@dataclass
class AuditChain:
    """Represents a chain of related audit events."""
    chain_id: str
    document_id: str
    start_timestamp: datetime
    end_timestamp: Optional[datetime] = None
    events: List[str] = field(default_factory=list)  # Event IDs
    status: str = "active"  # active, completed, error
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit chain to dictionary."""
        return {
            'chain_id': self.chain_id,
            'document_id': self.document_id,
            'start_timestamp': self.start_timestamp.isoformat(),
            'end_timestamp': self.end_timestamp.isoformat() if self.end_timestamp else None,
            'events': self.events,
            'status': self.status
        }


class AuditLogger:
    """Main audit logging system."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """Initialize audit logger.
        
        Args:
            storage_path: Path for storing audit files
        """
        self.storage_path = Path(storage_path) if storage_path else Path("audit_logs")
        self.storage_path.mkdir(exist_ok=True)
        
        self.events: Dict[str, AuditEvent] = {}
        self.chains: Dict[str, AuditChain] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.config = {
            'auto_flush_interval': 100,  # Events before auto-flush
            'enable_integrity_checks': True,
            'retention_days': 365,
            'compression_enabled': True
        }
        
        self.event_count = 0
    
    def start_session(self, user_id: str, ip_address: Optional[str] = None, user_agent: Optional[str] = None) -> str:
        """Start an audit session.
        
        Args:
            user_id: ID of the user
            ip_address: User's IP address
            user_agent: User's user agent string
            
        Returns:
            Session ID
        """
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        
        self.active_sessions[session_id] = {
            'user_id': user_id,
            'start_time': datetime.now(),
            'ip_address': ip_address,
            'user_agent': user_agent,
            'event_count': 0
        }
        
        # Log session start
        self.log_event(
            event_type=AuditEventType.USER_LOGIN,
            level=AuditLevel.MEDIUM,
            description=f"User {user_id} started session",
            context=AuditContext(
                session_id=session_id,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent
            )
        )
        
        return session_id
    
    def end_session(self, session_id: str) -> None:
        """End an audit session.
        
        Args:
            session_id: ID of session to end
        """
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        duration = datetime.now() - session['start_time']
        
        self.log_event(
            event_type=AuditEventType.USER_ACTION,
            level=AuditLevel.MEDIUM,
            description=f"Session ended after {duration}",
            context=AuditContext(
                session_id=session_id,
                user_id=session['user_id']
            ),
            metadata={
                'session_duration_seconds': duration.total_seconds(),
                'events_in_session': session['event_count']
            }
        )
        
        del self.active_sessions[session_id]
    
    def start_document_chain(self, document_id: str, context: Optional[AuditContext] = None) -> str:
        """Start an audit chain for document processing.
        
        Args:
            document_id: ID of the document
            context: Audit context
            
        Returns:
            Chain ID
        """
        chain_id = f"chain_{document_id}_{uuid.uuid4().hex[:8]}"
        
        chain = AuditChain(
            chain_id=chain_id,
            document_id=document_id,
            start_timestamp=datetime.now()
        )
        
        self.chains[chain_id] = chain
        
        # Log chain start
        self.log_event(
            event_type=AuditEventType.DOCUMENT_UPLOADED,
            level=AuditLevel.MEDIUM,
            description=f"Started processing chain for document {document_id}",
            context=context or AuditContext(),
            metadata={'chain_id': chain_id}
        )
        
        return chain_id
    
    def log_event(
        self,
        event_type: AuditEventType,
        level: AuditLevel,
        description: str,
        context: Optional[AuditContext] = None,
        before_state: Optional[Dict[str, Any]] = None,
        after_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log an audit event.
        
        Args:
            event_type: Type of event
            level: Audit level
            description: Event description
            context: Audit context
            before_state: State before the event
            after_state: State after the event
            metadata: Additional metadata
            
        Returns:
            Event ID
        """
        event_id = f"evt_{uuid.uuid4().hex[:12]}"
        
        event = AuditEvent(
            id=event_id,
            timestamp=datetime.now(),
            event_type=event_type,
            level=level,
            description=description,
            context=context or AuditContext(),
            before_state=before_state,
            after_state=after_state,
            metadata=metadata or {}
        )
        
        # Calculate checksum if enabled
        if self.config['enable_integrity_checks']:
            event.checksum = event.calculate_checksum()
        
        self.events[event_id] = event
        self.event_count += 1
        
        # Update session event count
        if event.context.session_id and event.context.session_id in self.active_sessions:
            self.active_sessions[event.context.session_id]['event_count'] += 1
        
        # Add to chain if document context exists
        if event.context.document_id:
            active_chain = self._find_active_chain(event.context.document_id)
            if active_chain:
                active_chain.events.append(event_id)
        
        # Auto-flush if needed
        if self.event_count % self.config['auto_flush_interval'] == 0:
            self.flush_to_storage()
        
        return event_id
    
    def log_field_extraction(
        self,
        field_name: str,
        extraction: FieldExtraction,
        context: AuditContext
    ) -> str:
        """Log field extraction event.
        
        Args:
            field_name: Name of the field
            extraction: Extraction result
            context: Audit context
            
        Returns:
            Event ID
        """
        return self.log_event(
            event_type=AuditEventType.FIELD_EXTRACTED,
            level=AuditLevel.LOW,
            description=f"Extracted field '{field_name}' with confidence {extraction.confidence:.2f}",
            context=context,
            after_state={
                'field_name': field_name,
                'value': extraction.value,
                'confidence': extraction.confidence,
                'is_valid': extraction.is_valid,
                'extraction_method': extraction.extraction_method,
                'validation_errors': extraction.validation_errors
            },
            metadata={
                'extraction_timestamp': extraction.timestamp.isoformat()
            }
        )
    
    def log_field_correction(
        self,
        field_name: str,
        old_value: Any,
        new_value: Any,
        user_id: str,
        context: AuditContext,
        reason: Optional[str] = None
    ) -> str:
        """Log field correction event.
        
        Args:
            field_name: Name of the field
            old_value: Original value
            new_value: Corrected value
            user_id: ID of user making correction
            context: Audit context
            reason: Reason for correction
            
        Returns:
            Event ID
        """
        return self.log_event(
            event_type=AuditEventType.FIELD_CORRECTED,
            level=AuditLevel.HIGH,
            description=f"User {user_id} corrected field '{field_name}'" + (f": {reason}" if reason else ""),
            context=context,
            before_state={
                'field_name': field_name,
                'value': old_value
            },
            after_state={
                'field_name': field_name,
                'value': new_value
            },
            metadata={
                'corrected_by': user_id,
                'correction_reason': reason
            }
        )
    
    def log_business_rule_event(
        self,
        rule_id: str,
        rule_name: str,
        violated: bool,
        context: AuditContext,
        affected_fields: Optional[List[str]] = None,
        violation_details: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log business rule event.
        
        Args:
            rule_id: ID of the business rule
            rule_name: Name of the business rule
            violated: Whether rule was violated
            context: Audit context
            affected_fields: Fields affected by the rule
            violation_details: Details of the violation
            
        Returns:
            Event ID
        """
        event_type = AuditEventType.BUSINESS_RULE_VIOLATED if violated else AuditEventType.BUSINESS_RULE_APPLIED
        level = AuditLevel.HIGH if violated else AuditLevel.LOW
        
        return self.log_event(
            event_type=event_type,
            level=level,
            description=f"Business rule '{rule_name}' {'violated' if violated else 'applied'}",
            context=context,
            metadata={
                'rule_id': rule_id,
                'rule_name': rule_name,
                'affected_fields': affected_fields or [],
                'violation_details': violation_details
            }
        )
    
    def log_template_event(
        self,
        template_id: str,
        action: str,
        context: AuditContext,
        template_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log template-related event.
        
        Args:
            template_id: ID of the template
            action: Action performed (applied, modified, created)
            context: Audit context
            template_data: Template data
            
        Returns:
            Event ID
        """
        event_type_map = {
            'applied': AuditEventType.TEMPLATE_APPLIED,
            'modified': AuditEventType.TEMPLATE_MODIFIED,
            'created': AuditEventType.TEMPLATE_MODIFIED
        }
        
        return self.log_event(
            event_type=event_type_map.get(action, AuditEventType.TEMPLATE_APPLIED),
            level=AuditLevel.MEDIUM,
            description=f"Template '{template_id}' {action}",
            context=context,
            after_state=template_data,
            metadata={
                'template_id': template_id,
                'action': action
            }
        )
    
    def log_data_export(
        self,
        export_type: str,
        file_path: str,
        record_count: int,
        context: AuditContext
    ) -> str:
        """Log data export event.
        
        Args:
            export_type: Type of export (csv, json, etc.)
            file_path: Path of exported file
            record_count: Number of records exported
            context: Audit context
            
        Returns:
            Event ID
        """
        return self.log_event(
            event_type=AuditEventType.DATA_EXPORTED,
            level=AuditLevel.HIGH,
            description=f"Exported {record_count} records to {export_type} file",
            context=context,
            metadata={
                'export_type': export_type,
                'file_path': file_path,
                'record_count': record_count
            }
        )
    
    def complete_document_chain(self, chain_id: str, status: str = "completed") -> None:
        """Complete a document processing chain.
        
        Args:
            chain_id: ID of the chain to complete
            status: Final status of the chain
        """
        if chain_id not in self.chains:
            return
        
        chain = self.chains[chain_id]
        chain.end_timestamp = datetime.now()
        chain.status = status
        
        # Log chain completion
        self.log_event(
            event_type=AuditEventType.DOCUMENT_PROCESSED,
            level=AuditLevel.MEDIUM,
            description=f"Completed processing chain {chain_id} with status {status}",
            context=AuditContext(document_id=chain.document_id),
            metadata={
                'chain_id': chain_id,
                'final_status': status,
                'event_count': len(chain.events),
                'processing_duration': (chain.end_timestamp - chain.start_timestamp).total_seconds()
            }
        )
    
    def query_events(
        self,
        event_types: Optional[List[AuditEventType]] = None,
        user_id: Optional[str] = None,
        document_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        level: Optional[AuditLevel] = None
    ) -> List[AuditEvent]:
        """Query audit events with filters.
        
        Args:
            event_types: Filter by event types
            user_id: Filter by user ID
            document_id: Filter by document ID
            start_time: Filter by start time
            end_time: Filter by end time
            level: Filter by audit level
            
        Returns:
            List of matching events
        """
        results = []
        
        for event in self.events.values():
            # Apply filters
            if event_types and event.event_type not in event_types:
                continue
            
            if user_id and event.context.user_id != user_id:
                continue
            
            if document_id and event.context.document_id != document_id:
                continue
            
            if start_time and event.timestamp < start_time:
                continue
            
            if end_time and event.timestamp > end_time:
                continue
            
            if level and event.level != level:
                continue
            
            results.append(event)
        
        # Sort by timestamp
        results.sort(key=lambda x: x.timestamp)
        return results
    
    def get_document_timeline(self, document_id: str) -> List[AuditEvent]:
        """Get complete timeline for a document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            Chronological list of events for the document
        """
        return self.query_events(document_id=document_id)
    
    def get_user_activity(self, user_id: str, time_window: Optional[timedelta] = None) -> List[AuditEvent]:
        """Get user activity within a time window.
        
        Args:
            user_id: ID of the user
            time_window: Time window for activity (defaults to 24 hours)
            
        Returns:
            List of user events
        """
        if time_window is None:
            time_window = timedelta(hours=24)
        
        start_time = datetime.now() - time_window
        
        return self.query_events(
            user_id=user_id,
            start_time=start_time
        )
    
    def verify_integrity(self, event_id: str) -> bool:
        """Verify integrity of an audit event.
        
        Args:
            event_id: ID of event to verify
            
        Returns:
            True if integrity check passes
        """
        if event_id not in self.events:
            return False
        
        event = self.events[event_id]
        
        if not event.checksum:
            return True  # No checksum to verify
        
        calculated_checksum = event.calculate_checksum()
        return calculated_checksum == event.checksum
    
    def generate_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime,
        include_user_activity: bool = True
    ) -> Dict[str, Any]:
        """Generate compliance report for a date range.
        
        Args:
            start_date: Start date for report
            end_date: End date for report
            include_user_activity: Whether to include user activity details
            
        Returns:
            Compliance report
        """
        events = self.query_events(start_time=start_date, end_time=end_date)
        
        # Categorize events
        event_summary = {}
        user_activity = {}
        document_activity = {}
        
        for event in events:
            # Event type summary
            event_type = event.event_type.value
            if event_type not in event_summary:
                event_summary[event_type] = 0
            event_summary[event_type] += 1
            
            # User activity
            if include_user_activity and event.context.user_id:
                user_id = event.context.user_id
                if user_id not in user_activity:
                    user_activity[user_id] = {'total_events': 0, 'event_types': {}}
                
                user_activity[user_id]['total_events'] += 1
                if event_type not in user_activity[user_id]['event_types']:
                    user_activity[user_id]['event_types'][event_type] = 0
                user_activity[user_id]['event_types'][event_type] += 1
            
            # Document activity
            if event.context.document_id:
                doc_id = event.context.document_id
                if doc_id not in document_activity:
                    document_activity[doc_id] = {'total_events': 0, 'processing_chain': []}
                
                document_activity[doc_id]['total_events'] += 1
                document_activity[doc_id]['processing_chain'].append({
                    'timestamp': event.timestamp.isoformat(),
                    'event_type': event_type,
                    'description': event.description
                })
        
        # Calculate integrity status
        integrity_checks = 0
        integrity_failures = 0
        
        for event in events:
            if event.checksum:
                integrity_checks += 1
                if not self.verify_integrity(event.id):
                    integrity_failures += 1
        
        return {
            'report_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'summary': {
                'total_events': len(events),
                'unique_users': len(user_activity),
                'unique_documents': len(document_activity),
                'event_type_breakdown': event_summary
            },
            'integrity_status': {
                'total_checks': integrity_checks,
                'failures': integrity_failures,
                'integrity_rate': (integrity_checks - integrity_failures) / integrity_checks if integrity_checks > 0 else 1.0
            },
            'user_activity': user_activity if include_user_activity else None,
            'document_processing': document_activity,
            'generated_at': datetime.now().isoformat()
        }
    
    def flush_to_storage(self) -> None:
        """Flush events to persistent storage."""
        if not self.events:
            return
        
        # Create daily audit file
        today = datetime.now().strftime('%Y-%m-%d')
        audit_file = self.storage_path / f"audit_{today}.json"
        
        # Prepare data for storage
        storage_data = {
            'events': [event.to_dict() for event in self.events.values()],
            'chains': [chain.to_dict() for chain in self.chains.values()],
            'flush_timestamp': datetime.now().isoformat(),
            'event_count': len(self.events)
        }
        
        # Write to file
        with open(audit_file, 'w') as f:
            json.dump(storage_data, f, indent=2)
        
        # Clear in-memory events
        self.events.clear()
        self.event_count = 0
    
    def load_from_storage(self, date: Optional[datetime] = None) -> None:
        """Load events from storage.
        
        Args:
            date: Date to load (defaults to today)
        """
        if date is None:
            date = datetime.now()
        
        date_str = date.strftime('%Y-%m-%d')
        audit_file = self.storage_path / f"audit_{date_str}.json"
        
        if not audit_file.exists():
            return
        
        with open(audit_file, 'r') as f:
            storage_data = json.load(f)
        
        # Restore events
        for event_data in storage_data.get('events', []):
            event = AuditEvent(
                id=event_data['id'],
                timestamp=datetime.fromisoformat(event_data['timestamp']),
                event_type=AuditEventType(event_data['event_type']),
                level=AuditLevel(event_data['level']),
                description=event_data['description'],
                context=AuditContext(**event_data['context']),
                before_state=event_data.get('before_state'),
                after_state=event_data.get('after_state'),
                metadata=event_data.get('metadata', {}),
                checksum=event_data.get('checksum')
            )
            self.events[event.id] = event
        
        # Restore chains
        for chain_data in storage_data.get('chains', []):
            chain = AuditChain(
                chain_id=chain_data['chain_id'],
                document_id=chain_data['document_id'],
                start_timestamp=datetime.fromisoformat(chain_data['start_timestamp']),
                end_timestamp=datetime.fromisoformat(chain_data['end_timestamp']) if chain_data.get('end_timestamp') else None,
                events=chain_data['events'],
                status=chain_data['status']
            )
            self.chains[chain.chain_id] = chain
    
    def _find_active_chain(self, document_id: str) -> Optional[AuditChain]:
        """Find active audit chain for a document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            Active audit chain or None
        """
        for chain in self.chains.values():
            if chain.document_id == document_id and chain.status == "active":
                return chain
        return None
    
    def cleanup_old_logs(self, retention_days: int = 365) -> int:
        """Clean up old audit logs.
        
        Args:
            retention_days: Number of days to retain
            
        Returns:
            Number of files deleted
        """
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        deleted_count = 0
        
        for audit_file in self.storage_path.glob("audit_*.json"):
            try:
                file_date_str = audit_file.stem.split('_')[1]
                file_date = datetime.strptime(file_date_str, '%Y-%m-%d')
                
                if file_date < cutoff_date:
                    audit_file.unlink()
                    deleted_count += 1
            except (ValueError, IndexError):
                # Skip files that don't match expected pattern
                continue
        
        return deleted_count
    
    def export_audit_data(
        self,
        file_path: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None
    ) -> None:
        """Export audit data to file.
        
        Args:
            file_path: Path for export file
            start_date: Start date filter
            end_date: End date filter
            event_types: Event type filter
        """
        events = self.query_events(
            event_types=event_types,
            start_time=start_date,
            end_time=end_date
        )
        
        export_data = {
            'export_metadata': {
                'generated_at': datetime.now().isoformat(),
                'event_count': len(events),
                'filters': {
                    'start_date': start_date.isoformat() if start_date else None,
                    'end_date': end_date.isoformat() if end_date else None,
                    'event_types': [et.value for et in event_types] if event_types else None
                }
            },
            'events': [event.to_dict() for event in events],
            'chains': [chain.to_dict() for chain in self.chains.values()]
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)