"""
Document Template Manager

Manages document templates with field definitions, validation schemas,
and template matching for automated document processing.
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from .field_types import FormField, BoundingBox, FieldType, ValidationLevel
from .field_types import (
    TextField, NumberField, DateField, EmailField, PhoneField,
    SignatureField, CheckboxField, RadioButtonField, DropdownField, TableField
)


@dataclass
class DocumentTemplate:
    """Document template with field definitions and metadata."""
    name: str
    version: str
    description: str
    fields: List[FormField]
    template_image_path: Optional[str] = None
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    tags: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.created_date is None:
            self.created_date = datetime.now()
        if self.modified_date is None:
            self.modified_date = datetime.now()
        if self.tags is None:
            self.tags = []
    
    def get_field_by_name(self, name: str) -> Optional[FormField]:
        """Get field by name."""
        for field in self.fields:
            if field.name == name:
                return field
        return None
    
    def get_fields_by_type(self, field_type: FieldType) -> List[FormField]:
        """Get all fields of a specific type."""
        return [field for field in self.fields if field.field_type == field_type]
    
    def validate_template(self) -> List[str]:
        """Validate template definition."""
        errors = []
        
        # Check for duplicate field names
        field_names = [field.name for field in self.fields]
        if len(field_names) != len(set(field_names)):
            errors.append("Duplicate field names found")
        
        # Check for overlapping bounding boxes
        for i, field1 in enumerate(self.fields):
            for j, field2 in enumerate(self.fields[i+1:], i+1):
                if self._boxes_overlap(field1.bounding_box, field2.bounding_box):
                    errors.append(f"Overlapping bounding boxes: {field1.name} and {field2.name}")
        
        return errors
    
    def _boxes_overlap(self, box1: BoundingBox, box2: BoundingBox) -> bool:
        """Check if two bounding boxes overlap."""
        return not (
            box1.left + box1.width < box2.left or
            box2.left + box2.width < box1.left or
            box1.top + box1.height < box2.top or
            box2.top + box2.height < box1.top
        )


class DocumentTemplateManager:
    """Manages document templates and provides template matching capabilities."""
    
    def __init__(self, templates_dir: str = "templates"):
        """Initialize template manager.
        
        Args:
            templates_dir: Directory to store template files
        """
        self.templates_dir = templates_dir
        os.makedirs(templates_dir, exist_ok=True)
        self._templates = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load all templates from the templates directory."""
        for filename in os.listdir(self.templates_dir):
            if filename.endswith('.json'):
                try:
                    template = self.load_template(filename[:-5])  # Remove .json extension
                    self._templates[template.name] = template
                except Exception as e:
                    print(f"Error loading template {filename}: {e}")
    
    def create_template(
        self,
        name: str,
        description: str = "",
        template_image_path: Optional[str] = None,
        version: str = "1.0.0"
    ) -> DocumentTemplate:
        """Create a new empty template.
        
        Args:
            name: Template name
            description: Template description
            template_image_path: Path to template image
            version: Template version
            
        Returns:
            New DocumentTemplate instance
        """
        template = DocumentTemplate(
            name=name,
            version=version,
            description=description,
            fields=[],
            template_image_path=template_image_path
        )
        return template
    
    def add_field_to_template(self, template: DocumentTemplate, field: FormField):
        """Add a field to a template.
        
        Args:
            template: Template to modify
            field: Field to add
        """
        # Check for duplicate names
        if template.get_field_by_name(field.name):
            raise ValueError(f"Field with name '{field.name}' already exists")
        
        template.fields.append(field)
        template.modified_date = datetime.now()
    
    def remove_field_from_template(self, template: DocumentTemplate, field_name: str):
        """Remove a field from a template.
        
        Args:
            template: Template to modify
            field_name: Name of field to remove
        """
        template.fields = [f for f in template.fields if f.name != field_name]
        template.modified_date = datetime.now()
    
    def save_template(self, template: DocumentTemplate):
        """Save template to file.
        
        Args:
            template: Template to save
        """
        template_path = os.path.join(self.templates_dir, f"{template.name}.json")
        
        # Convert template to dictionary for JSON serialization
        template_dict = self._template_to_dict(template)
        
        with open(template_path, 'w') as f:
            json.dump(template_dict, f, indent=2, default=str)
        
        self._templates[template.name] = template
    
    def load_template(self, name: str) -> DocumentTemplate:
        """Load template from file.
        
        Args:
            name: Template name
            
        Returns:
            Loaded DocumentTemplate
        """
        template_path = os.path.join(self.templates_dir, f"{name}.json")
        
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template {name} not found")
        
        with open(template_path, 'r') as f:
            template_dict = json.load(f)
        
        return self._dict_to_template(template_dict)
    
    def get_template(self, name: str) -> Optional[DocumentTemplate]:
        """Get template from memory.
        
        Args:
            name: Template name
            
        Returns:
            Template if found, None otherwise
        """
        return self._templates.get(name)
    
    def list_templates(self) -> List[str]:
        """List all available template names.
        
        Returns:
            List of template names
        """
        return list(self._templates.keys())
    
    def delete_template(self, name: str):
        """Delete a template.
        
        Args:
            name: Template name to delete
        """
        template_path = os.path.join(self.templates_dir, f"{name}.json")
        
        if os.path.exists(template_path):
            os.remove(template_path)
        
        if name in self._templates:
            del self._templates[name]
    
    def match_template(self, document_image_path: str) -> Optional[str]:
        """Match document to best template using image similarity.
        
        Args:
            document_image_path: Path to document image
            
        Returns:
            Best matching template name or None
        """
        # This would implement template matching logic
        # For now, return None (could be enhanced with computer vision)
        return None
    
    def _template_to_dict(self, template: DocumentTemplate) -> Dict[str, Any]:
        """Convert template to dictionary for JSON serialization."""
        template_dict = {
            'name': template.name,
            'version': template.version,
            'description': template.description,
            'template_image_path': template.template_image_path,
            'created_date': template.created_date.isoformat() if template.created_date else None,
            'modified_date': template.modified_date.isoformat() if template.modified_date else None,
            'tags': template.tags,
            'fields': []
        }
        
        for field in template.fields:
            field_dict = self._field_to_dict(field)
            template_dict['fields'].append(field_dict)
        
        return template_dict
    
    def _field_to_dict(self, field: FormField) -> Dict[str, Any]:
        """Convert field to dictionary for JSON serialization."""
        field_dict = {
            'name': field.name,
            'field_type': field.field_type.value,
            'bounding_box': field.bounding_box.to_dict(),
            'required': field.required,
            'label': field.label,
            'description': field.description,
            'validation_level': field.validation_level.value
        }
        
        # Add field-specific properties
        if isinstance(field, TextField):
            field_dict.update({
                'min_length': field.min_length,
                'max_length': field.max_length,
                'pattern': field.pattern,
                'case_sensitive': field.case_sensitive
            })
        elif isinstance(field, NumberField):
            field_dict.update({
                'min_value': field.min_value,
                'max_value': field.max_value,
                'decimal_places': field.decimal_places
            })
        elif isinstance(field, DateField):
            field_dict.update({
                'date_format': field.date_format,
                'min_date': field.min_date.isoformat() if field.min_date else None,
                'max_date': field.max_date.isoformat() if field.max_date else None
            })
        elif isinstance(field, EmailField):
            pass  # No additional properties
        elif isinstance(field, PhoneField):
            field_dict.update({
                'country_code': field.country_code,
                'format_pattern': field.format_pattern
            })
        elif isinstance(field, SignatureField):
            field_dict.update({
                'min_signature_area': field.min_signature_area,
                'require_ink_detection': field.require_ink_detection
            })
        elif isinstance(field, CheckboxField):
            field_dict.update({
                'default_value': field.default_value
            })
        elif isinstance(field, RadioButtonField):
            field_dict.update({
                'options': field.options,
                'allow_multiple': field.allow_multiple
            })
        elif isinstance(field, DropdownField):
            field_dict.update({
                'options': field.options,
                'allow_custom': field.allow_custom
            })
        elif isinstance(field, TableField):
            field_dict.update({
                'columns': field.columns,
                'min_rows': field.min_rows,
                'max_rows': field.max_rows
            })
        
        return field_dict
    
    def _dict_to_template(self, template_dict: Dict[str, Any]) -> DocumentTemplate:
        """Convert dictionary to template object."""
        # Parse dates
        created_date = None
        if template_dict.get('created_date'):
            created_date = datetime.fromisoformat(template_dict['created_date'])
        
        modified_date = None
        if template_dict.get('modified_date'):
            modified_date = datetime.fromisoformat(template_dict['modified_date'])
        
        # Parse fields
        fields = []
        for field_dict in template_dict.get('fields', []):
            field = self._dict_to_field(field_dict)
            fields.append(field)
        
        return DocumentTemplate(
            name=template_dict['name'],
            version=template_dict['version'],
            description=template_dict['description'],
            fields=fields,
            template_image_path=template_dict.get('template_image_path'),
            created_date=created_date,
            modified_date=modified_date,
            tags=template_dict.get('tags', [])
        )
    
    def _dict_to_field(self, field_dict: Dict[str, Any]) -> FormField:
        """Convert dictionary to field object."""
        field_type = FieldType(field_dict['field_type'])
        bounding_box = BoundingBox.from_dict(field_dict['bounding_box'])
        validation_level = ValidationLevel(field_dict.get('validation_level', 'moderate'))
        
        base_kwargs = {
            'name': field_dict['name'],
            'bounding_box': bounding_box,
            'required': field_dict.get('required', False),
            'label': field_dict.get('label', ''),
            'description': field_dict.get('description', ''),
            'validation_level': validation_level
        }
        
        if field_type == FieldType.TEXT:
            return TextField(
                **base_kwargs,
                min_length=field_dict.get('min_length'),
                max_length=field_dict.get('max_length'),
                pattern=field_dict.get('pattern'),
                case_sensitive=field_dict.get('case_sensitive', False)
            )
        elif field_type == FieldType.NUMBER:
            return NumberField(
                **base_kwargs,
                min_value=field_dict.get('min_value'),
                max_value=field_dict.get('max_value'),
                decimal_places=field_dict.get('decimal_places')
            )
        elif field_type == FieldType.DATE:
            min_date = None
            if field_dict.get('min_date'):
                min_date = datetime.fromisoformat(field_dict['min_date'])
            
            max_date = None
            if field_dict.get('max_date'):
                max_date = datetime.fromisoformat(field_dict['max_date'])
            
            return DateField(
                **base_kwargs,
                date_format=field_dict.get('date_format', '%Y-%m-%d'),
                min_date=min_date,
                max_date=max_date
            )
        elif field_type == FieldType.EMAIL:
            return EmailField(**base_kwargs)
        elif field_type == FieldType.PHONE:
            return PhoneField(
                **base_kwargs,
                country_code=field_dict.get('country_code'),
                format_pattern=field_dict.get('format_pattern')
            )
        elif field_type == FieldType.SIGNATURE:
            return SignatureField(
                **base_kwargs,
                min_signature_area=field_dict.get('min_signature_area', 0.1),
                require_ink_detection=field_dict.get('require_ink_detection', True)
            )
        elif field_type == FieldType.CHECKBOX:
            return CheckboxField(
                **base_kwargs,
                default_value=field_dict.get('default_value', False)
            )
        elif field_type == FieldType.RADIO_BUTTON:
            return RadioButtonField(
                **base_kwargs,
                options=field_dict.get('options', []),
                allow_multiple=field_dict.get('allow_multiple', False)
            )
        elif field_type == FieldType.DROPDOWN:
            return DropdownField(
                **base_kwargs,
                options=field_dict.get('options', []),
                allow_custom=field_dict.get('allow_custom', False)
            )
        elif field_type == FieldType.TABLE:
            return TableField(
                **base_kwargs,
                columns=field_dict.get('columns', []),
                min_rows=field_dict.get('min_rows', 0),
                max_rows=field_dict.get('max_rows')
            )
        else:
            # Fallback to TextField for unknown types
            return TextField(**base_kwargs)