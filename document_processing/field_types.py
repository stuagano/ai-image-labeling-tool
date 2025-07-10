"""
Field Types and Schema Definitions

Strongly typed field definitions for document processing with validation.
"""

from typing import Optional, List, Dict, Any, Union, Literal
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import re


class FieldType(Enum):
    """Enumeration of supported field types."""
    TEXT = "text"
    NUMBER = "number"
    DATE = "date"
    EMAIL = "email"
    PHONE = "phone"
    SIGNATURE = "signature"
    CHECKBOX = "checkbox"
    RADIO_BUTTON = "radio_button"
    DROPDOWN = "dropdown"
    MULTI_SELECT = "multi_select"
    TABLE = "table"
    IMAGE = "image"
    BARCODE = "barcode"
    QR_CODE = "qr_code"


class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"


@dataclass
class BoundingBox:
    """Bounding box coordinates for field extraction."""
    left: int
    top: int
    width: int
    height: int
    
    def to_dict(self) -> Dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> 'BoundingBox':
        return cls(
            left=data["left"],
            top=data["top"],
            width=data["width"],
            height=data["height"]
        )


@dataclass
class FieldExtraction:
    """Result of field extraction with confidence and validation."""
    value: Any
    confidence: float
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    extraction_method: str = "ai"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BaseField:
    """Base field definition for all form fields."""
    name: str
    bounding_box: BoundingBox
    field_type: Optional[FieldType] = None
    required: bool = False
    label: str = ""
    description: str = ""
    validation_level: ValidationLevel = ValidationLevel.MODERATE
    
    def validate(self, extraction: FieldExtraction) -> FieldExtraction:
        """Base validation logic."""
        if self.required and not extraction.value:
            extraction.is_valid = False
            extraction.validation_errors.append(f"Field '{self.name}' is required")
        return extraction


@dataclass
class TextField(BaseField):
    """Text field with pattern validation."""
    field_type: FieldType = FieldType.TEXT
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    case_sensitive: bool = False
    
    def validate(self, extraction: FieldExtraction) -> FieldExtraction:
        extraction = super().validate(extraction)
        
        if extraction.value:
            text_value = str(extraction.value)
            
            if self.min_length and len(text_value) < self.min_length:
                extraction.is_valid = False
                extraction.validation_errors.append(
                    f"Text too short. Minimum length: {self.min_length}"
                )
            
            if self.max_length and len(text_value) > self.max_length:
                extraction.is_valid = False
                extraction.validation_errors.append(
                    f"Text too long. Maximum length: {self.max_length}"
                )
            
            if self.pattern:
                pattern = self.pattern if self.case_sensitive else f"(?i){self.pattern}"
                if not re.match(pattern, text_value):
                    extraction.is_valid = False
                    extraction.validation_errors.append(
                        f"Text does not match required pattern"
                    )
        
        return extraction


@dataclass
class NumberField(BaseField):
    """Numeric field with range validation."""
    field_type: FieldType = FieldType.NUMBER
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    decimal_places: Optional[int] = None
    
    def validate(self, extraction: FieldExtraction) -> FieldExtraction:
        extraction = super().validate(extraction)
        
        if extraction.value is not None:
            try:
                numeric_value = float(extraction.value)
                
                if self.min_value is not None and numeric_value < self.min_value:
                    extraction.is_valid = False
                    extraction.validation_errors.append(
                        f"Value too small. Minimum: {self.min_value}"
                    )
                
                if self.max_value is not None and numeric_value > self.max_value:
                    extraction.is_valid = False
                    extraction.validation_errors.append(
                        f"Value too large. Maximum: {self.max_value}"
                    )
                
                # Update extraction value with proper type
                if self.decimal_places == 0:
                    extraction.value = int(numeric_value)
                else:
                    extraction.value = round(numeric_value, self.decimal_places)
                    
            except (ValueError, TypeError):
                extraction.is_valid = False
                extraction.validation_errors.append("Invalid numeric value")
        
        return extraction


@dataclass
class DateField(BaseField):
    """Date field with format validation."""
    field_type: FieldType = FieldType.DATE
    date_format: str = "%Y-%m-%d"
    min_date: Optional[datetime] = None
    max_date: Optional[datetime] = None
    
    def validate(self, extraction: FieldExtraction) -> FieldExtraction:
        extraction = super().validate(extraction)
        
        if extraction.value:
            try:
                if isinstance(extraction.value, str):
                    date_value = datetime.strptime(extraction.value, self.date_format)
                    extraction.value = date_value
                elif isinstance(extraction.value, datetime):
                    date_value = extraction.value
                else:
                    raise ValueError("Invalid date format")
                
                if self.min_date and date_value < self.min_date:
                    extraction.is_valid = False
                    extraction.validation_errors.append(
                        f"Date too early. Minimum: {self.min_date.strftime(self.date_format)}"
                    )
                
                if self.max_date and date_value > self.max_date:
                    extraction.is_valid = False
                    extraction.validation_errors.append(
                        f"Date too late. Maximum: {self.max_date.strftime(self.date_format)}"
                    )
                    
            except ValueError:
                extraction.is_valid = False
                extraction.validation_errors.append(f"Invalid date format. Expected: {self.date_format}")
        
        return extraction


@dataclass
class EmailField(BaseField):
    """Email field with format validation."""
    field_type: FieldType = FieldType.EMAIL
    
    def validate(self, extraction: FieldExtraction) -> FieldExtraction:
        extraction = super().validate(extraction)
        
        if extraction.value:
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, str(extraction.value)):
                extraction.is_valid = False
                extraction.validation_errors.append("Invalid email format")
        
        return extraction


@dataclass
class PhoneField(BaseField):
    """Phone field with format validation."""
    field_type: FieldType = FieldType.PHONE
    country_code: Optional[str] = None
    format_pattern: Optional[str] = None
    
    def validate(self, extraction: FieldExtraction) -> FieldExtraction:
        extraction = super().validate(extraction)
        
        if extraction.value:
            phone_value = str(extraction.value)
            
            # Remove common formatting characters
            cleaned_phone = re.sub(r'[^\d+]', '', phone_value)
            
            if self.format_pattern:
                if not re.match(self.format_pattern, phone_value):
                    extraction.is_valid = False
                    extraction.validation_errors.append("Invalid phone format")
            else:
                # Basic validation - should have at least 10 digits
                digits_only = re.sub(r'\D', '', cleaned_phone)
                if len(digits_only) < 10:
                    extraction.is_valid = False
                    extraction.validation_errors.append("Phone number too short")
        
        return extraction


@dataclass
class SignatureField(BaseField):
    """Signature field with presence and quality validation."""
    field_type: FieldType = FieldType.SIGNATURE
    min_signature_area: float = 0.1  # Minimum percentage of bounding box filled
    require_ink_detection: bool = True
    
    def validate(self, extraction: FieldExtraction) -> FieldExtraction:
        extraction = super().validate(extraction)
        
        if extraction.value:
            # Signature validation logic would go here
            # This could include checking for ink presence, signature area, etc.
            signature_data = extraction.value
            
            if isinstance(signature_data, dict):
                coverage = signature_data.get('coverage', 0)
                if coverage < self.min_signature_area:
                    extraction.is_valid = False
                    extraction.validation_errors.append(
                        f"Signature area too small. Minimum coverage: {self.min_signature_area * 100}%"
                    )
                
                if self.require_ink_detection and not signature_data.get('ink_detected', False):
                    extraction.is_valid = False
                    extraction.validation_errors.append("No ink signature detected")
        
        return extraction


@dataclass
class CheckboxField(BaseField):
    """Checkbox field with state validation."""
    field_type: FieldType = FieldType.CHECKBOX
    default_value: bool = False
    
    def validate(self, extraction: FieldExtraction) -> FieldExtraction:
        extraction = super().validate(extraction)
        
        if extraction.value is not None:
            try:
                if isinstance(extraction.value, str):
                    extraction.value = extraction.value.lower() in ['true', 'checked', 'yes', '1']
                else:
                    extraction.value = bool(extraction.value)
            except (ValueError, TypeError):
                extraction.is_valid = False
                extraction.validation_errors.append("Invalid checkbox value")
        
        return extraction


@dataclass
class RadioButtonField(BaseField):
    """Radio button field with option validation."""
    field_type: FieldType = FieldType.RADIO_BUTTON
    options: List[str] = field(default_factory=list)
    allow_multiple: bool = False
    
    def validate(self, extraction: FieldExtraction) -> FieldExtraction:
        extraction = super().validate(extraction)
        
        if extraction.value:
            if isinstance(extraction.value, list):
                if not self.allow_multiple and len(extraction.value) > 1:
                    extraction.is_valid = False
                    extraction.validation_errors.append("Multiple selections not allowed")
                
                for value in extraction.value:
                    if str(value) not in self.options:
                        extraction.is_valid = False
                        extraction.validation_errors.append(f"Invalid option: {value}")
            else:
                if str(extraction.value) not in self.options:
                    extraction.is_valid = False
                    extraction.validation_errors.append(f"Invalid option: {extraction.value}")
        
        return extraction


@dataclass
class DropdownField(BaseField):
    """Dropdown field with option validation."""
    field_type: FieldType = FieldType.DROPDOWN
    options: List[str] = field(default_factory=list)
    allow_custom: bool = False
    
    def validate(self, extraction: FieldExtraction) -> FieldExtraction:
        extraction = super().validate(extraction)
        
        if extraction.value and not self.allow_custom:
            if str(extraction.value) not in self.options:
                extraction.is_valid = False
                extraction.validation_errors.append(f"Invalid option: {extraction.value}")
        
        return extraction


@dataclass
class TableField(BaseField):
    """Table field with structure validation."""
    field_type: FieldType = FieldType.TABLE
    columns: List[str] = field(default_factory=list)
    min_rows: int = 0
    max_rows: Optional[int] = None
    
    def validate(self, extraction: FieldExtraction) -> FieldExtraction:
        extraction = super().validate(extraction)
        
        if extraction.value:
            if isinstance(extraction.value, list):
                if len(extraction.value) < self.min_rows:
                    extraction.is_valid = False
                    extraction.validation_errors.append(f"Too few rows. Minimum: {self.min_rows}")
                
                if self.max_rows and len(extraction.value) > self.max_rows:
                    extraction.is_valid = False
                    extraction.validation_errors.append(f"Too many rows. Maximum: {self.max_rows}")
                
                # Validate column structure
                for row_idx, row in enumerate(extraction.value):
                    if isinstance(row, dict):
                        missing_cols = set(self.columns) - set(row.keys())
                        if missing_cols:
                            extraction.validation_errors.append(
                                f"Row {row_idx}: Missing columns {missing_cols}"
                            )
            else:
                extraction.is_valid = False
                extraction.validation_errors.append("Table data must be a list of rows")
        
        return extraction


# Type alias for all field types
FormField = Union[
    TextField, NumberField, DateField, EmailField, PhoneField,
    SignatureField, CheckboxField, RadioButtonField, DropdownField, TableField
]