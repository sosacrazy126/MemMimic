"""
Input Validation Schemas

JSON schemas and validation rules for different types of inputs
in the MemMimic system.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import json


@dataclass
class ValidationSchema:
    """Base validation schema class."""
    name: str
    schema: Dict[str, Any]
    description: str = ""
    version: str = "1.0"


class MemoryInputSchema:
    """Schema for memory content input validation."""
    
    SCHEMA = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "minLength": 1,
                "maxLength": 102400,  # 100KB
                "description": "Memory content text"
            },
            "memory_type": {
                "type": "string",
                "enum": ["interaction", "reflection", "milestone"],
                "default": "interaction",
                "description": "Type of memory being stored"
            },
            "importance_score": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.5,
                "description": "Importance score for memory prioritization"
            },
            "metadata": {
                "type": "object",
                "properties": {
                    "cxd": {
                        "type": "string",
                        "enum": ["Control", "Context", "Data", "unknown"],
                        "description": "CXD classification"
                    },
                    "type": {
                        "type": "string",
                        "description": "Memory type metadata"
                    }
                },
                "additionalProperties": True,
                "description": "Additional metadata for memory"
            }
        },
        "required": ["content"],
        "additionalProperties": False
    }
    
    @classmethod
    def get_schema(cls) -> ValidationSchema:
        return ValidationSchema(
            name="memory_input",
            schema=cls.SCHEMA,
            description="Validation schema for memory content input"
        )


class TaleInputSchema:
    """Schema for tale input validation."""
    
    SCHEMA = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "minLength": 1,
                "maxLength": 200,
                "pattern": r"^[a-zA-Z0-9\s\-_\.]+$",
                "description": "Tale name (filesystem safe)"
            },
            "content": {
                "type": "string",
                "minLength": 1,
                "maxLength": 102400,  # 100KB
                "description": "Tale content"
            },
            "category": {
                "type": "string",
                "maxLength": 100,
                "enum": [
                    "claude/core", "claude/contexts", "claude/insights",
                    "claude/current", "claude/archive", "projects/", "misc/"
                ],
                "default": "claude/core",
                "description": "Tale category path"
            },
            "tags": {
                "type": "array",
                "items": {
                    "type": "string",
                    "maxLength": 50,
                    "pattern": r"^[a-zA-Z0-9\-_]+$"
                },
                "maxItems": 10,
                "uniqueItems": True,
                "description": "List of tags for tale organization"
            }
        },
        "required": ["name", "content"],
        "additionalProperties": False
    }
    
    @classmethod
    def get_schema(cls) -> ValidationSchema:
        return ValidationSchema(
            name="tale_input",
            schema=cls.SCHEMA,
            description="Validation schema for tale input data"
        )


class QueryInputSchema:
    """Schema for search query input validation."""
    
    SCHEMA = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "minLength": 1,
                "maxLength": 1000,
                "description": "Search query text"
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 1000,
                "default": 10,
                "description": "Maximum number of results to return"
            },
            "min_confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.0,
                "description": "Minimum confidence score for results"
            },
            "function_filter": {
                "type": "string",
                "enum": ["ALL", "CONTROL", "CONTEXT", "DATA", "C", "X", "D"],
                "default": "ALL",
                "description": "CXD function filter"
            },
            "db_name": {
                "type": "string",
                "maxLength": 100,
                "pattern": r"^[a-zA-Z0-9_\-]+$",
                "description": "Database name for search"
            },
            "search_type": {
                "type": "string",
                "enum": ["semantic", "lexical", "hybrid"],
                "default": "hybrid",
                "description": "Type of search to perform"
            },
            "filters": {
                "type": "object",
                "properties": {
                    "cxd_function": {
                        "type": "string",
                        "enum": ["Control", "Context", "Data"]
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": ["interaction", "reflection", "milestone"]
                    },
                    "importance_min": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "importance_max": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "additionalProperties": True,
                "description": "Additional search filters"
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
    
    @classmethod
    def get_schema(cls) -> ValidationSchema:
        return ValidationSchema(
            name="query_input",
            schema=cls.SCHEMA,
            description="Validation schema for search query input"
        )


class MCPRequestSchema:
    """Schema for MCP protocol request validation."""
    
    SCHEMA = {
        "type": "object",
        "properties": {
            "jsonrpc": {
                "type": "string",
                "enum": ["2.0"],
                "description": "JSON-RPC version"
            },
            "method": {
                "type": "string",
                "minLength": 1,
                "maxLength": 100,
                "pattern": r"^[a-zA-Z0-9_\-\.]+$",
                "description": "MCP method name"
            },
            "params": {
                "type": "object",
                "description": "Method parameters"
            },
            "id": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "number"},
                    {"type": "null"}
                ],
                "description": "Request identifier"
            }
        },
        "required": ["method", "params"],
        "additionalProperties": False
    }
    
    @classmethod
    def get_schema(cls) -> ValidationSchema:
        return ValidationSchema(
            name="mcp_request",
            schema=cls.SCHEMA,
            description="Validation schema for MCP protocol requests"
        )


class JSONInputSchema:
    """Schema for general JSON input validation."""
    
    SCHEMA = {
        "type": "object",
        "properties": {
            "data": {
                "description": "JSON data payload"
            }
        },
        "additionalProperties": True,
        "maxProperties": 1000,  # Prevent excessively large objects
        "description": "General JSON input validation"
    }
    
    @classmethod
    def get_schema(cls) -> ValidationSchema:
        return ValidationSchema(
            name="json_input",
            schema=cls.SCHEMA,
            description="Validation schema for general JSON input"
        )


# Specialized schemas for specific MCP methods

class MCPMemoryRecallSchema:
    """Schema for memory recall MCP requests."""
    
    SCHEMA = {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "enum": ["memmimic_recall_cxd", "memmimic_remember", "memmimic_think"]
            },
            "params": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 1000
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 10
                    },
                    "function_filter": {
                        "type": "string",
                        "enum": ["ALL", "CONTROL", "CONTEXT", "DATA"],
                        "default": "ALL"
                    },
                    "content": {
                        "type": "string",
                        "maxLength": 102400
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": ["interaction", "reflection", "milestone"]
                    },
                    "input_text": {
                        "type": "string",
                        "maxLength": 10000
                    }
                },
                "additionalProperties": False
            }
        },
        "required": ["method", "params"],
        "additionalProperties": False
    }


class MCPTaleManagementSchema:
    """Schema for tale management MCP requests."""
    
    SCHEMA = {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "enum": [
                    "memmimic_tales", "memmimic_save_tale", "memmimic_load_tale",
                    "memmimic_delete_tale", "memmimic_context_tale"
                ]
            },
            "params": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "maxLength": 200,
                        "pattern": r"^[a-zA-Z0-9\s\-_\.]+$"
                    },
                    "content": {
                        "type": "string",
                        "maxLength": 102400
                    },
                    "category": {
                        "type": "string",
                        "maxLength": 100
                    },
                    "tags": {
                        "type": "string",
                        "maxLength": 500
                    },
                    "query": {
                        "type": "string",
                        "maxLength": 1000
                    },
                    "style": {
                        "type": "string",
                        "enum": ["auto", "introduction", "technical", "philosophical"],
                        "default": "auto"
                    },
                    "max_memories": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 15
                    },
                    "confirm": {
                        "type": "boolean",
                        "default": False
                    }
                },
                "additionalProperties": False
            }
        },
        "required": ["method", "params"],
        "additionalProperties": False
    }


# Schema registry
_SCHEMA_REGISTRY: Dict[str, ValidationSchema] = {}


def register_schema(schema: ValidationSchema) -> None:
    """Register a validation schema."""
    _SCHEMA_REGISTRY[schema.name] = schema


def get_validation_schema(schema_name: str) -> Optional[ValidationSchema]:
    """Get a validation schema by name."""
    return _SCHEMA_REGISTRY.get(schema_name)


def list_available_schemas() -> List[str]:
    """List all available validation schema names."""
    return list(_SCHEMA_REGISTRY.keys())


def validate_against_schema(data: Dict[str, Any], schema_name: str) -> Dict[str, Any]:
    """
    Validate data against a named schema.
    
    Args:
        data: Data to validate
        schema_name: Name of the schema to use
        
    Returns:
        Dictionary with validation results
    """
    try:
        import jsonschema
        
        schema = get_validation_schema(schema_name)
        if not schema:
            return {
                "valid": False,
                "error": f"Schema '{schema_name}' not found",
                "errors": []
            }
        
        # Validate against JSON schema
        validator = jsonschema.Draft7Validator(schema.schema)
        errors = list(validator.iter_errors(data))
        
        return {
            "valid": len(errors) == 0,
            "schema_name": schema_name,
            "schema_version": schema.version,
            "errors": [
                {
                    "message": error.message,
                    "path": list(error.absolute_path),
                    "validator": error.validator,
                    "value": error.instance
                }
                for error in errors
            ]
        }
        
    except ImportError:
        # Fallback if jsonschema not available
        return {
            "valid": True,  # Assume valid if can't validate
            "error": "jsonschema library not available",
            "warning": "Schema validation skipped"
        }
    except Exception as e:
        return {
            "valid": False,
            "error": f"Validation error: {str(e)}",
            "errors": []
        }


# Initialize default schemas
def _initialize_schemas():
    """Initialize default validation schemas."""
    schemas = [
        MemoryInputSchema.get_schema(),
        TaleInputSchema.get_schema(),
        QueryInputSchema.get_schema(),
        MCPRequestSchema.get_schema(),
        JSONInputSchema.get_schema()
    ]
    
    for schema in schemas:
        register_schema(schema)


# Initialize schemas on import
_initialize_schemas()


# Export commonly used schemas
__all__ = [
    'ValidationSchema', 'MemoryInputSchema', 'TaleInputSchema', 'QueryInputSchema',
    'MCPRequestSchema', 'JSONInputSchema', 'MCPMemoryRecallSchema', 'MCPTaleManagementSchema',
    'register_schema', 'get_validation_schema', 'list_available_schemas',
    'validate_against_schema'
]