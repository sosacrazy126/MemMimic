"""
Base MCP protocol utilities and common functionality.

Provides shared functionality for all MCP handlers including request validation,
response formatting, and error handling.
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class MCPResponse:
    """Standard MCP response structure"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    metadata: Dict[str, Any] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}


class MCPBase:
    """
    Base class for MCP protocol handlers providing common functionality.
    
    Handles request validation, response formatting, error handling,
    and performance monitoring for MCP operations.
    """
    
    def __init__(self, handler_name: str):
        """
        Initialize MCP base handler.
        
        Args:
            handler_name: Name of the handler for logging and metrics
        """
        self.handler_name = handler_name
        self.logger = logging.getLogger(f"{__name__}.{handler_name}")
        
        # Performance metrics
        self._metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time_ms': 0.0,
            'total_response_time': 0.0,
            'last_request_time': None,
        }
        
        self.logger.info(f"MCP handler '{handler_name}' initialized")
    
    def validate_request(self, request: Dict[str, Any]) -> Optional[str]:
        """
        Validate MCP request structure and content.
        
        Args:
            request: MCP request dictionary
            
        Returns:
            Error message if validation fails, None if valid
        """
        try:
            # Check for required fields
            if not isinstance(request, dict):
                return "Request must be a dictionary"
            
            # Basic MCP structure validation
            if 'method' not in request:
                return "Request missing 'method' field"
            
            if 'params' not in request:
                return "Request missing 'params' field"
            
            method = request.get('method')
            if not isinstance(method, str) or not method.strip():
                return "Method must be a non-empty string"
            
            params = request.get('params')
            if not isinstance(params, dict):
                return "Params must be a dictionary"
            
            return None  # Valid request
            
        except Exception as e:
            return f"Request validation error: {str(e)}"
    
    def create_success_response(self, data: Any, 
                               metadata: Optional[Dict[str, Any]] = None) -> MCPResponse:
        """
        Create a successful MCP response.
        
        Args:
            data: Response data
            metadata: Optional response metadata
            
        Returns:
            MCPResponse object for successful operation
        """
        return MCPResponse(
            success=True,
            data=data,
            metadata=metadata or {}
        )
    
    def create_error_response(self, error_message: str, 
                             error_code: str = "UNKNOWN_ERROR",
                             metadata: Optional[Dict[str, Any]] = None) -> MCPResponse:
        """
        Create an error MCP response.
        
        Args:
            error_message: Human-readable error description
            error_code: Machine-readable error code
            metadata: Optional error metadata
            
        Returns:
            MCPResponse object for error condition
        """
        return MCPResponse(
            success=False,
            error=error_message,
            error_code=error_code,
            metadata=metadata or {}
        )
    
    def execute_with_metrics(self, operation_func, *args, **kwargs) -> MCPResponse:
        """
        Execute an operation with automatic metrics collection.
        
        Args:
            operation_func: Function to execute
            *args: Positional arguments for operation_func
            **kwargs: Keyword arguments for operation_func
            
        Returns:
            MCPResponse from operation_func
        """
        start_time = time.perf_counter()
        
        try:
            self._metrics['total_requests'] += 1
            self._metrics['last_request_time'] = datetime.now().isoformat()
            
            # Execute the operation
            result = operation_func(*args, **kwargs)
            
            # Update success metrics
            self._metrics['successful_requests'] += 1
            
            return result
            
        except Exception as e:
            # Update error metrics
            self._metrics['failed_requests'] += 1
            
            self.logger.error(f"Operation failed in {self.handler_name}: {e}")
            
            return self.create_error_response(
                error_message=f"Operation failed: {str(e)}",
                error_code="OPERATION_ERROR",
                metadata={'handler': self.handler_name}
            )
            
        finally:
            # Update timing metrics
            operation_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
            self._update_timing_metrics(operation_time)
    
    def _update_timing_metrics(self, operation_time_ms: float):
        """Update timing-related metrics."""
        self._metrics['total_response_time'] += operation_time_ms
        total_requests = self._metrics['total_requests']
        
        if total_requests > 0:
            self._metrics['avg_response_time_ms'] = (
                self._metrics['total_response_time'] / total_requests
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for this handler.
        
        Returns:
            Dictionary containing performance metrics
        """
        total_requests = self._metrics['total_requests']
        success_rate = (
            self._metrics['successful_requests'] / total_requests 
            if total_requests > 0 else 0.0
        )
        
        return {
            'handler_name': self.handler_name,
            'total_requests': total_requests,
            'successful_requests': self._metrics['successful_requests'],
            'failed_requests': self._metrics['failed_requests'],
            'success_rate': success_rate,
            'avg_response_time_ms': self._metrics['avg_response_time_ms'],
            'last_request_time': self._metrics['last_request_time'],
        }
    
    def reset_metrics(self):
        """Reset all performance metrics."""
        for key in self._metrics:
            if isinstance(self._metrics[key], (int, float)):
                self._metrics[key] = 0 if isinstance(self._metrics[key], int) else 0.0
            else:
                self._metrics[key] = None
                
        self.logger.info(f"Metrics reset for handler '{self.handler_name}'")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check for this MCP handler.
        
        Returns:
            Health status information
        """
        total_requests = self._metrics['total_requests']
        error_rate = (
            self._metrics['failed_requests'] / total_requests 
            if total_requests > 0 else 0.0
        )
        
        is_healthy = error_rate < 0.1  # Less than 10% error rate
        
        return {
            'handler_name': self.handler_name,
            'is_healthy': is_healthy,
            'error_rate': error_rate,
            'avg_response_time_ms': self._metrics['avg_response_time_ms'],
            'total_requests': total_requests,
            'status': 'healthy' if is_healthy else 'degraded'
        }


def parse_mcp_request(raw_request: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Parse raw MCP request from string or dict.
    
    Args:
        raw_request: Raw MCP request as string (JSON) or dict
        
    Returns:
        Parsed request dictionary
        
    Raises:
        ValueError: If request cannot be parsed
    """
    try:
        if isinstance(raw_request, str):
            return json.loads(raw_request)
        elif isinstance(raw_request, dict):
            return raw_request
        else:
            raise ValueError(f"Request must be string or dict, got {type(raw_request)}")
            
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in request: {e}")


def format_mcp_response(response: MCPResponse) -> str:
    """
    Format MCPResponse as JSON string for transmission.
    
    Args:
        response: MCPResponse to format
        
    Returns:
        JSON string representation of response
    """
    try:
        response_dict = asdict(response)
        return json.dumps(response_dict, indent=2, ensure_ascii=False)
        
    except Exception as e:
        # Fallback error response
        error_response = {
            'success': False,
            'error': f'Failed to format response: {str(e)}',
            'error_code': 'RESPONSE_FORMAT_ERROR',
            'timestamp': datetime.now().isoformat()
        }
        return json.dumps(error_response, indent=2)


def extract_search_params(mcp_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and validate search parameters from MCP request params.
    
    Args:
        mcp_params: MCP request parameters
        
    Returns:
        Dictionary of validated search parameters
        
    Raises:
        ValueError: If required parameters are missing or invalid
    """
    # Extract query text (required)
    query = mcp_params.get('query')
    if not query or not isinstance(query, str) or not query.strip():
        raise ValueError("Query parameter is required and must be non-empty string")
    
    # Extract optional parameters with defaults
    limit = mcp_params.get('limit', 5)
    if not isinstance(limit, int) or limit <= 0 or limit > 100:
        raise ValueError("Limit must be integer between 1 and 100")
    
    function_filter = mcp_params.get('function_filter', 'ALL').upper()
    valid_filters = ['CONTROL', 'CONTEXT', 'DATA', 'ALL', 'C', 'X', 'D']
    if function_filter not in valid_filters:
        raise ValueError(f"function_filter must be one of: {valid_filters}")
    
    # Normalize single-letter filters
    filter_map = {'C': 'CONTROL', 'X': 'CONTEXT', 'D': 'DATA'}
    function_filter = filter_map.get(function_filter, function_filter)
    
    db_name = mcp_params.get('db_name')
    if db_name is not None and not isinstance(db_name, str):
        raise ValueError("db_name must be string if provided")
    
    return {
        'query': query.strip(),
        'limit': limit,
        'function_filter': function_filter,
        'db_name': db_name
    }