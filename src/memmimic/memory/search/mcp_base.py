"""
Base classes and utilities for MCP (Model Context Protocol) handlers.

Provides common functionality and interfaces for MCP protocol handling
in the Memory Search System.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class MCPRequest:
    """
    Represents an MCP protocol request.
    
    Encapsulates request data, parameters, and metadata for processing
    by MCP handlers.
    """
    request_id: str
    method: str
    parameters: Dict[str, Any]
    timestamp: datetime
    client_info: Optional[Dict[str, Any]] = None
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get parameter value with optional default."""
        return self.parameters.get(key, default)
    
    def has_parameter(self, key: str) -> bool:
        """Check if parameter exists."""
        return key in self.parameters
    
    def validate_parameter_type(self, key: str, expected_type: type) -> bool:
        """Validate parameter type."""
        if not self.has_parameter(key):
            return False
        return isinstance(self.parameters[key], expected_type)


@dataclass
class MCPResponse:
    """
    Represents an MCP protocol response.
    
    Encapsulates response data, status, and metadata for returning
    results to MCP clients.
    """
    request_id: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    processing_time_ms: Optional[float] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for serialization."""
        response_dict = {
            'request_id': self.request_id,
            'success': self.success,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }
        
        if self.data is not None:
            response_dict['data'] = self.data
        
        if self.error is not None:
            response_dict['error'] = self.error
        
        if self.processing_time_ms is not None:
            response_dict['processing_time_ms'] = self.processing_time_ms
        
        return response_dict


class MCPError(Exception):
    """
    Custom exception for MCP protocol errors.
    
    Provides structured error information for MCP handlers
    with error codes and context.
    """
    
    def __init__(self, message: str, error_code: str = "UNKNOWN_ERROR", 
                 request_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.request_id = request_id
        self.context = context or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for response formatting."""
        return {
            'error_code': self.error_code,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context
        }


class MCPBaseHandler(ABC):
    """
    Abstract base class for MCP protocol handlers.
    
    Defines the interface and common functionality for handling
    MCP requests in the Memory Search System.
    """
    
    def __init__(self, handler_type: str):
        """
        Initialize base MCP handler.
        
        Args:
            handler_type: Type identifier for this handler
        """
        self.handler_type = handler_type
        self.created_at = datetime.now()
        
        # Base metrics tracking
        self._base_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_processing_time_ms': 0.0,
            'last_request_time': None,
            'handler_uptime_seconds': 0.0,
        }
        
        logger.info(f"MCP Handler '{handler_type}' initialized")
    
    @abstractmethod
    def handle_request(self, request: MCPRequest) -> MCPResponse:
        """
        Handle MCP request and return response.
        
        Args:
            request: MCP request to process
            
        Returns:
            MCP response with results or error information
        """
        pass
    
    def process_request(self, request: MCPRequest) -> MCPResponse:
        """
        Process MCP request with metrics tracking and error handling.
        
        Args:
            request: MCP request to process
            
        Returns:
            MCP response with processing metadata
        """
        start_time = time.time()
        self._base_metrics['total_requests'] += 1
        self._base_metrics['last_request_time'] = datetime.now()
        
        try:
            # Pre-processing validation
            self._validate_request(request)
            
            # Handle the request
            response = self.handle_request(request)
            
            # Post-processing
            processing_time = (time.time() - start_time) * 1000
            response.processing_time_ms = processing_time
            
            # Update success metrics
            self._base_metrics['successful_requests'] += 1
            self._update_processing_time_metrics(processing_time)
            
            logger.debug(f"Request {request.request_id} processed successfully in {processing_time:.2f}ms")
            return response
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self._base_metrics['failed_requests'] += 1
            self._update_processing_time_metrics(processing_time)
            
            logger.error(f"Request {request.request_id} failed: {e}")
            
            # Create error response
            error_response = MCPResponse(
                request_id=request.request_id,
                success=False,
                error=self._format_error(e),
                processing_time_ms=processing_time
            )
            
            return error_response
    
    def _validate_request(self, request: MCPRequest) -> None:
        """Validate basic request structure."""
        if not request.request_id:
            raise MCPError("Request ID is required", error_code="MISSING_REQUEST_ID")
        
        if not request.method:
            raise MCPError("Request method is required", error_code="MISSING_METHOD")
        
        if not isinstance(request.parameters, dict):
            raise MCPError("Parameters must be a dictionary", error_code="INVALID_PARAMETERS")
    
    def _format_error(self, error: Exception) -> Dict[str, Any]:
        """Format exception as error dictionary."""
        if isinstance(error, MCPError):
            return error.to_dict()
        else:
            return {
                'error_code': 'INTERNAL_ERROR',
                'message': str(error),
                'error_type': type(error).__name__,
                'timestamp': datetime.now().isoformat()
            }
    
    def _update_processing_time_metrics(self, processing_time_ms: float):
        """Update average processing time metrics."""
        total_requests = self._base_metrics['total_requests']
        current_avg = self._base_metrics['avg_processing_time_ms']
        
        # Calculate new average
        self._base_metrics['avg_processing_time_ms'] = (
            (current_avg * (total_requests - 1) + processing_time_ms) / total_requests
        )
    
    def get_base_metrics(self) -> Dict[str, Any]:
        """Get base handler metrics."""
        uptime_seconds = (datetime.now() - self.created_at).total_seconds()
        self._base_metrics['handler_uptime_seconds'] = uptime_seconds
        
        total_requests = self._base_metrics['total_requests']
        success_rate = (
            self._base_metrics['successful_requests'] / total_requests 
            if total_requests > 0 else 0.0
        )
        
        return {
            'handler_type': self.handler_type,
            'created_at': self.created_at.isoformat(),
            'uptime_seconds': uptime_seconds,
            'total_requests': total_requests,
            'successful_requests': self._base_metrics['successful_requests'],
            'failed_requests': self._base_metrics['failed_requests'],
            'success_rate': success_rate,
            'avg_processing_time_ms': self._base_metrics['avg_processing_time_ms'],
            'last_request_time': (
                self._base_metrics['last_request_time'].isoformat() 
                if self._base_metrics['last_request_time'] else None
            ),
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform handler health check.
        
        Returns:
            Health status information
        """
        try:
            metrics = self.get_base_metrics()
            
            # Basic health indicators
            error_rate = (
                metrics['failed_requests'] / metrics['total_requests'] 
                if metrics['total_requests'] > 0 else 0.0
            )
            
            # Determine health status
            healthy = (
                error_rate < 0.1 and  # Less than 10% error rate
                metrics['avg_processing_time_ms'] < 5000  # Less than 5 seconds average
            )
            
            return {
                'healthy': healthy,
                'error_rate': error_rate,
                'avg_processing_time_ms': metrics['avg_processing_time_ms'],
                'uptime_seconds': metrics['uptime_seconds'],
                'last_check': datetime.now().isoformat(),
                'status': 'healthy' if healthy else 'degraded'
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'healthy': False,
                'error': str(e),
                'last_check': datetime.now().isoformat(),
                'status': 'unhealthy'
            }


class MCPHandlerRegistry:
    """
    Registry for managing multiple MCP handlers.
    
    Provides centralized management and routing of MCP requests
    to appropriate handlers.
    """
    
    def __init__(self):
        """Initialize handler registry."""
        self._handlers: Dict[str, MCPBaseHandler] = {}
        self._method_mappings: Dict[str, str] = {}
        
        logger.info("MCP Handler Registry initialized")
    
    def register_handler(self, method: str, handler: MCPBaseHandler) -> None:
        """
        Register MCP handler for specific method.
        
        Args:
            method: MCP method name to handle
            handler: Handler instance
        """
        handler_id = f"{method}_{handler.handler_type}"
        self._handlers[handler_id] = handler
        self._method_mappings[method] = handler_id
        
        logger.info(f"Registered handler '{handler.handler_type}' for method '{method}'")
    
    def get_handler(self, method: str) -> Optional[MCPBaseHandler]:
        """
        Get handler for specific method.
        
        Args:
            method: MCP method name
            
        Returns:
            Handler instance if found, None otherwise
        """
        handler_id = self._method_mappings.get(method)
        if handler_id:
            return self._handlers.get(handler_id)
        return None
    
    def route_request(self, request: MCPRequest) -> MCPResponse:
        """
        Route request to appropriate handler.
        
        Args:
            request: MCP request to route
            
        Returns:
            MCP response from handler
        """
        try:
            handler = self.get_handler(request.method)
            if not handler:
                raise MCPError(
                    f"No handler registered for method: {request.method}",
                    error_code="UNKNOWN_METHOD",
                    request_id=request.request_id
                )
            
            return handler.process_request(request)
            
        except Exception as e:
            logger.error(f"Request routing failed: {e}")
            return MCPResponse(
                request_id=request.request_id,
                success=False,
                error={
                    'error_code': 'ROUTING_ERROR',
                    'message': str(e),
                    'timestamp': datetime.now().isoformat()
                }
            )
    
    def get_registry_status(self) -> Dict[str, Any]:
        """Get registry status and handler information."""
        handlers_info = {}
        
        for handler_id, handler in self._handlers.items():
            try:
                health = handler.health_check()
                metrics = handler.get_base_metrics()
                
                handlers_info[handler_id] = {
                    'handler_type': handler.handler_type,
                    'healthy': health['healthy'],
                    'status': health['status'],
                    'total_requests': metrics['total_requests'],
                    'success_rate': metrics['success_rate'],
                    'avg_processing_time_ms': metrics['avg_processing_time_ms'],
                    'uptime_seconds': metrics['uptime_seconds']
                }
            except Exception as e:
                handlers_info[handler_id] = {
                    'handler_type': handler.handler_type,
                    'healthy': False,
                    'error': str(e)
                }
        
        return {
            'total_handlers': len(self._handlers),
            'registered_methods': list(self._method_mappings.keys()),
            'handlers': handlers_info,
            'registry_uptime': datetime.now().isoformat()
        }


def create_mcp_request(request_id: str, method: str, parameters: Dict[str, Any],
                      client_info: Optional[Dict[str, Any]] = None) -> MCPRequest:
    """
    Factory function to create MCP request.
    
    Args:
        request_id: Unique request identifier
        method: MCP method name
        parameters: Request parameters
        client_info: Optional client information
        
    Returns:
        MCPRequest instance
    """
    return MCPRequest(
        request_id=request_id,
        method=method,
        parameters=parameters,
        timestamp=datetime.now(),
        client_info=client_info
    )


def create_success_response(request_id: str, data: Dict[str, Any],
                          processing_time_ms: Optional[float] = None) -> MCPResponse:
    """
    Factory function to create successful MCP response.
    
    Args:
        request_id: Request identifier
        data: Response data
        processing_time_ms: Processing time in milliseconds
        
    Returns:
        MCPResponse instance
    """
    return MCPResponse(
        request_id=request_id,
        success=True,
        data=data,
        processing_time_ms=processing_time_ms
    )


def create_error_response(request_id: str, error: Union[Exception, Dict[str, Any]],
                         processing_time_ms: Optional[float] = None) -> MCPResponse:
    """
    Factory function to create error MCP response.
    
    Args:
        request_id: Request identifier
        error: Error information (exception or dict)
        processing_time_ms: Processing time in milliseconds
        
    Returns:
        MCPResponse instance
    """
    if isinstance(error, Exception):
        if isinstance(error, MCPError):
            error_dict = error.to_dict()
        else:
            error_dict = {
                'error_code': 'INTERNAL_ERROR',
                'message': str(error),
                'error_type': type(error).__name__,
                'timestamp': datetime.now().isoformat()
            }
    else:
        error_dict = error
    
    return MCPResponse(
        request_id=request_id,
        success=False,
        error=error_dict,
        processing_time_ms=processing_time_ms
    )