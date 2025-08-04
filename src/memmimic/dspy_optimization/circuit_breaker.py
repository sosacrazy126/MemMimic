"""
DSPy Circuit Breaker Pattern

Reliability and safety mechanism for DSPy consciousness optimization that provides
graceful degradation when optimization fails or performance degrades.
"""

import time
import asyncio
import logging
from enum import Enum
from typing import Any, Callable, Dict, Optional, Awaitable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

from ..errors import MemMimicError, with_error_context, get_error_logger

logger = get_error_logger(__name__)

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation, requests pass through
    OPEN = "open"          # Failure threshold exceeded, requests fail fast
    HALF_OPEN = "half_open"  # Testing recovery, limited requests pass through

@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker performance metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeouts: int = 0
    circuit_opens: int = 0
    circuit_recoveries: int = 0
    current_failure_rate: float = 0.0
    average_response_time: float = 0.0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0

class DSPyOptimizationError(MemMimicError):
    """Error during DSPy optimization operations"""
    pass

class DSPyTimeoutError(DSPyOptimizationError):
    """DSPy operation timeout error"""
    pass

class DSPyCircuitBreaker:
    """
    Circuit breaker for DSPy consciousness optimization operations.
    
    Provides automatic failure detection, graceful degradation, and recovery
    mechanisms to protect the consciousness vault from DSPy optimization failures.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        timeout: int = 5,
        expected_exception: type = DSPyOptimizationError,
        fallback_handler: Optional[Callable] = None,
        name: str = "DSPyCircuitBreaker"
    ):
        """
        Initialize circuit breaker with safety parameters.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            timeout: Operation timeout in seconds
            expected_exception: Exception type that triggers circuit opening
            fallback_handler: Function to call when circuit is open
            name: Circuit breaker identifier for logging
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.fallback_handler = fallback_handler
        self.name = name
        
        # Circuit state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.last_success_time = time.time()
        
        # Performance tracking
        self.metrics = CircuitBreakerMetrics()
        self._response_times: list = []
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or fallback result
            
        Raises:
            DSPyOptimizationError: When circuit is open and no fallback available
        """
        async with self._lock:
            self.metrics.total_requests += 1
            
            # Check circuit state
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    logger.info(f"{self.name}: Circuit moving to HALF_OPEN for testing")
                else:
                    # Circuit is open, use fallback or fail fast
                    return await self._handle_open_circuit(*args, **kwargs)
            
            # Execute function with timeout protection
            start_time = time.time()
            try:
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.timeout
                )
                
                # Record success
                response_time = time.time() - start_time
                await self._record_success(response_time)
                
                return result
                
            except asyncio.TimeoutError:
                await self._record_timeout()
                raise DSPyTimeoutError(f"DSPy operation timed out after {self.timeout}s")
                
            except self.expected_exception as e:
                await self._record_failure(e)
                raise
                
            except Exception as e:
                # Unexpected error, treat as failure but re-raise original exception
                await self._record_failure(e)
                raise DSPyOptimizationError(f"Unexpected error in DSPy operation: {e}") from e
    
    async def _record_success(self, response_time: float) -> None:
        """Record successful operation"""
        self.metrics.successful_requests += 1
        self.metrics.last_success_time = time.time()
        
        # Update response time tracking
        self._response_times.append(response_time)
        if len(self._response_times) > 100:  # Keep last 100 measurements
            self._response_times.pop(0)
        
        self.metrics.average_response_time = sum(self._response_times) / len(self._response_times)
        
        # Reset failure count on success
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.metrics.circuit_recoveries += 1
            logger.info(f"{self.name}: Circuit recovered, moving to CLOSED")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)  # Gradual recovery
        
        self._update_failure_rate()
    
    async def _record_failure(self, error: Exception) -> None:
        """Record failed operation"""
        self.metrics.failed_requests += 1
        self.metrics.last_failure_time = time.time()
        self.last_failure_time = time.time()
        self.failure_count += 1
        
        logger.warning(
            f"{self.name}: Operation failed ({self.failure_count}/{self.failure_threshold}): {error}"
        )
        
        # Check if circuit should open
        if self.failure_count >= self.failure_threshold and self.state != CircuitState.OPEN:
            self.state = CircuitState.OPEN
            self.metrics.circuit_opens += 1
            logger.error(f"{self.name}: Circuit OPENED due to {self.failure_count} failures")
        
        self._update_failure_rate()
    
    async def _record_timeout(self) -> None:
        """Record timeout as failure"""
        self.metrics.timeouts += 1
        await self._record_failure(DSPyTimeoutError("Operation timeout"))
    
    def _update_failure_rate(self) -> None:
        """Update current failure rate"""
        if self.metrics.total_requests > 0:
            self.metrics.current_failure_rate = (
                self.metrics.failed_requests / self.metrics.total_requests
            )
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset to half-open"""
        if self.state != CircuitState.OPEN:
            return False
        
        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.recovery_timeout
    
    async def _handle_open_circuit(self, *args, **kwargs) -> Any:
        """Handle request when circuit is open"""
        if self.fallback_handler:
            try:
                logger.info(f"{self.name}: Circuit OPEN, using fallback handler")
                return await self.fallback_handler(*args, **kwargs)
            except Exception as e:
                logger.error(f"{self.name}: Fallback handler failed: {e}")
                raise DSPyOptimizationError("Circuit open and fallback failed") from e
        else:
            raise DSPyOptimizationError(
                f"Circuit breaker {self.name} is OPEN, operation not available"
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current circuit breaker metrics"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "timeouts": self.metrics.timeouts,
                "circuit_opens": self.metrics.circuit_opens,
                "circuit_recoveries": self.metrics.circuit_recoveries,
                "current_failure_rate": round(self.metrics.current_failure_rate, 3),
                "average_response_time": round(self.metrics.average_response_time, 3),
                "last_failure_time": self.metrics.last_failure_time,
                "last_success_time": self.metrics.last_success_time
            }
        }
    
    def reset(self) -> None:
        """Manually reset circuit breaker to closed state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        logger.info(f"{self.name}: Circuit manually reset to CLOSED")
    
    @asynccontextmanager
    async def protect(self, func: Callable[..., Awaitable[Any]]):
        """Context manager for circuit breaker protection"""
        try:
            result = await self.call(func)
            yield result
        except Exception as e:
            logger.error(f"{self.name}: Protected operation failed: {e}")
            raise

class DSPyCircuitBreakerManager:
    """
    Manager for multiple DSPy circuit breakers with different protection levels.
    """
    
    def __init__(self):
        self.breakers: Dict[str, DSPyCircuitBreaker] = {}
        self.default_config = {
            "biological_reflex": {
                "failure_threshold": 3,
                "recovery_timeout": 10,
                "timeout": 0.01,  # 10ms timeout for biological reflexes
            },
            "consciousness_pattern": {
                "failure_threshold": 5,
                "recovery_timeout": 30,
                "timeout": 0.05,  # 50ms timeout for consciousness operations
            },
            "optimization": {
                "failure_threshold": 10,
                "recovery_timeout": 60,
                "timeout": 5.0,  # 5s timeout for optimization operations
            }
        }
    
    def get_breaker(
        self,
        name: str,
        protection_level: str = "consciousness_pattern",
        fallback_handler: Optional[Callable] = None
    ) -> DSPyCircuitBreaker:
        """
        Get or create circuit breaker with specified protection level.
        
        Args:
            name: Circuit breaker identifier
            protection_level: Protection level: biological_reflex, consciousness_pattern, optimization
            fallback_handler: Fallback function for when circuit is open
            
        Returns:
            DSPyCircuitBreaker instance
        """
        if name not in self.breakers:
            config = self.default_config.get(protection_level, self.default_config["consciousness_pattern"])
            
            self.breakers[name] = DSPyCircuitBreaker(
                name=name,
                fallback_handler=fallback_handler,
                **config
            )
        
        return self.breakers[name]
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics from all circuit breakers"""
        return {name: breaker.get_metrics() for name, breaker in self.breakers.items()}
    
    def reset_all(self) -> None:
        """Reset all circuit breakers"""
        for breaker in self.breakers.values():
            breaker.reset()

# Global circuit breaker manager instance
circuit_breaker_manager = DSPyCircuitBreakerManager()