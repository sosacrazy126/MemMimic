"""
Governance-Integrated Storage for MemMimic v2.0
Enhanced AMMS Storage with integrated governance validation.
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..errors import (
    MemoryStorageError, handle_errors, with_error_context, get_error_logger
)
from .enhanced_amms_storage import EnhancedAMMSStorage
from .enhanced_memory import EnhancedMemory
from .governance import SimpleGovernance, GovernanceResult, GovernanceConfig


@dataclass
class GovernanceAwareResult:
    """Result of governance-aware storage operation"""
    success: bool
    memory_id: Optional[str]
    governance_result: GovernanceResult
    message: str
    error: Optional[str] = None
    total_processing_time: float = 0.0
    storage_time: float = 0.0
    governance_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {
            'success': self.success,
            'memory_id': self.memory_id,
            'governance_result': self.governance_result.to_dict(),
            'message': self.message,
            'total_processing_time': self.total_processing_time,
            'storage_time': self.storage_time,
            'governance_time': self.governance_time
        }
        if self.error:
            result['error'] = self.error
        return result


class GovernanceIntegratedStorage(EnhancedAMMSStorage):
    """
    Enhanced AMMS Storage with integrated governance validation.
    
    Provides seamless integration between storage operations and governance:
    - Pre-storage governance validation
    - Configurable enforcement modes
    - Performance tracking for both governance and storage
    - Graceful error handling and recovery
    - Real-time metrics and monitoring
    """
    
    def __init__(
        self, 
        db_path: str, 
        pool_size: Optional[int] = None, 
        config: Optional[Dict[str, Any]] = None
    ):
        # Initialize enhanced storage
        super().__init__(db_path, pool_size, config)
        
        # Initialize governance framework
        governance_config_data = config.get('governance', {}) if config else {}
        if governance_config_data:
            self.governance = SimpleGovernance(governance_config_data)
        else:
            # Use default governance configuration
            self.governance = SimpleGovernance(GovernanceConfig())
        
        # Governance performance tracking
        self._governance_metrics = {
            'total_validations': 0,
            'approvals': 0,
            'rejections': 0,
            'warnings': 0,
            'governance_time_total': 0.0,
            'storage_time_total': 0.0,
            'end_to_end_time_total': 0.0
        }
        
        self.logger.info(f"Governance-integrated storage initialized with {self.governance}")
    
    @handle_errors(catch=[Exception], reraise=True)
    async def store_with_governance(self, memory: EnhancedMemory) -> GovernanceAwareResult:
        """
        Store memory with comprehensive governance validation.
        
        Process:
        1. Pre-storage governance validation
        2. Store if approved (or log if audit-only mode)
        3. Update governance status in stored memory
        4. Return comprehensive result with metrics
        
        Args:
            memory: EnhancedMemory object to store
            
        Returns:
            GovernanceAwareResult with detailed outcome
        """
        operation_start_time = time.perf_counter()
        
        with with_error_context(
            operation="store_with_governance",
            component="governance_integrated_storage",
            metadata={
                "content_size": memory.context_size,
                "tag_count": memory.tag_count,
                "governance_enabled": self.governance.config.enabled
            }
        ):
            self._metrics['total_operations'] += 1
            self._governance_metrics['total_validations'] += 1
            
            # Step 1: Pre-storage governance validation
            governance_start = time.perf_counter()
            governance_result = await self.governance.validate_memory(
                memory, operation_context="store"
            )
            governance_time = (time.perf_counter() - governance_start) * 1000
            self._governance_metrics['governance_time_total'] += governance_time
            
            # Handle governance result based on enforcement mode
            if not governance_result.approved and self.governance.config.enforcement_mode != "audit_only":
                # Storage rejected due to governance violations
                total_time = (time.perf_counter() - operation_start_time) * 1000
                
                self._governance_metrics['rejections'] += 1
                self._log_governance_rejection(governance_result)
                
                return GovernanceAwareResult(
                    success=False,
                    memory_id=None,
                    governance_result=governance_result,
                    message=f"Storage rejected due to governance violations: {len(governance_result.violations)} violations",
                    total_processing_time=total_time,
                    governance_time=governance_time,
                    storage_time=0.0
                )
            
            # Step 2: Store the memory (governance approved or audit-only mode)
            try:
                storage_start = time.perf_counter()
                
                # Update memory with governance status
                memory.governance_status = governance_result.status
                
                # Store using enhanced storage capabilities
                memory_id = await self.store_enhanced_memory_optimized(memory)
                
                storage_time = (time.perf_counter() - storage_start) * 1000
                self._governance_metrics['storage_time_total'] += storage_time
                
                # Update governance metrics
                if governance_result.status == "approved":
                    self._governance_metrics['approvals'] += 1
                elif governance_result.warnings:
                    self._governance_metrics['approvals'] += 1
                    self._governance_metrics['warnings'] += 1
                
                total_time = (time.perf_counter() - operation_start_time) * 1000
                self._governance_metrics['end_to_end_time_total'] += total_time
                
                # Log successful storage with governance
                self.logger.debug(
                    f"Memory {memory_id} stored with governance: "
                    f"status={governance_result.status}, "
                    f"governance_time={governance_time:.2f}ms, "
                    f"storage_time={storage_time:.2f}ms, "
                    f"total_time={total_time:.2f}ms"
                )
                
                return GovernanceAwareResult(
                    success=True,
                    memory_id=memory_id,
                    governance_result=governance_result,
                    message="Memory stored successfully with governance compliance",
                    total_processing_time=total_time,
                    governance_time=governance_time,
                    storage_time=storage_time
                )
                
            except Exception as e:
                storage_time = (time.perf_counter() - storage_start) * 1000
                total_time = (time.perf_counter() - operation_start_time) * 1000
                
                self.logger.error(f"Storage failed after governance approval: {e}")
                self._metrics['failed_operations'] += 1
                
                return GovernanceAwareResult(
                    success=False,
                    memory_id=None,
                    governance_result=governance_result,
                    message=f"Storage failed after governance approval: {e}",
                    error=str(e),
                    total_processing_time=total_time,
                    governance_time=governance_time,
                    storage_time=storage_time
                )
    
    async def update_with_governance(
        self, 
        memory_id: str, 
        updated_memory: EnhancedMemory
    ) -> GovernanceAwareResult:
        """
        Update existing memory with governance validation.
        
        Args:
            memory_id: ID of memory to update
            updated_memory: Updated memory object
            
        Returns:
            GovernanceAwareResult with update outcome
        """
        operation_start_time = time.perf_counter()
        
        with with_error_context(
            operation="update_with_governance",
            component="governance_integrated_storage",
            metadata={
                "memory_id": memory_id,
                "content_size": updated_memory.context_size
            }
        ):
            # Validate the updated memory
            governance_result = await self.governance.validate_memory(
                updated_memory, operation_context="update"
            )
            
            if not governance_result.approved and self.governance.config.enforcement_mode != "audit_only":
                total_time = (time.perf_counter() - operation_start_time) * 1000
                return GovernanceAwareResult(
                    success=False,
                    memory_id=memory_id,
                    governance_result=governance_result,
                    message="Update rejected due to governance violations",
                    total_processing_time=total_time
                )
            
            # Update would require implementing update method in parent class
            # For now, return a placeholder result
            total_time = (time.perf_counter() - operation_start_time) * 1000
            
            return GovernanceAwareResult(
                success=False,
                memory_id=memory_id,
                governance_result=governance_result,
                message="Update operation not implemented in base storage",
                error="Method not implemented",
                total_processing_time=total_time
            )
    
    async def retrieve_with_governance_check(
        self, 
        memory_id: str, 
        context_level: str = "summary"
    ) -> GovernanceAwareResult:
        """
        Retrieve memory with optional governance compliance check.
        
        Args:
            memory_id: ID of memory to retrieve
            context_level: Level of context to retrieve (summary or full)
            
        Returns:
            GovernanceAwareResult with retrieval outcome and governance status
        """
        operation_start_time = time.perf_counter()
        
        with with_error_context(
            operation="retrieve_with_governance_check",
            component="governance_integrated_storage",
            metadata={
                "memory_id": memory_id,
                "context_level": context_level
            }
        ):
            try:
                # Retrieve the memory
                if context_level == "summary":
                    summary = await self.retrieve_summary_optimized(memory_id)
                    if summary:
                        # Create minimal EnhancedMemory for governance check
                        memory = EnhancedMemory(
                            id=memory_id,
                            content=summary,
                            summary=summary
                        )
                    else:
                        memory = None
                else:
                    memory = await self.retrieve_full_context_optimized(memory_id)
                
                if not memory:
                    total_time = (time.perf_counter() - operation_start_time) * 1000
                    return GovernanceAwareResult(
                        success=False,
                        memory_id=memory_id,
                        governance_result=GovernanceResult(
                            approved=False,
                            status="not_found",
                            violations=[],
                            processing_time=0.0
                        ),
                        message="Memory not found",
                        total_processing_time=total_time
                    )
                
                # Optional: Validate retrieved memory against current governance rules
                # This is useful for compliance auditing
                governance_result = await self.governance.validate_memory(
                    memory, operation_context="retrieve"
                )
                
                total_time = (time.perf_counter() - operation_start_time) * 1000
                
                return GovernanceAwareResult(
                    success=True,
                    memory_id=memory_id,
                    governance_result=governance_result,
                    message="Memory retrieved successfully",
                    total_processing_time=total_time
                )
                
            except Exception as e:
                total_time = (time.perf_counter() - operation_start_time) * 1000
                
                return GovernanceAwareResult(
                    success=False,
                    memory_id=memory_id,
                    governance_result=GovernanceResult(
                        approved=False,
                        status="error",
                        violations=[],
                        error=str(e),
                        processing_time=0.0
                    ),
                    message=f"Retrieval failed: {e}",
                    error=str(e),
                    total_processing_time=total_time
                )
    
    async def batch_store_with_governance(
        self, 
        memories: List[EnhancedMemory]
    ) -> List[GovernanceAwareResult]:
        """
        Batch store multiple memories with governance validation.
        
        Args:
            memories: List of EnhancedMemory objects to store
            
        Returns:
            List of GovernanceAwareResult for each memory
        """
        results = []
        
        for memory in memories:
            result = await self.store_with_governance(memory)
            results.append(result)
            
            # Early exit if too many rejections in strict mode
            if (self.governance.config.enforcement_mode == "strict" and 
                len([r for r in results if not r.success]) > len(memories) * 0.5):
                self.logger.warning(f"Stopping batch operation due to high rejection rate")
                break
        
        return results
    
    def get_governance_configuration(self) -> Dict[str, Any]:
        """Get current governance configuration"""
        return {
            'enabled': self.governance.config.enabled,
            'enforcement_mode': self.governance.config.enforcement_mode,
            'environment': self.governance.config.environment,
            'thresholds': {
                'content_size': self.governance.config.content_size,
                'summary_length': self.governance.config.summary_length,
                'metadata_size': self.governance.config.metadata_size,
                'tag_count': self.governance.config.tag_count,
                'tag_length': self.governance.config.tag_length,
                'governance_timeout': self.governance.config.governance_timeout
            }
        }
    
    def update_governance_configuration(self, new_config: Dict[str, Any]):
        """Update governance configuration with hot-reload"""
        try:
            self.governance.reload_config(new_config)
            self.logger.info("Governance configuration updated successfully")
        except Exception as e:
            self.logger.error(f"Failed to update governance configuration: {e}")
            raise
    
    def adjust_governance_thresholds(self, context: str, adjustments: Dict[str, Any]):
        """Dynamically adjust governance thresholds for specific context"""
        self.governance.adjust_thresholds(context, adjustments)
        self.logger.info(f"Governance thresholds adjusted for context '{context}': {adjustments}")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive stats including governance metrics"""
        base_stats = self.get_enhanced_stats()
        governance_metrics = self.governance.get_performance_metrics()
        
        # Calculate governance performance ratios
        total_validations = self._governance_metrics['total_validations']
        avg_governance_time = (
            self._governance_metrics['governance_time_total'] / total_validations
            if total_validations > 0 else 0.0
        )
        avg_storage_time = (
            self._governance_metrics['storage_time_total'] / total_validations
            if total_validations > 0 else 0.0
        )
        avg_end_to_end_time = (
            self._governance_metrics['end_to_end_time_total'] / total_validations
            if total_validations > 0 else 0.0
        )
        
        governance_stats = {
            'governance_integration': {
                'enabled': True,
                'framework_version': '2.0',
                'total_validations': total_validations,
                'approval_rate': (
                    self._governance_metrics['approvals'] / total_validations * 100
                    if total_validations > 0 else 0.0
                ),
                'rejection_rate': (
                    self._governance_metrics['rejections'] / total_validations * 100
                    if total_validations > 0 else 0.0
                ),
                'warning_rate': (
                    self._governance_metrics['warnings'] / total_validations * 100
                    if total_validations > 0 else 0.0
                ),
                'performance': {
                    'avg_governance_time_ms': avg_governance_time,
                    'avg_storage_time_ms': avg_storage_time,
                    'avg_end_to_end_time_ms': avg_end_to_end_time,
                    'governance_overhead_percent': (
                        avg_governance_time / avg_end_to_end_time * 100
                        if avg_end_to_end_time > 0 else 0.0
                    )
                }
            },
            'governance_framework_metrics': governance_metrics
        }
        
        # Merge base stats with governance stats
        base_stats.update(governance_stats)
        return base_stats
    
    def _log_governance_rejection(self, governance_result: GovernanceResult):
        """Log governance rejection with detailed information"""
        violation_summary = []
        for violation in governance_result.violations:
            violation_summary.append(
                f"{violation.type}({violation.severity}): {violation.message}"
            )
        
        self.logger.warning(
            f"Governance rejection: status={governance_result.status}, "
            f"violations={len(governance_result.violations)}, "
            f"warnings={len(governance_result.warnings)}, "
            f"processing_time={governance_result.processing_time:.2f}ms, "
            f"details={'; '.join(violation_summary)}"
        )
    
    async def close(self):
        """Enhanced cleanup including governance resources"""
        # Log final governance statistics
        governance_stats = self.get_comprehensive_stats()['governance_integration']
        self.logger.info(
            f"Closing governance-integrated storage with final stats: "
            f"validations={governance_stats['total_validations']}, "
            f"approval_rate={governance_stats['approval_rate']:.1f}%, "
            f"avg_governance_time={governance_stats['performance']['avg_governance_time_ms']:.2f}ms"
        )
        
        # Call parent cleanup
        await super().close()


def create_governance_integrated_storage(
    db_path: str, 
    config: Optional[Dict[str, Any]] = None
) -> GovernanceIntegratedStorage:
    """Factory function to create Governance-Integrated storage"""
    return GovernanceIntegratedStorage(db_path, config=config)