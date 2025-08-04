"""
SharedRealityManager - Multi-Agent Coordination System

Maintains shared reality states between multiple agent instances to enable
coordinated intelligence and collaborative processing. This component
ensures consistency across distributed agent interactions while preserving
individual agent autonomy.

Key features:
- Shared memory state synchronization
- Conflict resolution for concurrent operations
- Distributed consensus mechanisms
- Agent identity and capability tracking
"""

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import threading
import weakref

from ..errors import MemMimicError, with_error_context, get_error_logger
from ..memory.storage.amms_storage import create_amms_storage


class AgentRole(Enum):
    """Roles that agents can take in shared reality"""
    PRIMARY = "primary"          # Main decision-making agent
    SECONDARY = "secondary"      # Supporting agent
    OBSERVER = "observer"        # Read-only agent
    SPECIALIST = "specialist"    # Domain-specific agent
    COORDINATOR = "coordinator"  # Multi-agent coordination


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving conflicts between agents"""
    PRIORITY_BASED = "priority_based"      # Based on agent priority
    TIMESTAMP_BASED = "timestamp_based"    # First-come-first-served
    CONSENSUS_BASED = "consensus_based"    # Require majority agreement
    MERGE_BASED = "merge_based"           # Attempt to merge changes
    SPECIALIST_DEFERS = "specialist_defers" # Defer to domain specialist


@dataclass
class AgentIdentity:
    """Identity and capabilities of an agent in shared reality"""
    agent_id: str
    name: str
    role: AgentRole
    capabilities: List[str] = field(default_factory=list)
    priority: int = 1  # Higher number = higher priority
    last_seen: float = field(default_factory=time.time)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SharedState:
    """Shared state object that can be synchronized across agents"""
    state_id: str
    state_type: str  # 'memory', 'context', 'decision', 'analysis'
    data: Dict[str, Any]
    version: int = 1
    last_modified: float = field(default_factory=time.time)
    modified_by: str = ""
    lock_holder: Optional[str] = None
    lock_expires: Optional[float] = None
    access_log: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ConflictEvent:
    """Represents a conflict between agents"""
    conflict_id: str
    state_id: str
    conflicting_agents: List[str]
    conflict_type: str  # 'concurrent_modification', 'incompatible_changes', 'access_violation'
    resolution_strategy: ConflictResolutionStrategy
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolution_result: Optional[Dict[str, Any]] = None


class SharedRealityManager:
    """
    Multi-Agent Coordination System
    
    Manages shared reality states between multiple agent instances, ensuring
    consistency and coordination while preserving individual agent autonomy.
    """
    
    def __init__(self, db_path: str = None, reality_id: str = None):
        self.reality_id = reality_id or str(uuid.uuid4())
        self.db_path = db_path or "./src/memmimic/mcp/memmimic.db"
        self.logger = get_error_logger("shared_reality_manager")
        
        # Agent management
        self.agents: Dict[str, AgentIdentity] = {}
        self.agent_sessions: Dict[str, str] = {}  # session_id -> agent_id
        
        # Shared state management
        self.shared_states: Dict[str, SharedState] = {}
        self.state_locks: Dict[str, threading.RLock] = {}
        
        # Conflict resolution
        self.conflicts: Dict[str, ConflictEvent] = {}
        self.conflict_resolvers: Dict[ConflictResolutionStrategy, Callable] = {}
        
        # Synchronization
        self._sync_lock = threading.RLock()
        self._change_listeners: List[Callable] = []
        self._heartbeat_task: Optional[asyncio.Task] = None
        
        # Storage
        self._storage = None
        self._initialized = False
        
        # Performance tracking
        self._sync_metrics = {
            'total_syncs': 0,
            'successful_syncs': 0,
            'conflicts_resolved': 0,
            'average_sync_time_ms': 0.0
        }
        
        # Initialize conflict resolvers
        self._initialize_conflict_resolvers()
    
    async def initialize(self) -> None:
        """Initialize shared reality management system"""
        if self._initialized:
            return
            
        try:
            # Initialize storage
            self._storage = create_amms_storage(self.db_path)
            
            # Load existing shared reality state
            await self._load_shared_reality_state()
            
            # Start heartbeat monitoring
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            self._initialized = True
            self.logger.info(f"Shared reality manager initialized for reality: {self.reality_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize shared reality manager: {e}")
            raise MemMimicError(f"Shared reality manager initialization failed: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown shared reality manager"""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # Save shared reality state
        await self._save_shared_reality_state()
    
    async def register_agent(self, agent_id: str, name: str, role: AgentRole, 
                           capabilities: List[str] = None, priority: int = 1,
                           metadata: Dict[str, Any] = None) -> str:
        """Register an agent in the shared reality"""
        if not self._initialized:
            await self.initialize()
        
        with self._sync_lock:
            session_id = str(uuid.uuid4())
            
            agent = AgentIdentity(
                agent_id=agent_id,
                name=name,
                role=role,
                capabilities=capabilities or [],
                priority=priority,
                session_id=session_id,
                metadata=metadata or {}
            )
            
            self.agents[agent_id] = agent
            self.agent_sessions[session_id] = agent_id
            
            self.logger.info(f"Registered agent {agent_id} ({name}) with role {role.value}")
            
            # Notify other agents
            await self._notify_agent_change('agent_registered', agent_id)
            
            return session_id
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from shared reality"""
        with self._sync_lock:
            if agent_id not in self.agents:
                return False
            
            agent = self.agents[agent_id]
            
            # Release any locks held by this agent
            await self._release_agent_locks(agent_id)
            
            # Remove agent
            del self.agents[agent_id]
            if agent.session_id in self.agent_sessions:
                del self.agent_sessions[agent.session_id]
            
            self.logger.info(f"Unregistered agent {agent_id}")
            
            # Notify other agents
            await self._notify_agent_change('agent_unregistered', agent_id)
            
            return True
    
    async def create_shared_state(self, state_id: str, state_type: str, 
                                data: Dict[str, Any], agent_id: str) -> bool:
        """Create a new shared state"""
        if not self._initialized:
            await self.initialize()
        
        with self._sync_lock:
            if state_id in self.shared_states:
                return False
            
            shared_state = SharedState(
                state_id=state_id,
                state_type=state_type,
                data=data.copy(),
                modified_by=agent_id
            )
            
            shared_state.access_log.append({
                'action': 'created',
                'agent_id': agent_id,
                'timestamp': time.time()
            })
            
            self.shared_states[state_id] = shared_state
            self.state_locks[state_id] = threading.RLock()
            
            self.logger.info(f"Created shared state {state_id} by agent {agent_id}")
            
            # Notify change listeners
            await self._notify_state_change('state_created', state_id, agent_id)
            
            return True
    
    async def update_shared_state(self, state_id: str, updates: Dict[str, Any], 
                                agent_id: str, require_lock: bool = True) -> bool:
        """Update a shared state with conflict detection"""
        if state_id not in self.shared_states:
            return False
        
        with self.state_locks[state_id]:
            shared_state = self.shared_states[state_id]
            
            # Check lock requirements
            if require_lock and shared_state.lock_holder != agent_id:
                if shared_state.lock_holder is not None:
                    # State is locked by another agent
                    await self._handle_conflict(state_id, agent_id, 'access_violation')
                    return False
            
            # Check for concurrent modifications
            if shared_state.modified_by != agent_id and time.time() - shared_state.last_modified < 1.0:
                # Recent modification by another agent
                conflict_id = await self._handle_conflict(state_id, agent_id, 'concurrent_modification')
                if not conflict_id:
                    return False
            
            # Apply updates
            old_version = shared_state.version
            shared_state.data.update(updates)
            shared_state.version += 1
            shared_state.last_modified = time.time()
            shared_state.modified_by = agent_id
            
            shared_state.access_log.append({
                'action': 'updated',
                'agent_id': agent_id,
                'timestamp': time.time(),
                'old_version': old_version,
                'new_version': shared_state.version
            })
            
            self.logger.info(f"Updated shared state {state_id} by agent {agent_id} (v{shared_state.version})")
            
            # Notify change listeners
            await self._notify_state_change('state_updated', state_id, agent_id)
            
            return True
    
    async def acquire_state_lock(self, state_id: str, agent_id: str, 
                               duration_seconds: float = 30.0) -> bool:
        """Acquire a lock on a shared state"""
        if state_id not in self.shared_states:
            return False
        
        with self.state_locks[state_id]:
            shared_state = self.shared_states[state_id]
            
            # Check if already locked
            if shared_state.lock_holder is not None:
                # Check if lock has expired
                if shared_state.lock_expires and time.time() > shared_state.lock_expires:
                    # Lock expired, can acquire
                    pass
                elif shared_state.lock_holder == agent_id:
                    # Agent already holds the lock, extend it
                    shared_state.lock_expires = time.time() + duration_seconds
                    return True
                else:
                    # Lock held by another agent
                    return False
            
            # Acquire lock
            shared_state.lock_holder = agent_id
            shared_state.lock_expires = time.time() + duration_seconds
            
            shared_state.access_log.append({
                'action': 'lock_acquired',
                'agent_id': agent_id,
                'timestamp': time.time(),
                'duration': duration_seconds
            })
            
            self.logger.info(f"Agent {agent_id} acquired lock on state {state_id}")
            return True
    
    async def release_state_lock(self, state_id: str, agent_id: str) -> bool:
        """Release a lock on a shared state"""
        if state_id not in self.shared_states:
            return False
        
        with self.state_locks[state_id]:
            shared_state = self.shared_states[state_id]
            
            if shared_state.lock_holder != agent_id:
                return False
            
            shared_state.lock_holder = None
            shared_state.lock_expires = None
            
            shared_state.access_log.append({
                'action': 'lock_released',
                'agent_id': agent_id,
                'timestamp': time.time()
            })
            
            self.logger.info(f"Agent {agent_id} released lock on state {state_id}")
            return True
    
    async def get_shared_state(self, state_id: str, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get a shared state (read-only access)"""
        if state_id not in self.shared_states:
            return None
        
        shared_state = self.shared_states[state_id]
        
        # Log access
        shared_state.access_log.append({
            'action': 'read',
            'agent_id': agent_id,
            'timestamp': time.time()
        })
        
        return {
            'state_id': shared_state.state_id,
            'state_type': shared_state.state_type,
            'data': shared_state.data.copy(),
            'version': shared_state.version,
            'last_modified': shared_state.last_modified,
            'modified_by': shared_state.modified_by,
            'locked': shared_state.lock_holder is not None
        }
    
    async def _handle_conflict(self, state_id: str, agent_id: str, conflict_type: str) -> Optional[str]:
        """Handle conflicts between agents"""
        conflict_id = str(uuid.uuid4())
        
        # Determine other agents involved
        shared_state = self.shared_states[state_id]
        conflicting_agents = [agent_id]
        
        if shared_state.lock_holder and shared_state.lock_holder != agent_id:
            conflicting_agents.append(shared_state.lock_holder)
        if shared_state.modified_by and shared_state.modified_by != agent_id:
            conflicting_agents.append(shared_state.modified_by)
        
        # Create conflict event
        conflict = ConflictEvent(
            conflict_id=conflict_id,
            state_id=state_id,
            conflicting_agents=list(set(conflicting_agents)),
            conflict_type=conflict_type,
            resolution_strategy=self._determine_resolution_strategy(conflicting_agents)
        )
        
        self.conflicts[conflict_id] = conflict
        
        # Attempt to resolve conflict
        resolution_result = await self._resolve_conflict(conflict)
        
        if resolution_result:
            conflict.resolved = True
            conflict.resolution_result = resolution_result
            self._sync_metrics['conflicts_resolved'] += 1
            return conflict_id
        
        return None
    
    def _determine_resolution_strategy(self, conflicting_agents: List[str]) -> ConflictResolutionStrategy:
        """Determine the best conflict resolution strategy"""
        # Simple strategy selection based on agent roles and priorities
        agent_roles = [self.agents[aid].role for aid in conflicting_agents if aid in self.agents]
        
        if AgentRole.COORDINATOR in agent_roles:
            return ConflictResolutionStrategy.PRIORITY_BASED
        elif AgentRole.SPECIALIST in agent_roles:
            return ConflictResolutionStrategy.SPECIALIST_DEFERS
        else:
            return ConflictResolutionStrategy.TIMESTAMP_BASED
    
    async def _resolve_conflict(self, conflict: ConflictEvent) -> Optional[Dict[str, Any]]:
        """Resolve a conflict using the appropriate strategy"""
        resolver = self.conflict_resolvers.get(conflict.resolution_strategy)
        if resolver:
            return await resolver(conflict)
        return None
    
    def _initialize_conflict_resolvers(self) -> None:
        """Initialize conflict resolution strategies"""
        self.conflict_resolvers = {
            ConflictResolutionStrategy.PRIORITY_BASED: self._resolve_by_priority,
            ConflictResolutionStrategy.TIMESTAMP_BASED: self._resolve_by_timestamp,
            ConflictResolutionStrategy.SPECIALIST_DEFERS: self._resolve_by_specialist,
        }
    
    async def _resolve_by_priority(self, conflict: ConflictEvent) -> Dict[str, Any]:
        """Resolve conflict based on agent priority"""
        highest_priority_agent = max(
            conflict.conflicting_agents,
            key=lambda aid: self.agents.get(aid, AgentIdentity("", "", AgentRole.OBSERVER)).priority
        )
        
        return {
            'resolution_method': 'priority_based',
            'winning_agent': highest_priority_agent,
            'action': 'grant_access'
        }
    
    async def _resolve_by_timestamp(self, conflict: ConflictEvent) -> Dict[str, Any]:
        """Resolve conflict based on first-come-first-served"""
        shared_state = self.shared_states[conflict.state_id]
        
        return {
            'resolution_method': 'timestamp_based',
            'winning_agent': shared_state.modified_by,
            'action': 'maintain_current_state'
        }
    
    async def _resolve_by_specialist(self, conflict: ConflictEvent) -> Dict[str, Any]:
        """Resolve conflict by deferring to specialist agent"""
        specialist_agent = None
        for agent_id in conflict.conflicting_agents:
            if agent_id in self.agents and self.agents[agent_id].role == AgentRole.SPECIALIST:
                specialist_agent = agent_id
                break
        
        if specialist_agent:
            return {
                'resolution_method': 'specialist_defers',
                'winning_agent': specialist_agent,
                'action': 'defer_to_specialist'
            }
        
        # Fallback to priority-based
        return await self._resolve_by_priority(conflict)
    
    async def _release_agent_locks(self, agent_id: str) -> None:
        """Release all locks held by an agent"""
        for state_id, shared_state in self.shared_states.items():
            if shared_state.lock_holder == agent_id:
                await self.release_state_lock(state_id, agent_id)
    
    async def _notify_agent_change(self, event_type: str, agent_id: str) -> None:
        """Notify listeners of agent changes"""
        for listener in self._change_listeners:
            try:
                await listener(event_type, {'agent_id': agent_id})
            except Exception as e:
                self.logger.warning(f"Change listener failed: {e}")
    
    async def _notify_state_change(self, event_type: str, state_id: str, agent_id: str) -> None:
        """Notify listeners of state changes"""
        for listener in self._change_listeners:
            try:
                await listener(event_type, {'state_id': state_id, 'agent_id': agent_id})
            except Exception as e:
                self.logger.warning(f"Change listener failed: {e}")
    
    async def _heartbeat_loop(self) -> None:
        """Background heartbeat loop for agent monitoring"""
        while True:
            try:
                current_time = time.time()
                
                # Check for expired locks
                for state_id, shared_state in self.shared_states.items():
                    if (shared_state.lock_expires and 
                        current_time > shared_state.lock_expires):
                        
                        self.logger.info(f"Lock expired on state {state_id}")
                        shared_state.lock_holder = None
                        shared_state.lock_expires = None
                
                # Check for inactive agents
                inactive_agents = []
                for agent_id, agent in self.agents.items():
                    if current_time - agent.last_seen > 300:  # 5 minutes
                        inactive_agents.append(agent_id)
                
                for agent_id in inactive_agents:
                    self.logger.info(f"Removing inactive agent {agent_id}")
                    await self.unregister_agent(agent_id)
                
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.warning(f"Heartbeat loop error: {e}")
                await asyncio.sleep(60)
    
    async def _load_shared_reality_state(self) -> None:
        """Load shared reality state from persistent storage"""
        # Implementation would load from database
        pass
    
    async def _save_shared_reality_state(self) -> None:
        """Save shared reality state to persistent storage"""
        # Implementation would save to database
        pass
    
    def add_change_listener(self, listener: Callable) -> None:
        """Add a listener for shared reality changes"""
        self._change_listeners.append(listener)
    
    def remove_change_listener(self, listener: Callable) -> None:
        """Remove a change listener"""
        if listener in self._change_listeners:
            self._change_listeners.remove(listener)
    
    def get_reality_status(self) -> Dict[str, Any]:
        """Get current shared reality status"""
        return {
            'reality_id': self.reality_id,
            'active_agents': len(self.agents),
            'shared_states': len(self.shared_states),
            'active_conflicts': len([c for c in self.conflicts.values() if not c.resolved]),
            'sync_metrics': self._sync_metrics.copy(),
            'agents': {aid: {
                'name': agent.name,
                'role': agent.role.value,
                'priority': agent.priority,
                'last_seen': agent.last_seen
            } for aid, agent in self.agents.items()}
        }
    
    def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific agent"""
        if agent_id not in self.agents:
            return None
        
        agent = self.agents[agent_id]
        return {
            'agent_id': agent.agent_id,
            'name': agent.name,
            'role': agent.role.value,
            'capabilities': agent.capabilities,
            'priority': agent.priority,
            'last_seen': agent.last_seen,
            'session_id': agent.session_id,
            'metadata': agent.metadata
        }
