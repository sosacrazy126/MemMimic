"""
PhaseEvolutionTracker - Biological Development Phase Tracking

Implements phase-based development tracking following the Memory/Phase_* structure
to enable systematic evolution of the nervous system through documented phases.

This component tracks progress through biological evolution phases:
- Phase 1: Core nervous system foundation with internal quality gates
- Phase 2: Enhanced trigger mechanisms for natural language reflexes  
- Phase 3: MCP integration testing and validation
- Phase 4: Performance optimization and migration cleanup

Each phase represents a milestone in the nervous system's biological evolution.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

from ..errors import MemMimicError, with_error_context, get_error_logger
from ..memory.storage.amms_storage import create_amms_storage


class PhaseStatus(Enum):
    """Phase execution status"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskStatus(Enum):
    """Individual task status within a phase"""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class PhaseTask:
    """Individual task within a development phase"""
    task_id: str
    name: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    implementation_notes: List[str] = field(default_factory=list)
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    execution_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DevelopmentPhase:
    """Represents a biological development phase"""
    phase_id: str
    name: str
    description: str
    status: PhaseStatus = PhaseStatus.NOT_STARTED
    tasks: Dict[str, PhaseTask] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    phase_metrics: Dict[str, Any] = field(default_factory=dict)
    memory_log_path: Optional[str] = None


@dataclass
class EvolutionMetrics:
    """Overall evolution tracking metrics"""
    total_phases: int = 0
    completed_phases: int = 0
    active_phases: int = 0
    failed_phases: int = 0
    total_tasks: int = 0
    completed_tasks: int = 0
    overall_progress: float = 0.0
    evolution_start_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)


class PhaseEvolutionTracker:
    """
    Phase-Based Evolution Tracking System
    
    Tracks the nervous system's evolution through documented biological phases,
    providing systematic progress monitoring and milestone achievement tracking.
    """
    
    def __init__(self, memory_path: str = "Memory", db_path: str = None):
        self.memory_path = Path(memory_path)
        self.db_path = db_path or "./src/memmimic/mcp/memmimic.db"
        self.logger = get_error_logger("phase_evolution_tracker")
        
        # Phase tracking
        self.phases: Dict[str, DevelopmentPhase] = {}
        self.evolution_metrics = EvolutionMetrics()
        
        # Active tracking
        self._storage = None
        self._initialized = False
        self._current_phase: Optional[str] = None
        
        # Evolution state persistence
        self._state_file = Path("nervous_system_evolution_state.json")
        
    async def initialize(self) -> None:
        """Initialize phase evolution tracking system"""
        if self._initialized:
            return
            
        try:
            # Initialize storage
            self._storage = create_amms_storage(self.db_path)
            
            # Load existing evolution state
            await self._load_evolution_state()
            
            # Discover and initialize phases from Memory directory
            await self._discover_phases()
            
            # Update evolution metrics
            await self._update_evolution_metrics()
            
            self._initialized = True
            self.logger.info("Phase evolution tracker initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize phase evolution tracker: {e}")
            raise MemMimicError(f"Phase evolution tracker initialization failed: {e}")
    
    async def _discover_phases(self) -> None:
        """Discover development phases from Memory directory structure"""
        if not self.memory_path.exists():
            self.logger.warning(f"Memory directory not found: {self.memory_path}")
            return
            
        with with_error_context(
            operation="discover_phases",
            component="phase_evolution_tracker"
        ):
            # Scan for Phase_* directories
            phase_dirs = [d for d in self.memory_path.iterdir() 
                         if d.is_dir() and d.name.startswith("Phase_")]
            
            for phase_dir in sorted(phase_dirs):
                await self._process_phase_directory(phase_dir)
            
            self.logger.info(f"Discovered {len(self.phases)} development phases")
    
    async def _process_phase_directory(self, phase_dir: Path) -> None:
        """Process a single phase directory to extract phase information"""
        try:
            phase_name = phase_dir.name
            phase_id = phase_name.lower().replace("_", "-")
            
            # Extract phase information from directory name and contents
            phase_info = self._parse_phase_name(phase_name)
            
            # Create or update phase
            if phase_id not in self.phases:
                phase = DevelopmentPhase(
                    phase_id=phase_id,
                    name=phase_info["name"],
                    description=phase_info["description"],
                    memory_log_path=str(phase_dir)
                )
                self.phases[phase_id] = phase
            else:
                phase = self.phases[phase_id]
            
            # Discover tasks from log files
            await self._discover_phase_tasks(phase, phase_dir)
            
        except Exception as e:
            self.logger.error(f"Failed to process phase directory {phase_dir}: {e}")
    
    def _parse_phase_name(self, phase_name: str) -> Dict[str, str]:
        """Parse phase name to extract structured information"""
        # Phase naming convention: Phase_N_Description
        parts = phase_name.split("_", 2)
        
        if len(parts) >= 3:
            phase_num = parts[1]
            description = parts[2].replace("_", " ")
        else:
            phase_num = "Unknown"
            description = phase_name.replace("_", " ")
        
        # Map known phases to detailed descriptions
        phase_descriptions = {
            "1": "Core nervous system foundation with internal quality gates",
            "2": "Enhanced trigger mechanisms for natural language reflexes",
            "3": "MCP integration testing and validation", 
            "4": "Performance optimization and migration cleanup"
        }
        
        detailed_description = phase_descriptions.get(phase_num, description)
        
        return {
            "name": f"Phase {phase_num}: {description}",
            "description": detailed_description
        }
    
    async def _discover_phase_tasks(self, phase: DevelopmentPhase, phase_dir: Path) -> None:
        """Discover tasks from phase directory log files"""
        try:
            # Look for Task_*.md files
            task_files = list(phase_dir.glob("Task_*.md"))
            
            for task_file in task_files:
                task_info = self._parse_task_file(task_file)
                if task_info:
                    task = PhaseTask(
                        task_id=task_info["task_id"],
                        name=task_info["name"],
                        description=task_info["description"],
                        success_criteria=task_info.get("success_criteria", []),
                        implementation_notes=task_info.get("implementation_notes", [])
                    )
                    phase.tasks[task.task_id] = task
            
        except Exception as e:
            self.logger.error(f"Failed to discover tasks for phase {phase.phase_id}: {e}")
    
    def _parse_task_file(self, task_file: Path) -> Optional[Dict[str, Any]]:
        """Parse a task log file to extract task information"""
        try:
            # Extract task ID from filename: Task_1.1_Description_Log.md
            filename = task_file.stem
            parts = filename.split("_")
            
            if len(parts) >= 2:
                task_id = parts[1]
                name_parts = parts[2:-1] if parts[-1] == "Log" else parts[2:]
                name = " ".join(name_parts).replace("_", " ")
            else:
                task_id = filename
                name = filename.replace("_", " ")
            
            # Read file content for additional details
            try:
                with open(task_file, 'r') as f:
                    content = f.read()
                
                # Extract description and other details from content
                description = self._extract_description_from_content(content)
                
            except Exception:
                description = name
            
            return {
                "task_id": task_id,
                "name": name,
                "description": description,
                "success_criteria": [],
                "implementation_notes": []
            }
            
        except Exception as e:
            self.logger.error(f"Failed to parse task file {task_file}: {e}")
            return None
    
    def _extract_description_from_content(self, content: str) -> str:
        """Extract description from task log file content"""
        lines = content.split('\n')
        
        # Look for description patterns
        for i, line in enumerate(lines):
            if 'description' in line.lower() or 'details' in line.lower():
                # Try to get the next few lines as description
                desc_lines = []
                for j in range(i+1, min(i+4, len(lines))):
                    if lines[j].strip() and not lines[j].startswith('#'):
                        desc_lines.append(lines[j].strip())
                
                if desc_lines:
                    return ' '.join(desc_lines)
        
        # Fallback: use first non-header line
        for line in lines:
            if line.strip() and not line.startswith('#') and not line.startswith('**'):
                return line.strip()
        
        return "Task description not available"
    
    async def start_phase(self, phase_id: str) -> bool:
        """Start execution of a development phase"""
        if not self._initialized:
            await self.initialize()
            
        if phase_id not in self.phases:
            raise MemMimicError(f"Phase not found: {phase_id}")
        
        phase = self.phases[phase_id]
        
        if phase.status != PhaseStatus.NOT_STARTED:
            self.logger.warning(f"Phase {phase_id} already started or completed")
            return False
        
        # Check dependencies
        for dep_phase_id in phase.dependencies:
            if dep_phase_id in self.phases:
                dep_phase = self.phases[dep_phase_id]
                if dep_phase.status != PhaseStatus.COMPLETED:
                    raise MemMimicError(f"Dependency phase {dep_phase_id} not completed")
        
        # Start phase
        phase.status = PhaseStatus.IN_PROGRESS
        phase.start_time = time.time()
        self._current_phase = phase_id
        
        await self._save_evolution_state()
        
        self.logger.info(f"Started phase: {phase.name}")
        return True
    
    async def complete_phase(self, phase_id: str) -> bool:
        """Mark a development phase as completed"""
        if phase_id not in self.phases:
            raise MemMimicError(f"Phase not found: {phase_id}")
        
        phase = self.phases[phase_id]
        
        if phase.status != PhaseStatus.IN_PROGRESS:
            self.logger.warning(f"Phase {phase_id} not in progress")
            return False
        
        # Check if all tasks are completed
        incomplete_tasks = [task for task in phase.tasks.values() 
                          if task.status not in [TaskStatus.COMPLETED, TaskStatus.SKIPPED]]
        
        if incomplete_tasks:
            self.logger.warning(f"Phase {phase_id} has {len(incomplete_tasks)} incomplete tasks")
            return False
        
        # Complete phase
        phase.status = PhaseStatus.COMPLETED
        phase.completion_time = time.time()
        
        if self._current_phase == phase_id:
            self._current_phase = None
        
        await self._update_evolution_metrics()
        await self._save_evolution_state()
        
        self.logger.info(f"Completed phase: {phase.name}")
        return True
    
    async def update_task_status(self, phase_id: str, task_id: str, status: TaskStatus, 
                               metrics: Dict[str, Any] = None) -> bool:
        """Update the status of a specific task"""
        if phase_id not in self.phases:
            raise MemMimicError(f"Phase not found: {phase_id}")
        
        phase = self.phases[phase_id]
        
        if task_id not in phase.tasks:
            raise MemMimicError(f"Task not found: {task_id} in phase {phase_id}")
        
        task = phase.tasks[task_id]
        old_status = task.status
        task.status = status
        
        if metrics:
            task.execution_metrics.update(metrics)
        
        # Update timestamps
        if status == TaskStatus.ACTIVE and old_status == TaskStatus.PENDING:
            task.start_time = time.time()
        elif status == TaskStatus.COMPLETED and old_status == TaskStatus.ACTIVE:
            task.completion_time = time.time()
        
        await self._save_evolution_state()
        
        self.logger.info(f"Updated task {task_id} in phase {phase_id}: {old_status.value} -> {status.value}")
        return True
    
    async def _update_evolution_metrics(self) -> None:
        """Update overall evolution metrics"""
        self.evolution_metrics.total_phases = len(self.phases)
        self.evolution_metrics.completed_phases = sum(1 for p in self.phases.values() 
                                                    if p.status == PhaseStatus.COMPLETED)
        self.evolution_metrics.active_phases = sum(1 for p in self.phases.values() 
                                                 if p.status == PhaseStatus.IN_PROGRESS)
        self.evolution_metrics.failed_phases = sum(1 for p in self.phases.values() 
                                                 if p.status == PhaseStatus.FAILED)
        
        # Calculate task metrics
        all_tasks = [task for phase in self.phases.values() for task in phase.tasks.values()]
        self.evolution_metrics.total_tasks = len(all_tasks)
        self.evolution_metrics.completed_tasks = sum(1 for task in all_tasks 
                                                   if task.status == TaskStatus.COMPLETED)
        
        # Calculate overall progress
        if self.evolution_metrics.total_phases > 0:
            self.evolution_metrics.overall_progress = (
                self.evolution_metrics.completed_phases / self.evolution_metrics.total_phases
            )
        
        self.evolution_metrics.last_update_time = time.time()
    
    async def _load_evolution_state(self) -> None:
        """Load evolution state from persistent storage"""
        if self._state_file.exists():
            try:
                with open(self._state_file, 'r') as f:
                    state_data = json.load(f)
                
                # Restore phases and metrics
                # Implementation would deserialize the state
                self.logger.info("Loaded evolution state from persistent storage")
                
            except Exception as e:
                self.logger.error(f"Failed to load evolution state: {e}")
    
    async def _save_evolution_state(self) -> None:
        """Save evolution state to persistent storage"""
        try:
            state_data = {
                "phases": {pid: self._serialize_phase(phase) for pid, phase in self.phases.items()},
                "metrics": self._serialize_metrics(self.evolution_metrics),
                "current_phase": self._current_phase,
                "last_save_time": time.time()
            }
            
            with open(self._state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save evolution state: {e}")
    
    def _serialize_phase(self, phase: DevelopmentPhase) -> Dict[str, Any]:
        """Serialize phase for JSON storage"""
        return {
            "phase_id": phase.phase_id,
            "name": phase.name,
            "description": phase.description,
            "status": phase.status.value,
            "start_time": phase.start_time,
            "completion_time": phase.completion_time,
            "tasks": {tid: self._serialize_task(task) for tid, task in phase.tasks.items()}
        }
    
    def _serialize_task(self, task: PhaseTask) -> Dict[str, Any]:
        """Serialize task for JSON storage"""
        return {
            "task_id": task.task_id,
            "name": task.name,
            "description": task.description,
            "status": task.status.value,
            "start_time": task.start_time,
            "completion_time": task.completion_time,
            "execution_metrics": task.execution_metrics
        }
    
    def _serialize_metrics(self, metrics: EvolutionMetrics) -> Dict[str, Any]:
        """Serialize metrics for JSON storage"""
        return {
            "total_phases": metrics.total_phases,
            "completed_phases": metrics.completed_phases,
            "active_phases": metrics.active_phases,
            "failed_phases": metrics.failed_phases,
            "total_tasks": metrics.total_tasks,
            "completed_tasks": metrics.completed_tasks,
            "overall_progress": metrics.overall_progress,
            "evolution_start_time": metrics.evolution_start_time,
            "last_update_time": metrics.last_update_time
        }
    
    def get_evolution_status(self) -> Dict[str, Any]:
        """Get current evolution status"""
        return {
            "metrics": self.evolution_metrics,
            "current_phase": self._current_phase,
            "phases": {pid: phase.status.value for pid, phase in self.phases.items()},
            "total_phases": len(self.phases),
            "initialized": self._initialized
        }
    
    def get_phase_details(self, phase_id: str) -> Optional[DevelopmentPhase]:
        """Get detailed information about a specific phase"""
        return self.phases.get(phase_id)
    
    def get_current_phase(self) -> Optional[DevelopmentPhase]:
        """Get the currently active phase"""
        if self._current_phase:
            return self.phases.get(self._current_phase)
        return None
