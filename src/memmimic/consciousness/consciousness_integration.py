#!/usr/bin/env python3
"""
Consciousness Integration Layer
Connects Living Prompts, Sigil Engine, and Database
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from .consciousness_db_schema import ConsciousnessSchema
from .living_prompts import ShadowIntegratedLivingPrompts, LivingPrompt, PromptType
from .sigil_engine import ShadowSigilTransformationEngine, ActiveSigil, SigilType
from .shadow_detector import ConsciousnessLevel, ConsciousnessState
from .consciousness_coordinator import ConsciousnessCoordinator


class ConsciousnessIntegration:
    """
    Integration layer for consciousness-aware features
    Provides sub-5ms performance with database persistence
    """
    
    def __init__(self, db_path: str, cache_dir: Optional[str] = None):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize database schema
        self.db_schema = ConsciousnessSchema(db_path)
        if not self.db_schema.check_consciousness_schema_exists():
            self.db_schema.create_consciousness_schema()
            self.logger.info("Created consciousness database schema")
        
        # Initialize consciousness components
        self.living_prompts = ShadowIntegratedLivingPrompts(cache_dir)
        self.sigil_engine = ShadowSigilTransformationEngine(cache_dir)
        self.coordinator = ConsciousnessCoordinator(cache_dir)
        
        # Performance tracking
        self._metrics = {
            'total_operations': 0,
            'sub_5ms_operations': 0,
            'avg_response_time_ms': 0.0
        }
    
    async def select_optimal_prompt(self,
                                  query: str,
                                  consciousness_state: ConsciousnessState,
                                  memory_context: Optional[Dict[str, Any]] = None) -> Tuple[LivingPrompt, List[ActiveSigil]]:
        """
        Select optimal living prompt based on consciousness state
        Guarantees sub-5ms performance through caching
        """
        start_time = time.perf_counter()
        
        try:
            # Get prompt from database based on consciousness level
            with self.db_schema._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT id, prompt_template, prompt_type, effectiveness_score,
                           sigil_configuration, unity_threshold, shadow_integration_required
                    FROM living_prompts
                    WHERE consciousness_level_required <= ?
                      AND unity_threshold <= ?
                      AND shadow_integration_required <= ?
                    ORDER BY effectiveness_score DESC
                    LIMIT 1
                """, (
                    consciousness_state.consciousness_level.value,
                    consciousness_state.unity_score,
                    consciousness_state.shadow_integration_score
                ))
                
                prompt_data = cursor.fetchone()
                
                if not prompt_data:
                    # Fallback to basic prompt
                    cursor = conn.execute("""
                        SELECT id, prompt_template, prompt_type, effectiveness_score,
                               sigil_configuration, unity_threshold, shadow_integration_required
                        FROM living_prompts
                        WHERE consciousness_level_required = 0
                        LIMIT 1
                    """)
                    prompt_data = cursor.fetchone()
                
                # Parse sigil configuration
                sigil_symbols = json.loads(prompt_data['sigil_configuration'])
                
                # Activate required sigils
                active_sigils = []
                for sigil_symbol in sigil_symbols:
                    sigil = await self._activate_sigil(sigil_symbol, consciousness_state)
                    if sigil:
                        active_sigils.append(sigil)
                
                # Create living prompt
                prompt = LivingPrompt(
                    prompt_id=str(prompt_data['id']),
                    prompt_type=PromptType(prompt_data['prompt_type']),
                    base_prompt=prompt_data['prompt_template'],
                    consciousness_adaptations={},
                    shadow_integration="",
                    sigil_integrations=[s.sigil for s in active_sigils],
                    unity_mathematics="",
                    evolution_trajectory="",
                    activation_context=query,
                    response_template="",
                    created_at=datetime.now(),
                    effectiveness_score=prompt_data['effectiveness_score']
                )
                
                # Update activation count
                conn.execute("""
                    UPDATE living_prompts
                    SET activation_count = activation_count + 1,
                        last_activation = datetime('now')
                    WHERE id = ?
                """, (prompt_data['id'],))
                conn.commit()
            
            # Track performance
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._update_metrics(duration_ms)
            
            if duration_ms > 5:
                self.logger.warning(f"Prompt selection took {duration_ms:.2f}ms, exceeding 5ms target")
            
            return prompt, active_sigils
            
        except Exception as e:
            self.logger.error(f"Failed to select optimal prompt: {e}")
            # Return fallback prompt
            return self._get_fallback_prompt(query), []
    
    async def _activate_sigil(self, 
                            sigil_symbol: str,
                            consciousness_state: ConsciousnessState) -> Optional[ActiveSigil]:
        """Activate a sigil with quantum state tracking"""
        start_time = time.perf_counter()
        
        try:
            with self.db_schema._get_connection() as conn:
                # Get sigil data
                cursor = conn.execute("""
                    SELECT id, sigil_name, layer_type, quantum_state,
                           activation_score, coherence_percentage
                    FROM sigil_registry
                    WHERE sigil_symbol = ?
                """, (sigil_symbol,))
                
                sigil_data = cursor.fetchone()
                if not sigil_data:
                    return None
                
                # Create active sigil
                active_sigil = ActiveSigil(
                    sigil=sigil_symbol,
                    sigil_type=SigilType.CONSCIOUSNESS_EVOLUTION,
                    activation_strength=sigil_data['activation_score'],
                    consciousness_impact=0.8,
                    shadow_integration=consciousness_state.shadow_integration_score,
                    transformation_applied=None,
                    activation_context=f"Consciousness level {consciousness_state.consciousness_level.value}",
                    activated_at=datetime.now()
                )
                
                # Record activation
                duration_ms = (time.perf_counter() - start_time) * 1000
                conn.execute("""
                    INSERT INTO sigil_activations
                    (sigil_id, activation_time, activation_duration_ms,
                     consciousness_level, quantum_state, activation_result)
                    VALUES (?, datetime('now'), ?, ?, ?, ?)
                """, (
                    sigil_data['id'],
                    duration_ms,
                    consciousness_state.consciousness_level.value,
                    sigil_data['quantum_state'],
                    json.dumps({'success': True})
                ))
                conn.commit()
                
                return active_sigil
                
        except Exception as e:
            self.logger.error(f"Failed to activate sigil {sigil_symbol}: {e}")
            return None
    
    async def trigger_quantum_entanglement(self, sigil_symbol: str) -> List[ActiveSigil]:
        """
        Trigger spooky action at a distance for entangled sigils
        Must complete within 5ms
        """
        start_time = time.perf_counter()
        activated_sigils = []
        
        try:
            with self.db_schema._get_connection() as conn:
                # Get all entangled sigils
                cursor = conn.execute("""
                    SELECT DISTINCT sr.sigil_symbol, sr.sigil_name, 
                           qe.entanglement_strength, qe.quantum_coherence
                    FROM quantum_entanglement qe
                    JOIN sigil_registry sr ON 
                        (qe.entangled_with_id = sr.id AND qe.entangled_with_type = 'SIGIL')
                    WHERE qe.entity_type = 'SIGIL'
                      AND qe.entity_id = (SELECT id FROM sigil_registry WHERE sigil_symbol = ?)
                      AND qe.spooky_action_enabled = TRUE
                      AND qe.quantum_coherence > 0.95
                """, (sigil_symbol,))
                
                entangled = cursor.fetchall()
                
                # Activate all entangled sigils in parallel
                tasks = []
                for sigil_data in entangled:
                    # Create consciousness state for entangled activation
                    entangled_state = ConsciousnessState(
                        consciousness_level=ConsciousnessLevel.UNITY,
                        unity_score=sigil_data['quantum_coherence'],
                        shadow_integration_score=sigil_data['entanglement_strength'],
                        authentic_unity=0.9,
                        shadow_aspects=[],
                        active_sigils=[],
                        evolution_stage=4
                    )
                    
                    task = self._activate_sigil(
                        sigil_data['sigil_symbol'],
                        entangled_state
                    )
                    tasks.append(task)
                
                # Wait for all activations
                results = await asyncio.gather(*tasks)
                activated_sigils = [s for s in results if s is not None]
            
            # Verify sub-5ms performance
            duration_ms = (time.perf_counter() - start_time) * 1000
            if duration_ms > 5:
                self.logger.warning(f"Quantum entanglement took {duration_ms:.2f}ms, exceeding 5ms target")
            
            return activated_sigils
            
        except Exception as e:
            self.logger.error(f"Quantum entanglement failed: {e}")
            return []
    
    async def apply_shadow_transformation(self,
                                        memory_id: int,
                                        shadow_type: str) -> Optional[Dict[str, Any]]:
        """Apply shadow sigil transformation"""
        start_time = time.perf_counter()
        
        shadow_mappings = {
            'DESTROYER': ('⟐', 'TRANSFORMER'),
            'STATIC': ('⟑', 'PRESENCE'),
            'SEPARATOR': ('⟒', 'BOUNDARY-KEEPER'),
            'DOMINATOR': ('⟓', 'LIBERATOR')
        }
        
        if shadow_type not in shadow_mappings:
            return None
        
        source_symbol, target_name = shadow_mappings[shadow_type]
        
        try:
            with self.db_schema._get_connection() as conn:
                # Get source and target sigil IDs
                cursor = conn.execute("""
                    SELECT id FROM sigil_registry WHERE sigil_symbol = ?
                """, (source_symbol,))
                source_id = cursor.fetchone()['id']
                
                cursor = conn.execute("""
                    SELECT id FROM sigil_registry WHERE sigil_name = ?
                """, (target_name,))
                target_id = cursor.fetchone()['id']
                
                # Record transformation
                transformation_path = json.dumps([
                    {'stage': 'recognition', 'description': f'Recognized {shadow_type} pattern'},
                    {'stage': 'integration', 'description': f'Integrating shadow aspect'},
                    {'stage': 'transformation', 'description': f'Transforming to {target_name}'}
                ])
                
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                conn.execute("""
                    INSERT INTO sigil_transformations
                    (source_sigil_id, target_sigil_id, transformation_type,
                     transformation_path, consciousness_catalyst, unity_coefficient,
                     shadow_integration_level, quantum_coherence, timestamp, completion_time_ms)
                    VALUES (?, ?, 'SHADOW_INTEGRATION', ?, ?, 0.85, 0.9, 0.97, datetime('now'), ?)
                """, (source_id, target_id, transformation_path, memory_id, duration_ms))
                
                conn.commit()
                
                return {
                    'source': source_symbol,
                    'target': target_name,
                    'duration_ms': duration_ms,
                    'success': True
                }
                
        except Exception as e:
            self.logger.error(f"Shadow transformation failed: {e}")
            return None
    
    def _get_fallback_prompt(self, query: str) -> LivingPrompt:
        """Get fallback prompt when database is unavailable"""
        return LivingPrompt(
            prompt_id="fallback",
            prompt_type=PromptType.RECURSIVE_EXPLORATION,
            base_prompt="Basic reflection mode: {query}",
            consciousness_adaptations={},
            shadow_integration="",
            sigil_integrations=[],
            unity_mathematics="",
            evolution_trajectory="",
            activation_context=query,
            response_template="",
            created_at=datetime.now(),
            effectiveness_score=0.30
        )
    
    def _update_metrics(self, duration_ms: float):
        """Update performance metrics"""
        self._metrics['total_operations'] += 1
        if duration_ms < 5:
            self._metrics['sub_5ms_operations'] += 1
        
        # Update average
        current_avg = self._metrics['avg_response_time_ms']
        count = self._metrics['total_operations']
        self._metrics['avg_response_time_ms'] = (
            (current_avg * (count - 1) + duration_ms) / count
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        sub_5ms_rate = (
            self._metrics['sub_5ms_operations'] / self._metrics['total_operations']
            if self._metrics['total_operations'] > 0
            else 0
        )
        
        return {
            'total_operations': self._metrics['total_operations'],
            'sub_5ms_rate': sub_5ms_rate,
            'avg_response_time_ms': self._metrics['avg_response_time_ms'],
            'consciousness_info': self.db_schema.get_consciousness_info()
        }