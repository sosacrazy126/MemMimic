#!/usr/bin/env python3
"""
Updated MCP tools that work with the storage adapter
Supports both SQLite and Markdown backends transparently
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Import the storage adapter
from storage_adapter import create_storage_adapter, Memory
# Import enhanced thinking system
from enhanced_think_with_memory import EnhancedThinkWithMemory
# Import enhanced CXD classifier
from enhanced_cxd_classifier import EnhancedCXDClassifier

# Configuration - can be overridden by environment variables
STORAGE_TYPE = os.environ.get('MEMMIMIC_STORAGE', 'hybrid')  # 'sqlite', 'markdown', or 'hybrid'
SQLITE_PATH = os.environ.get('MEMMIMIC_DB_PATH', 'memmimic.db')
MARKDOWN_DIR = os.environ.get('MEMMIMIC_MD_DIR', '.')
WRITE_TO = os.environ.get('MEMMIMIC_WRITE_TO', 'both')  # For hybrid: 'sqlite', 'markdown', or 'both'


class MemMimicMCP:
    """Updated MCP interface using storage adapter"""
    
    def __init__(self):
        """Initialize with configured storage adapter"""
        self.adapter = create_storage_adapter(
            storage_type=STORAGE_TYPE,
            sqlite_path=SQLITE_PATH,
            markdown_dir=MARKDOWN_DIR,
            write_to=WRITE_TO
        )

        # For backward compatibility
        self.storage = self.adapter

        # Initialize enhanced CXD classifier
        self.cxd_classifier = EnhancedCXDClassifier()
    
    def remember(self, content: str, memory_type: str = "interaction", 
                 metadata: Dict = None) -> Dict:
        """
        Store a memory
        
        Args:
            content: Memory content
            memory_type: Type of memory (interaction, reflection, milestone)
            metadata: Additional metadata
        
        Returns:
            Success response with memory ID
        """
        try:
            # Prepare metadata
            full_metadata = metadata or {}
            full_metadata['type'] = memory_type

            # Auto-classify with CXD
            cxd_result = self.cxd_classifier.classify(content)
            cxd_type = cxd_result.function
            cxd_confidence = cxd_result.confidence

            full_metadata['cxd'] = cxd_type
            full_metadata['cxd_confidence'] = round(cxd_confidence, 2)

            # Calculate importance
            importance = self._calculate_importance(content, memory_type)

            # Calculate quality score
            quality = self._calculate_quality_score(content, full_metadata)
            full_metadata['quality'] = quality

            # Detect duplicates
            duplicates = self._detect_duplicates(content, threshold=0.8)

            # Find similar memories for relationships
            similar = self._find_similar_memories(content, limit=5)

            # Build relationships metadata
            relationships = {
                'similar_memories': [s['id'] for s in similar],
                'relationship_strength': similar[0]['similarity'] if similar else 0.0
            }
            full_metadata['relationships'] = relationships

            # Extract tags
            tags = self._extract_tags(content, cxd_type)
            full_metadata['tags'] = tags

            # Create memory object
            memory = Memory(
                content=content,
                metadata=full_metadata,
                importance=importance
            )

            # Store using adapter
            memory_id = self.adapter.store(memory)

            return {
                'status': 'success',
                'memory_id': memory_id,
                'type': memory_type,
                'cxd': cxd_type,
                'cxd_confidence': cxd_confidence,
                'importance': importance,
                'quality_score': quality['overall_score'],
                'auto_approved': quality['auto_approved'],
                'duplicates_found': len(duplicates),
                'duplicate_ids': duplicates if duplicates else None,
                'similar_memories': len(similar),
                'tags': tags,
                'storage': STORAGE_TYPE
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def recall_cxd(self, query: str, function_filter: str = "ALL",
                   limit: int = 10) -> List[Dict]:
        """
        Recall memories with CXD filtering

        Args:
            query: Search query
            function_filter: CXD filter (CONTROL, CONTEXT, DATA, or ALL)
            limit: Maximum results

        Returns:
            List of matching memories
        """
        try:
            # Search using adapter
            memories = self.adapter.search(query, limit * 2)  # Get extra for filtering

            # Filter by CXD if specified
            if function_filter != "ALL":
                memories = [m for m in memories
                          if m.metadata.get('cxd', '').upper() == function_filter.upper()]

            # Convert to response format
            results = []
            for memory in memories[:limit]:
                results.append({
                    'id': memory.id,
                    'content': memory.content,
                    'metadata': memory.metadata,
                    'importance': memory.importance,
                    'created': memory.created_at.isoformat(),
                    'cxd': memory.metadata.get('cxd', 'unknown'),
                    'type': memory.metadata.get('type', 'interaction')
                })

            return results

        except Exception as e:
            return [{
                'error': str(e),
                'status': 'error'
            }]

    def advanced_search(self, query: str = "", filters: Dict = None, limit: int = 10) -> Dict:
        """
        Advanced search with multiple filters

        Args:
            query: Search query (optional)
            filters: Dict with filter criteria:
                - cxd: CXD type (CONTROL, CONTEXT, DATA, unknown)
                - min_quality: Minimum quality score (0.0-1.0)
                - max_quality: Maximum quality score (0.0-1.0)
                - min_importance: Minimum importance (0.0-1.0)
                - memory_type: Type of memory (interaction, milestone, etc.)
                - tags: List of tags to match (any match)
                - has_relationships: True/False
                - created_after: ISO date string
                - created_before: ISO date string
            limit: Maximum results

        Returns:
            Dict with results and filter stats
        """
        try:
            filters = filters or {}

            # Get all memories or search results
            if query:
                memories = self.adapter.search(query, limit * 5)  # Get extra for filtering
            else:
                memories = self.adapter.get_all(limit * 5)

            # Apply filters
            filtered = []
            filter_stats = {
                'total_scanned': len(memories),
                'filters_applied': [],
                'results_after_filters': 0
            }

            for memory in memories:
                metadata = memory.metadata

                # CXD filter
                if 'cxd' in filters:
                    if metadata.get('cxd', '').upper() != filters['cxd'].upper():
                        continue
                    if 'cxd' not in filter_stats['filters_applied']:
                        filter_stats['filters_applied'].append('cxd')

                # Quality filters
                quality_score = metadata.get('quality', {}).get('overall_score', 0.5)
                if 'min_quality' in filters:
                    if quality_score < filters['min_quality']:
                        continue
                    if 'min_quality' not in filter_stats['filters_applied']:
                        filter_stats['filters_applied'].append('min_quality')

                if 'max_quality' in filters:
                    if quality_score > filters['max_quality']:
                        continue
                    if 'max_quality' not in filter_stats['filters_applied']:
                        filter_stats['filters_applied'].append('max_quality')

                # Importance filter
                if 'min_importance' in filters:
                    if memory.importance < filters['min_importance']:
                        continue
                    if 'min_importance' not in filter_stats['filters_applied']:
                        filter_stats['filters_applied'].append('min_importance')

                # Memory type filter
                if 'memory_type' in filters:
                    if metadata.get('type', '') != filters['memory_type']:
                        continue
                    if 'memory_type' not in filter_stats['filters_applied']:
                        filter_stats['filters_applied'].append('memory_type')

                # Tags filter (match any)
                if 'tags' in filters and filters['tags']:
                    memory_tags = metadata.get('tags', [])
                    if not any(tag in memory_tags for tag in filters['tags']):
                        continue
                    if 'tags' not in filter_stats['filters_applied']:
                        filter_stats['filters_applied'].append('tags')

                # Relationships filter
                if 'has_relationships' in filters:
                    has_rels = len(metadata.get('relationships', {}).get('similar_memories', [])) > 0
                    if has_rels != filters['has_relationships']:
                        continue
                    if 'has_relationships' not in filter_stats['filters_applied']:
                        filter_stats['filters_applied'].append('has_relationships')

                # Date filters
                if 'created_after' in filters:
                    from datetime import datetime
                    after_date = datetime.fromisoformat(filters['created_after'])
                    if memory.created_at < after_date:
                        continue
                    if 'created_after' not in filter_stats['filters_applied']:
                        filter_stats['filters_applied'].append('created_after')

                if 'created_before' in filters:
                    from datetime import datetime
                    before_date = datetime.fromisoformat(filters['created_before'])
                    if memory.created_at > before_date:
                        continue
                    if 'created_before' not in filter_stats['filters_applied']:
                        filter_stats['filters_applied'].append('created_before')

                # Passed all filters
                filtered.append(memory)

            filter_stats['results_after_filters'] = len(filtered)

            # Convert to response format
            results = []
            for memory in filtered[:limit]:
                results.append({
                    'id': memory.id,
                    'content': memory.content,
                    'metadata': memory.metadata,
                    'importance': memory.importance,
                    'created': memory.created_at.isoformat(),
                    'quality_score': memory.metadata.get('quality', {}).get('overall_score', 0.5),
                    'cxd': memory.metadata.get('cxd', 'unknown'),
                    'type': memory.metadata.get('type', 'interaction'),
                    'tags': memory.metadata.get('tags', [])
                })

            return {
                'status': 'success',
                'results': results,
                'count': len(results),
                'filter_stats': filter_stats
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def think_with_memory(self, input_text: str, mode: str = 'enhanced', max_thoughts: int = 10) -> Dict:
        """
        Process input with memory context
        
        Args:
            input_text: Input to process
            mode: 'simple' for basic search, 'enhanced' for sequential thinking
            max_thoughts: Maximum thoughts for enhanced mode
        
        Returns:
            Response with relevant memories and analysis
        """
        try:
            if mode == 'enhanced':
                # Use the enhanced thinking system
                thinker = EnhancedThinkWithMemory(self.adapter)
                return thinker.think(input_text, max_thoughts=max_thoughts)
            else:
                # Use simple mode (legacy)
                # Find relevant memories
                memories = self.adapter.search(input_text, limit=5)
                
                # Build context
                context = []
                for memory in memories:
                    context.append({
                        'content': memory.content[:200],  # Preview
                        'relevance': self._calculate_relevance(input_text, memory.content),
                        'cxd': memory.metadata.get('cxd', 'unknown')
                    })
                
                # Sort by relevance
                context.sort(key=lambda x: x['relevance'], reverse=True)
                
                # Generate response
                response = {
                    'status': 'success',
                    'input': input_text,
                    'relevant_memories': len(memories),
                    'context': context[:3],  # Top 3 most relevant
                    'analysis': self._generate_analysis(input_text, memories),
                    'storage': STORAGE_TYPE
                }
                
                return response
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_analytics(self) -> Dict:
        """
        Get comprehensive memory analytics and insights

        Returns:
            Dict with analytics including quality trends, tag clouds, relationship metrics
        """
        try:
            from collections import Counter
            from datetime import datetime, timedelta

            # Get all memories
            all_memories = self.adapter.get_all(limit=500)

            analytics = {
                'total_memories': len(all_memories),
                'quality_analysis': {},
                'tag_cloud': {},
                'relationship_metrics': {},
                'temporal_analysis': {},
                'cxd_insights': {}
            }

            # Initialize counters
            quality_scores = []
            all_tags = []
            cxd_by_quality = {'CONTROL': [], 'CONTEXT': [], 'DATA': [], 'unknown': []}
            relationship_counts = []
            memories_by_month = Counter()

            for memory in all_memories:
                metadata = memory.metadata

                # Quality analysis
                quality = metadata.get('quality', {})
                quality_score = quality.get('overall_score', 0.5)
                quality_scores.append(quality_score)

                # Tag collection
                tags = metadata.get('tags', [])
                all_tags.extend(tags)

                # CXD quality correlation
                cxd = metadata.get('cxd', 'unknown')
                cxd_by_quality[cxd].append(quality_score)

                # Relationship metrics
                rels = metadata.get('relationships', {})
                similar_count = len(rels.get('similar_memories', []))
                relationship_counts.append(similar_count)

                # Temporal analysis
                month_key = memory.created_at.strftime('%Y-%m')
                memories_by_month[month_key] += 1

            # Quality Analysis
            if quality_scores:
                analytics['quality_analysis'] = {
                    'average_quality': round(sum(quality_scores) / len(quality_scores), 2),
                    'min_quality': round(min(quality_scores), 2),
                    'max_quality': round(max(quality_scores), 2),
                    'high_quality_count': sum(1 for q in quality_scores if q >= 0.7),
                    'low_quality_count': sum(1 for q in quality_scores if q < 0.5),
                    'distribution': {
                        'excellent (0.8+)': sum(1 for q in quality_scores if q >= 0.8),
                        'good (0.6-0.8)': sum(1 for q in quality_scores if 0.6 <= q < 0.8),
                        'fair (0.4-0.6)': sum(1 for q in quality_scores if 0.4 <= q < 0.6),
                        'poor (<0.4)': sum(1 for q in quality_scores if q < 0.4)
                    }
                }

            # Tag Cloud (top 20)
            tag_counts = Counter(all_tags)
            analytics['tag_cloud'] = dict(tag_counts.most_common(20))

            # Relationship Metrics
            if relationship_counts:
                analytics['relationship_metrics'] = {
                    'average_relationships': round(sum(relationship_counts) / len(relationship_counts), 2),
                    'max_relationships': max(relationship_counts),
                    'isolated_memories': sum(1 for c in relationship_counts if c == 0),
                    'well_connected': sum(1 for c in relationship_counts if c >= 3)
                }

            # CXD Quality Insights
            for cxd, scores in cxd_by_quality.items():
                if scores:
                    analytics['cxd_insights'][cxd] = {
                        'count': len(scores),
                        'avg_quality': round(sum(scores) / len(scores), 2),
                        'quality_trend': 'high' if sum(scores) / len(scores) > 0.6 else 'medium' if sum(scores) / len(scores) > 0.4 else 'low'
                    }

            # Temporal Analysis
            analytics['temporal_analysis'] = {
                'memories_by_month': dict(sorted(memories_by_month.items(), reverse=True)[:6]),
                'most_active_month': max(memories_by_month.items(), key=lambda x: x[1])[0] if memories_by_month else None,
                'recent_7_days': sum(1 for m in all_memories if (datetime.now() - m.created_at).days <= 7),
                'recent_30_days': sum(1 for m in all_memories if (datetime.now() - m.created_at).days <= 30)
            }

            return {
                'status': 'success',
                'analytics': analytics
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    def status(self) -> Dict:
        """
        Get system status
        
        Returns:
            Status information
        """
        try:
            total_memories = self.adapter.count()
            
            # Get recent memories
            recent = self.adapter.get_all(limit=5)
            
            # Calculate statistics
            stats = {
                'total_memories': total_memories,
                'storage_type': STORAGE_TYPE,
                'recent_memories': len(recent)
            }
            
            # Add storage-specific info
            if STORAGE_TYPE == 'markdown':
                stats['markdown_dir'] = MARKDOWN_DIR
                stats['index_exists'] = (Path(MARKDOWN_DIR) / 'memories' / 'index.json').exists()
            elif STORAGE_TYPE == 'sqlite':
                stats['database_path'] = SQLITE_PATH
                stats['database_exists'] = Path(SQLITE_PATH).exists()
            elif STORAGE_TYPE == 'hybrid':
                stats['sqlite_path'] = SQLITE_PATH
                stats['markdown_dir'] = MARKDOWN_DIR
                stats['write_to'] = WRITE_TO
            
            # Memory type breakdown
            type_counts = {}
            for memory in self.adapter.get_all(limit=100):
                mem_type = memory.metadata.get('type', 'unknown')
                type_counts[mem_type] = type_counts.get(mem_type, 0) + 1
            
            stats['memory_types'] = type_counts
            
            # CXD breakdown with confidence averages
            cxd_counts = {}
            cxd_confidences = {}
            for memory in self.adapter.get_all(limit=100):
                cxd = memory.metadata.get('cxd', 'unknown')
                confidence = memory.metadata.get('cxd_confidence', 0.0)

                cxd_counts[cxd] = cxd_counts.get(cxd, 0) + 1

                # Track confidences for averaging
                if cxd not in cxd_confidences:
                    cxd_confidences[cxd] = []
                cxd_confidences[cxd].append(confidence)

            stats['cxd_distribution'] = cxd_counts

            # Add average confidence per category
            cxd_avg_confidence = {}
            for cxd, confidences in cxd_confidences.items():
                if confidences:
                    avg = sum(confidences) / len(confidences)
                    cxd_avg_confidence[cxd] = round(avg, 2)

            stats['cxd_confidence'] = cxd_avg_confidence

            # Add tale statistics
            tale_stats = self._get_tale_statistics()
            stats['tale_stats'] = tale_stats

            return {
                'status': 'success',
                'stats': stats
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def update_memory(self, memory_id: str, content: str = None, 
                     metadata: Dict = None) -> Dict:
        """
        Update an existing memory
        
        Args:
            memory_id: Memory ID to update
            content: New content (optional)
            metadata: New metadata (optional)
        
        Returns:
            Success response
        """
        try:
            # Retrieve existing memory
            memory = self.adapter.retrieve(memory_id)
            if not memory:
                return {
                    'status': 'error',
                    'error': f"Memory {memory_id} not found"
                }
            
            # Update fields
            if content:
                memory.content = content
            if metadata:
                memory.metadata.update(metadata)
            
            # Update importance if content changed
            if content:
                memory.importance = self._calculate_importance(content, 
                                                              memory.metadata.get('type', 'interaction'))
            
            # Save updates
            success = self.adapter.update(memory_id, memory)
            
            return {
                'status': 'success' if success else 'error',
                'memory_id': memory_id,
                'updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def export_memories(self, filters: Dict = None, format: str = "json") -> Dict:
        """
        Export memories to JSON or markdown archive.

        Args:
            filters: Optional filters (same as advanced_search)
            format: 'json' or 'markdown'

        Returns:
            Export data or status
        """
        try:
            from datetime import datetime
            import json

            # Get memories using advanced search if filters provided
            if filters:
                result = self.advanced_search(query="", filters=filters, limit=1000)
                if result['status'] == 'error':
                    return result
                memories = result['results']
            else:
                # Get all memories
                all_mem = self.adapter.get_all(limit=1000)
                memories = [{
                    'id': m.id,
                    'content': m.content,
                    'metadata': m.metadata,
                    'importance': m.importance,
                    'created': m.created_at.isoformat(),
                    'quality_score': m.metadata.get('quality', {}).get('overall_score', 0.5),
                    'cxd': m.metadata.get('cxd', 'unknown'),
                    'type': m.metadata.get('type', 'interaction'),
                    'tags': m.metadata.get('tags', [])
                } for m in all_mem]

            if format == "json":
                export_data = {
                    'export_date': datetime.now().isoformat(),
                    'total_memories': len(memories),
                    'memories': memories
                }

                return {
                    'status': 'success',
                    'format': 'json',
                    'data': export_data,
                    'count': len(memories)
                }

            elif format == "markdown":
                # Create markdown export
                lines = [
                    "# MemMimic Memory Export",
                    f"\n**Export Date:** {datetime.now().isoformat()}",
                    f"**Total Memories:** {len(memories)}\n",
                    "---\n"
                ]

                for mem in memories:
                    lines.append(f"\n## {mem['id']}\n")
                    lines.append(f"**Type:** {mem['type']} | **CXD:** {mem['cxd']} | **Quality:** {mem['quality_score']:.2f}\n")
                    lines.append(f"**Tags:** {', '.join(mem['tags'][:5])}\n")
                    lines.append(f"**Created:** {mem['created']}\n")
                    lines.append(f"\n{mem['content']}\n")
                    lines.append("\n---\n")

                export_text = "\n".join(lines)

                return {
                    'status': 'success',
                    'format': 'markdown',
                    'data': export_text,
                    'count': len(memories)
                }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    def cleanup_memories(self, criteria: str = "low_quality", threshold: float = 0.4,
                        dry_run: bool = True) -> Dict:
        """
        Identify or delete memories based on cleanup criteria.

        Args:
            criteria: 'low_quality', 'isolated', 'old', or 'duplicates'
            threshold: Threshold value (quality score, age in days, similarity)
            dry_run: If True, only report what would be deleted

        Returns:
            Dict with cleanup report
        """
        try:
            from datetime import datetime, timedelta

            candidates = []
            all_memories = self.adapter.get_all(limit=500)

            if criteria == "low_quality":
                # Find memories below quality threshold
                for mem in all_memories:
                    quality = mem.metadata.get('quality', {}).get('overall_score', 0.5)
                    if quality < threshold:
                        candidates.append({
                            'id': mem.id,
                            'reason': f'Low quality ({quality:.2f} < {threshold})',
                            'quality': quality,
                            'content_preview': mem.content[:100]
                        })

            elif criteria == "isolated":
                # Find memories with no relationships
                for mem in all_memories:
                    rels = mem.metadata.get('relationships', {})
                    similar = rels.get('similar_memories', [])
                    if len(similar) == 0:
                        candidates.append({
                            'id': mem.id,
                            'reason': 'No relationships',
                            'content_preview': mem.content[:100]
                        })

            elif criteria == "old":
                # Find memories older than threshold days
                cutoff_date = datetime.now() - timedelta(days=threshold)
                for mem in all_memories:
                    if mem.created_at < cutoff_date:
                        days_old = (datetime.now() - mem.created_at).days
                        candidates.append({
                            'id': mem.id,
                            'reason': f'Old ({days_old} days)',
                            'created': mem.created_at.isoformat(),
                            'content_preview': mem.content[:100]
                        })

            elif criteria == "duplicates":
                # Find near-duplicate memories
                seen = set()
                for mem in all_memories:
                    # Get similar memories above threshold
                    similar = self._find_similar_memories(mem.content, limit=10)
                    for sim in similar:
                        if sim['similarity'] >= threshold and sim['id'] != mem.id:
                            if sim['id'] not in seen:
                                candidates.append({
                                    'id': sim['id'],
                                    'reason': f"Duplicate of {mem.id} ({sim['similarity']:.2f} similarity)",
                                    'similarity': sim['similarity'],
                                    'content_preview': sim['preview']
                                })
                                seen.add(sim['id'])

            # Delete if not dry run
            deleted = []
            if not dry_run:
                for candidate in candidates:
                    success = self.adapter.delete(candidate['id'])
                    if success:
                        deleted.append(candidate['id'])

            return {
                'status': 'success',
                'criteria': criteria,
                'threshold': threshold,
                'dry_run': dry_run,
                'candidates_found': len(candidates),
                'deleted': len(deleted),
                'candidates': candidates[:20],  # Limit to 20 for display
                'message': 'Dry run - no changes made' if dry_run else f'Deleted {len(deleted)} memories'
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    def bulk_update(self, filters: Dict, updates: Dict, dry_run: bool = True) -> Dict:
        """
        Bulk update memories matching filters.

        Args:
            filters: Filter criteria (same as advanced_search)
            updates: Dict of fields to update:
                - importance: New importance value
                - memory_type: New type
                - tags_add: Tags to add
                - tags_remove: Tags to remove
            dry_run: If True, only report what would be updated

        Returns:
            Dict with update report
        """
        try:
            # Find matching memories
            result = self.advanced_search(query="", filters=filters, limit=500)
            if result['status'] == 'error':
                return result

            matching = result['results']
            updated = []

            # Get all memories to work with
            all_memories = self.adapter.get_all()
            memory_map = {m.id: m for m in all_memories}

            for mem_data in matching:
                memory_id = mem_data['id']
                memory = memory_map.get(memory_id)

                if not memory:
                    continue

                changed = False

                # Update importance
                if 'importance' in updates:
                    memory.importance = updates['importance']
                    changed = True

                # Update type
                if 'memory_type' in updates:
                    memory.metadata['type'] = updates['memory_type']
                    changed = True

                # Add tags
                if 'tags_add' in updates:
                    current_tags = set(memory.metadata.get('tags', []))
                    new_tags = set(updates['tags_add'])
                    memory.metadata['tags'] = sorted(list(current_tags | new_tags))
                    changed = True

                # Remove tags
                if 'tags_remove' in updates:
                    current_tags = set(memory.metadata.get('tags', []))
                    remove_tags = set(updates['tags_remove'])
                    memory.metadata['tags'] = sorted(list(current_tags - remove_tags))
                    changed = True

                if changed:
                    if not dry_run:
                        self.adapter.update(memory_id, memory)
                    updated.append(memory_id)

            return {
                'status': 'success',
                'dry_run': dry_run,
                'matched': len(matching),
                'updated': len(updated),
                'updated_ids': updated[:20],  # Limit to 20 for display
                'message': 'Dry run - no changes made' if dry_run else f'Updated {len(updated)} memories'
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    def delete_memory(self, memory_id: str) -> Dict:
        """
        Delete a memory
        
        Args:
            memory_id: Memory ID to delete
        
        Returns:
            Success response
        """
        try:
            success = self.adapter.delete(memory_id)
            
            return {
                'status': 'success' if success else 'error',
                'memory_id': memory_id,
                'deleted': success
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def migrate_to_markdown(self) -> Dict:
        """
        Special tool to migrate from SQLite to Markdown
        
        Returns:
            Migration status
        """
        try:
            if STORAGE_TYPE != 'hybrid':
                return {
                    'status': 'error',
                    'error': 'Migration requires hybrid storage mode'
                }
            
            # Get all memories from SQLite
            sqlite_adapter = create_storage_adapter('sqlite', db_path=SQLITE_PATH)
            all_memories = sqlite_adapter.get_all()
            
            # Store each in markdown
            markdown_adapter = create_storage_adapter('markdown', base_dir=MARKDOWN_DIR)
            migrated = 0
            failed = 0
            
            for memory in all_memories:
                try:
                    markdown_adapter.store(memory)
                    migrated += 1
                except Exception as e:
                    print(f"Failed to migrate {memory.id}: {e}")
                    failed += 1
            
            return {
                'status': 'success',
                'total': len(all_memories),
                'migrated': migrated,
                'failed': failed
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    # Helper methods

    def _get_tale_statistics(self) -> Dict:
        """Get tale statistics from filesystem"""
        from datetime import datetime, timedelta

        tales_dir = Path(MARKDOWN_DIR) / 'tales'

        if not tales_dir.exists():
            return {
                'total_tales': 0,
                'by_category': {},
                'recent_tales': 0
            }

        # Count tales by category
        by_category = {}
        total = 0
        recent = 0
        week_ago = datetime.now() - timedelta(days=7)

        for tale_file in tales_dir.rglob('*.md'):
            total += 1

            # Extract category from path (tales/category/name.md)
            if len(tale_file.parts) > 2:
                category = tale_file.parts[-2]
            else:
                category = 'general'

            by_category[category] = by_category.get(category, 0) + 1

            # Check if recent
            if tale_file.stat().st_mtime > week_ago.timestamp():
                recent += 1

        return {
            'total_tales': total,
            'by_category': by_category,
            'recent_tales': recent
        }

    def _calculate_quality_score(self, content: str, metadata: Dict) -> Dict:
        """
        Calculate quality score for a memory based on multiple dimensions.

        Returns a dict with dimension scores and overall score.
        """
        dimensions = {}

        # 1. Clarity (0.0 - 1.0)
        # Based on length, structure, formatting
        word_count = len(content.split())
        if word_count > 50 and word_count < 1000:
            clarity = 0.8
        elif word_count > 20:
            clarity = 0.6
        else:
            clarity = 0.4

        # Boost for structured content (lists, headings)
        if any(marker in content for marker in ['#', '-', '*', '1.', '2.']):
            clarity += 0.2

        dimensions['clarity'] = min(1.0, clarity)

        # 2. Information Density (0.0 - 1.0)
        # Higher for content with specific details
        unique_words = len(set(content.lower().split()))
        total_words = len(content.split())
        lexical_diversity = unique_words / total_words if total_words > 0 else 0

        dimensions['information_density'] = min(1.0, lexical_diversity * 1.5)

        # 3. Contextual Relevance (0.0 - 1.0)
        # Based on CXD classification confidence
        cxd_confidence = metadata.get('cxd_confidence', 0.5)
        dimensions['contextual_relevance'] = cxd_confidence

        # 4. Uniqueness (0.0 - 1.0)
        # Higher for longer, more specific content
        # (Simple heuristic - could be enhanced with similarity checking)
        if word_count > 200:
            uniqueness = 0.9
        elif word_count > 100:
            uniqueness = 0.7
        else:
            uniqueness = 0.5

        dimensions['uniqueness'] = uniqueness

        # 5. Importance Potential (0.0 - 1.0)
        # Based on memory type and explicit importance
        importance = metadata.get('importance', 0.5)
        memory_type = metadata.get('type', 'interaction')

        type_multiplier = {
            'milestone': 1.2,
            'reflection': 1.1,
            'technical': 1.0,
            'interaction': 0.9,
            'test': 0.5
        }.get(memory_type, 1.0)

        dimensions['importance_potential'] = min(1.0, importance * type_multiplier)

        # 6. Factual Accuracy (0.0 - 1.0)
        # Higher for content with verifiable details
        # Check for dates, numbers, specific references
        has_specifics = any([
            any(char.isdigit() for char in content),  # Contains numbers
            any(word in content.lower() for word in ['date', 'time', 'version']),
            '://' in content  # URLs
        ])

        dimensions['factual_accuracy'] = 0.7 if has_specifics else 0.5

        # Calculate overall score (weighted average)
        weights = {
            'clarity': 0.15,
            'information_density': 0.20,
            'contextual_relevance': 0.20,
            'uniqueness': 0.15,
            'importance_potential': 0.20,
            'factual_accuracy': 0.10
        }

        overall = sum(dimensions[dim] * weights[dim] for dim in dimensions)

        return {
            'dimensions': dimensions,
            'overall_score': round(overall, 2),
            'auto_approved': overall >= 0.6  # Auto-approve if quality >= 60%
        }

    def _calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate Jaccard similarity between two texts.
        Returns similarity score between 0 and 1.
        """
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1 & tokens2
        union = tokens1 | tokens2

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def _find_similar_memories(self, content: str, limit: int = 5) -> List[Dict]:
        """
        Find similar memories based on content overlap.
        Returns list of similar memories with similarity scores.
        """
        similar = []

        # Get all memories
        all_memories = self.adapter.get_all(limit=200)

        for memory in all_memories:
            # Calculate Jaccard similarity
            similarity = self._calculate_jaccard_similarity(content, memory.content)

            # Only consider memories with >30% similarity
            if similarity > 0.3:
                    similar.append({
                        'id': memory.id,
                        'similarity': round(similarity, 2),
                        'preview': memory.content[:100]
                    })

        # Sort by similarity (descending) and return top results
        similar.sort(key=lambda x: x['similarity'], reverse=True)
        return similar[:limit]

    def _detect_duplicates(self, content: str, threshold: float = 0.8) -> List[str]:
        """
        Detect potential duplicate memories.
        Returns list of memory IDs that are likely duplicates (>80% similar).
        """
        similar = self._find_similar_memories(content, limit=10)

        # Filter for high similarity (likely duplicates)
        duplicates = [
            mem['id'] for mem in similar
            if mem['similarity'] >= threshold
        ]

        return duplicates

    def _extract_tags(self, content: str, cxd: str) -> List[str]:
        """
        Auto-extract tags from memory content.
        Returns list of relevant tags based on content and CXD classification.
        """
        content_lower = content.lower()
        tags = set()

        # CXD-based tags
        if cxd == 'CONTROL':
            cxd_keywords = ['search', 'filter', 'execute', 'manage', 'control',
                           'decision', 'action', 'command', 'process']
        elif cxd == 'CONTEXT':
            cxd_keywords = ['relate', 'context', 'reference', 'understand',
                           'explain', 'why', 'because', 'relationship']
        elif cxd == 'DATA':
            cxd_keywords = ['data', 'process', 'transform', 'generate',
                           'extract', 'analyze', 'metric', 'statistic']
        else:
            cxd_keywords = []

        # Add matching CXD keywords as tags
        for keyword in cxd_keywords:
            if keyword in content_lower:
                tags.add(keyword)

        # Technical terms (common programming/system concepts)
        tech_terms = [
            'api', 'database', 'memory', 'system', 'architecture',
            'algorithm', 'performance', 'optimization', 'test', 'bug',
            'feature', 'integration', 'deployment', 'security', 'auth',
            'cache', 'queue', 'async', 'sync', 'model', 'classifier',
            'score', 'metric', 'quality', 'confidence'
        ]

        for term in tech_terms:
            if term in content_lower:
                tags.add(term)

        # Domain-specific terms
        domain_terms = [
            'cxd', 'classification', 'memmimic', 'tale', 'amplifier',
            'knowledge', 'synthesis', 'embedding', 'vector', 'search'
        ]

        for term in domain_terms:
            if term in content_lower:
                tags.add(term)

        # Extract potential acronyms (2-5 uppercase letters)
        import re
        acronyms = re.findall(r'\b[A-Z]{2,5}\b', content)
        for acronym in acronyms[:5]:  # Limit to 5 acronyms
            if acronym not in ['THE', 'AND', 'FOR']:  # Filter common false positives
                tags.add(acronym.lower())

        # Sort and return top 10 tags
        return sorted(list(tags))[:10]

    def _classify_cxd(self, content: str) -> str:
        """Enhanced CXD classification using multi-layered analysis"""
        result = self.cxd_classifier.classify(content)
        return result.function  # Returns CONTROL/CONTEXT/DATA/unknown
    
    def _calculate_importance(self, content: str, memory_type: str) -> float:
        """Calculate importance score for a memory"""
        base_score = 0.5
        
        # Type-based adjustments
        if memory_type == 'milestone':
            base_score += 0.3
        elif memory_type == 'reflection':
            base_score += 0.2
        
        # Length-based adjustments
        if len(content) > 500:
            base_score += 0.1
        elif len(content) < 50:
            base_score -= 0.1
        
        # Keyword-based adjustments
        important_keywords = ['important', 'critical', 'essential', 'key', 'vital']
        if any(kw in content.lower() for kw in important_keywords):
            base_score += 0.1
        
        return min(1.0, max(0.0, base_score))
    
    def _calculate_relevance(self, query: str, content: str) -> float:
        """Calculate relevance score between query and content"""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words:
            return 0.0
        
        # Calculate word overlap
        overlap = len(query_words & content_words)
        relevance = overlap / len(query_words)
        
        return min(1.0, relevance)
    
    def _generate_analysis(self, input_text: str, memories: List[Memory]) -> str:
        """Generate analysis based on input and memories"""
        if not memories:
            return "No relevant memories found for context."
        
        # Analyze memory types
        types = [m.metadata.get('type', 'unknown') for m in memories]
        type_summary = f"Found {len(memories)} relevant memories"
        
        # Analyze CXD distribution
        cxd_types = [m.metadata.get('cxd', 'unknown') for m in memories]
        cxd_counts = {}
        for cxd in cxd_types:
            cxd_counts[cxd] = cxd_counts.get(cxd, 0) + 1
        
        # Generate analysis
        analysis = f"{type_summary}. "
        if cxd_counts:
            dominant_cxd = max(cxd_counts, key=cxd_counts.get)
            analysis += f"Context primarily relates to {dominant_cxd} aspects. "
        
        # Add temporal analysis
        if memories:
            latest = max(m.created_at for m in memories)
            oldest = min(m.created_at for m in memories)
            time_span = (latest - oldest).days
            if time_span > 0:
                analysis += f"Memories span {time_span} days of history."
        
        return analysis

    def compare_memories(self, memory_ids: List[str], comparison_type: str = "detailed") -> Dict:
        """
        Compare multiple memories to identify similarities and differences

        Args:
            memory_ids: List of memory IDs to compare (2-5 memories recommended)
            comparison_type: Type of comparison
                - "detailed": Full comparison with all dimensions
                - "quick": High-level overview
                - "diff": Focus on differences only

        Returns:
            Dict with comparison results including:
                - memories: List of compared memories
                - similarities: What the memories have in common
                - differences: How the memories differ
                - quality_comparison: Quality scores comparison
                - relationship_analysis: How memories relate
                - recommendations: Suggested actions (merge, keep separate, etc.)
        """
        if len(memory_ids) < 2:
            return {
                'error': 'Need at least 2 memories to compare',
                'provided': len(memory_ids)
            }

        if len(memory_ids) > 5:
            return {
                'error': 'Maximum 5 memories can be compared at once',
                'provided': len(memory_ids),
                'suggestion': 'Try comparing in smaller groups'
            }

        # Fetch memories
        all_memories = self.adapter.get_all()
        memories_map = {m.id: m for m in all_memories}

        memories = []
        missing = []
        for mem_id in memory_ids:
            if mem_id in memories_map:
                memories.append(memories_map[mem_id])
            else:
                missing.append(mem_id)

        if missing:
            return {
                'error': f'Memories not found: {missing}',
                'found': [m.id for m in memories]
            }

        # Build comparison result
        result = {
            'comparison_type': comparison_type,
            'memory_count': len(memories),
            'memories': []
        }

        # Add memory summaries
        for mem in memories:
            mem_dict = mem.to_dict()
            result['memories'].append({
                'id': mem.id,
                'created': mem.created_at.isoformat(),
                'type': mem_dict.get('type', 'unknown'),
                'cxd': mem_dict.get('cxd', 'unknown'),
                'cxd_confidence': mem_dict.get('cxd_confidence', 0.0),
                'quality': mem_dict.get('quality', {}),
                'tags': mem_dict.get('tags', []),
                'importance': mem.importance,
                'content_preview': mem.content[:200] + '...' if len(mem.content) > 200 else mem.content
            })

        # Analyze similarities
        similarities = self._find_similarities(memories)
        result['similarities'] = similarities

        # Analyze differences
        differences = self._find_differences(memories)
        result['differences'] = differences

        # Quality comparison
        if comparison_type in ['detailed', 'quick']:
            quality_comp = self._compare_quality(memories)
            result['quality_comparison'] = quality_comp

        # Relationship analysis
        if comparison_type == 'detailed':
            rel_analysis = self._analyze_relationships(memories)
            result['relationship_analysis'] = rel_analysis

        # Generate recommendations
        recommendations = self._generate_recommendations(memories, similarities, differences)
        result['recommendations'] = recommendations

        return result

    def _find_similarities(self, memories: List[Memory]) -> Dict:
        """Find what memories have in common"""
        similarities = {}

        # Common CXD types
        cxd_types = [m.metadata.get('cxd', 'unknown') for m in memories]
        if len(set(cxd_types)) == 1:
            similarities['cxd'] = f"All memories are {cxd_types[0]} type"

        # Common memory types
        mem_types = [m.metadata.get('type', 'unknown') for m in memories]
        if len(set(mem_types)) == 1:
            similarities['memory_type'] = f"All are {mem_types[0]} memories"

        # Common tags
        all_tags = [set(m.metadata.get('tags', [])) for m in memories]
        if all_tags:
            common_tags = set.intersection(*all_tags)
            if common_tags:
                similarities['common_tags'] = list(common_tags)

        # Content similarity (Jaccard)
        content_similarity = []
        for i in range(len(memories)):
            for j in range(i + 1, len(memories)):
                sim = self._calculate_jaccard_similarity(
                    memories[i].content,
                    memories[j].content
                )
                content_similarity.append({
                    'pair': f"{memories[i].id[:8]} vs {memories[j].id[:8]}",
                    'similarity': round(sim, 3)
                })

        similarities['content_overlap'] = content_similarity

        # Time proximity
        dates = [m.created_at for m in memories]
        time_span = (max(dates) - min(dates)).days
        similarities['temporal_span'] = f"{time_span} days between earliest and latest"

        return similarities

    def _find_differences(self, memories: List[Memory]) -> Dict:
        """Find how memories differ"""
        differences = {}

        # CXD differences
        cxd_types = [m.metadata.get('cxd', 'unknown') for m in memories]
        if len(set(cxd_types)) > 1:
            differences['cxd_distribution'] = dict(zip(
                [m.id[:8] for m in memories],
                cxd_types
            ))

        # Quality differences
        qualities = []
        for m in memories:
            quality = m.metadata.get('quality', {})
            overall = quality.get('overall_score', 0.0)
            qualities.append({
                'id': m.id[:8],
                'score': overall
            })

        if qualities:
            qualities.sort(key=lambda x: x['score'], reverse=True)
            differences['quality_ranking'] = qualities

        # Importance differences
        importances = [(m.id[:8], m.importance) for m in memories]
        importances.sort(key=lambda x: x[1], reverse=True)
        differences['importance_ranking'] = [
            {'id': id, 'importance': imp} for id, imp in importances
        ]

        # Tag differences
        all_tags = {m.id[:8]: set(m.metadata.get('tags', [])) for m in memories}
        unique_tags = {}
        for mem_id, tags in all_tags.items():
            other_tags = set()
            for other_id, other_tag_set in all_tags.items():
                if other_id != mem_id:
                    other_tags.update(other_tag_set)
            unique = tags - other_tags
            if unique:
                unique_tags[mem_id] = list(unique)

        if unique_tags:
            differences['unique_tags'] = unique_tags

        return differences

    def _compare_quality(self, memories: List[Memory]) -> Dict:
        """Compare quality metrics across memories"""
        comparison = {
            'dimension_comparison': {},
            'overall_ranking': []
        }

        # Compare each quality dimension
        dimensions = ['clarity', 'information_density', 'contextual_relevance',
                     'uniqueness', 'importance_potential', 'factual_accuracy']

        for dim in dimensions:
            scores = []
            for m in memories:
                quality = m.metadata.get('quality', {})
                dim_scores = quality.get('dimensions', {})
                score = dim_scores.get(dim, 0.0)
                scores.append({
                    'id': m.id[:8],
                    'score': round(score, 3)
                })

            scores.sort(key=lambda x: x['score'], reverse=True)
            comparison['dimension_comparison'][dim] = scores

        # Overall ranking
        overall = []
        for m in memories:
            quality = m.metadata.get('quality', {})
            overall_score = quality.get('overall_score', 0.0)
            auto_approved = quality.get('auto_approved', False)
            overall.append({
                'id': m.id[:8],
                'score': round(overall_score, 3),
                'auto_approved': auto_approved
            })

        overall.sort(key=lambda x: x['score'], reverse=True)
        comparison['overall_ranking'] = overall

        return comparison

    def _analyze_relationships(self, memories: List[Memory]) -> Dict:
        """Analyze how memories relate to each other"""
        analysis = {
            'direct_relationships': [],
            'relationship_strength': []
        }

        # Check if memories reference each other
        mem_ids = {m.id for m in memories}

        for m in memories:
            relationships = m.metadata.get('relationships', {})
            similar = relationships.get('similar_memories', [])

            # Find which compared memories are in relationships
            related_in_set = [s for s in similar if s in mem_ids]

            if related_in_set:
                analysis['direct_relationships'].append({
                    'memory': m.id[:8],
                    'relates_to': [r[:8] for r in related_in_set],
                    'strength': relationships.get('relationship_strength', 0.0)
                })

        # Calculate pairwise relationships
        for i in range(len(memories)):
            for j in range(i + 1, len(memories)):
                similarity = self._calculate_jaccard_similarity(
                    memories[i].content,
                    memories[j].content
                )

                if similarity > 0.3:  # Significant relationship
                    analysis['relationship_strength'].append({
                        'pair': f"{memories[i].id[:8]} ↔ {memories[j].id[:8]}",
                        'similarity': round(similarity, 3),
                        'strength': 'strong' if similarity > 0.6 else 'moderate'
                    })

        return analysis

    def _generate_recommendations(self, memories: List[Memory],
                                 similarities: Dict, differences: Dict) -> List[str]:
        """Generate actionable recommendations based on comparison"""
        recommendations = []

        # Check for duplicates
        content_overlaps = similarities.get('content_overlap', [])
        high_similarity = [s for s in content_overlaps if s['similarity'] > 0.8]

        if high_similarity:
            recommendations.append(
                f"⚠️ High content similarity detected ({len(high_similarity)} pairs >80%). "
                "Consider merging duplicate memories."
            )

        # Check quality disparities
        quality_ranking = differences.get('quality_ranking', [])
        if len(quality_ranking) >= 2:
            highest = quality_ranking[0]['score']
            lowest = quality_ranking[-1]['score']

            if highest - lowest > 0.3:
                low_id = quality_ranking[-1]['id']
                recommendations.append(
                    f"💡 Quality disparity detected. Memory {low_id} could be enhanced "
                    "or deprecated in favor of higher quality versions."
                )

        # Check for relationship gaps
        if len(memories) > 2:
            relationships = [m.metadata.get('relationships', {}).get('similar_memories', [])
                           for m in memories]
            mem_ids = {m.id for m in memories}

            unconnected = []
            for i, m in enumerate(memories):
                related = set(relationships[i])
                has_connection = bool(related & mem_ids)
                if not has_connection:
                    unconnected.append(m.id[:8])

            if unconnected:
                recommendations.append(
                    f"🔗 Memories {', '.join(unconnected)} have no relationships "
                    "with other compared memories. Consider linking related content."
                )

        # Common tags suggest topic clustering
        common_tags = similarities.get('common_tags', [])
        if len(common_tags) >= 3:
            recommendations.append(
                f"📚 Strong thematic connection via tags: {', '.join(common_tags)}. "
                "Consider creating a tale or collection around this topic."
            )

        # If no specific recommendations
        if not recommendations:
            recommendations.append(
                "✅ Memories are appropriately distinct with reasonable similarity. "
                "No immediate actions needed."
            )

        return recommendations


# Command-line interface for testing
def main():
    """CLI for testing MCP tools"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MemMimic MCP Tools')
    parser.add_argument('command', choices=['remember', 'recall', 'think', 'status', 'migrate'],
                       help='Command to execute')
    parser.add_argument('--content', help='Content for remember/think')
    parser.add_argument('--query', help='Query for recall')
    parser.add_argument('--type', default='interaction', help='Memory type')
    parser.add_argument('--cxd', default='ALL', help='CXD filter')
    parser.add_argument('--limit', type=int, default=10, help='Result limit')
    
    args = parser.parse_args()
    
    # Initialize MCP
    mcp = MemMimicMCP()
    
    # Execute command
    if args.command == 'remember':
        if not args.content:
            print("Error: --content required for remember")
            return
        result = mcp.remember(args.content, args.type)
        print(json.dumps(result, indent=2))
    
    elif args.command == 'recall':
        if not args.query:
            print("Error: --query required for recall")
            return
        results = mcp.recall_cxd(args.query, args.cxd, args.limit)
        print(json.dumps(results, indent=2, default=str))
    
    elif args.command == 'think':
        if not args.content:
            print("Error: --content required for think")
            return
        result = mcp.think_with_memory(args.content)
        print(json.dumps(result, indent=2, default=str))
    
    elif args.command == 'status':
        result = mcp.status()
        print(json.dumps(result, indent=2))
    
    elif args.command == 'migrate':
        result = mcp.migrate_to_markdown()
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()