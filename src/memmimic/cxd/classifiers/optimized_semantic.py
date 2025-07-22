"""
Optimized Semantic CXD Classifier implementation.

This module implements an optimized version of the semantic classifier with
advanced caching, FAISS indexing, and enhanced performance features.
"""

import logging
import time
from typing import Any, Dict, Optional

from ..core.config import CXDConfig
from ..core.interfaces import CanonicalExampleProvider, EmbeddingModel, VectorStore
from ..providers.embedding_models import create_cached_model, create_embedding_model
from ..providers.examples import YamlExampleProvider
from ..providers.vector_store import create_vector_store
from .semantic import SemanticCXDClassifier

logger = logging.getLogger(__name__)


class OptimizedSemanticCXDClassifier(SemanticCXDClassifier):
    """
    Optimized semantic classifier with advanced caching and indexing.

    Extends the base semantic classifier with:
    - Persistent FAISS indexing with cache management
    - Advanced embedding models with caching
    - Checksum-based cache invalidation
    - Enhanced performance monitoring
    - Configurable optimization strategies
    """

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        example_provider: Optional[CanonicalExampleProvider] = None,
        vector_store: Optional[VectorStore] = None,
        config: Optional[CXDConfig] = None,
        enable_cache_persistence: bool = True,
        rebuild_cache: bool = False,
    ):
        """
        Initialize optimized semantic classifier.

        Args:
            embedding_model: Embedding model (auto-optimized if None)
            example_provider: Example provider (YAML-based if None)
            vector_store: Vector store (FAISS-optimized if None)
            config: Configuration object
            enable_cache_persistence: Enable persistent caching
            rebuild_cache: Force cache rebuild on initialization
        """
        self.config = config or CXDConfig()
        self.enable_cache_persistence = enable_cache_persistence
        self.rebuild_cache = rebuild_cache

        # Initialize optimized components
        optimized_embedding_model = (
            embedding_model or self._create_optimized_embedding_model()
        )
        optimized_example_provider = (
            example_provider or self._create_optimized_example_provider()
        )
        optimized_vector_store = vector_store or self._create_optimized_vector_store(
            optimized_embedding_model.dimension
        )

        # Initialize parent with optimized components
        super().__init__(
            embedding_model=optimized_embedding_model,
            example_provider=optimized_example_provider,
            vector_store=optimized_vector_store,
            config=self.config,
        )

        # Optimization-specific state
        self._cache_dir = self.config.get_cache_path("semantic_classifier")
        self._cache_checksum: Optional[str] = None
        self._cache_valid = False

        # Enhanced statistics
        self.optimization_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "index_loads": 0,
            "index_saves": 0,
            "checksum_validations": 0,
            "optimization_time_saved_ms": 0.0,
        }

        # Initialize optimized index
        self._initialize_optimized_index()

        logger.info(
            f"Initialized OptimizedSemanticCXDClassifier with cache persistence: {enable_cache_persistence}"
        )

    def _create_optimized_embedding_model(self) -> EmbeddingModel:
        """Create optimized embedding model with caching."""
        try:
            # Create base model
            base_model = create_embedding_model(
                model_type="auto",
                model_name=self.config.models.embedding.name,
                device=self.config.get_effective_device(),
            )

            # Wrap with caching if enabled
            if self.enable_cache_persistence:
                cache_dir = self.config.get_cache_path("embeddings")
                cached_model = create_cached_model(
                    base_model=base_model, cache_dir=str(cache_dir), max_cache_size=1000
                )
                return cached_model
            else:
                return base_model

        except Exception as e:
            logger.warning(f"Failed to create optimized embedding model: {e}")
            # Fallback to mock model
            return create_embedding_model(
                model_type="mock",
                dimension=self.config.models.mock.dimension,
                seed=self.config.models.mock.seed,
            )

    def _create_optimized_example_provider(self) -> CanonicalExampleProvider:
        """Create optimized example provider."""
        try:
            examples_path = self.config.paths.canonical_examples
            if examples_path.exists():
                return YamlExampleProvider(
                    yaml_path=examples_path,
                    cache_parsed=True,  # Enable in-memory caching
                )
            else:
                logger.warning(f"Canonical examples file not found: {examples_path}")
                # Create empty in-memory provider
                from ..providers.examples import InMemoryExampleProvider

                return InMemoryExampleProvider()

        except Exception as e:
            logger.warning(f"Failed to create optimized example provider: {e}")
            from ..providers.examples import InMemoryExampleProvider

            return InMemoryExampleProvider()

    def _create_optimized_vector_store(self, dimension: int) -> VectorStore:
        """Create optimized vector store with FAISS if available."""
        return create_vector_store(
            dimension=dimension,
            metric=self.config.algorithms.search.metric.value,
            prefer_faiss=self.config.features.faiss_indexing,
            index_type="flat",  # Use flat index for stability
            normalize_vectors=True,  # Normalize for cosine similarity
        )

    def _initialize_optimized_index(self) -> None:
        """Initialize index with optimization strategies."""
        try:
            # Check if we should rebuild
            if self.rebuild_cache:
                logger.info("Forcing cache rebuild as requested")
                self._clear_cache()
                self._build_index()
                return

            # Try to load from cache
            if self.enable_cache_persistence and self._load_index_from_cache():
                logger.info("Loaded semantic index from cache")
                self.optimization_stats["cache_hits"] += 1
                return

            # Build new index
            logger.info("Building new semantic index")
            self._build_index()
            self.optimization_stats["cache_misses"] += 1

            # Save to cache if enabled
            if self.enable_cache_persistence:
                self._save_index_to_cache()

        except Exception as e:
            logger.error(f"Failed to initialize optimized index: {e}")
            # Fallback to base implementation
            super()._build_index()

    def _load_index_from_cache(self) -> bool:
        """
        Load index from cache with checksum validation.

        Returns:
            bool: True if successfully loaded from cache
        """
        try:
            if not self._cache_dir.exists():
                return False

            # Validate cache checksum
            current_checksum = self.example_provider.get_checksum()
            cache_checksum_file = self._cache_dir / "checksum.txt"

            if cache_checksum_file.exists():
                with open(cache_checksum_file, "r") as f:
                    cached_checksum = f.read().strip()

                self.optimization_stats["checksum_validations"] += 1

                if cached_checksum != current_checksum:
                    logger.info("Cache invalidated due to checksum mismatch")
                    return False
            else:
                logger.info("No cache checksum found")
                return False

            # Load vector store
            if not self.vector_store.load(self._cache_dir):
                logger.warning("Failed to load vector store from cache")
                return False

            # Load metadata
            metadata_file = self._cache_dir / "example_metadata.json"
            if metadata_file.exists():
                import json

                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata_list = json.load(f)

                # Reconstruct metadata with proper CXDFunction objects
                from ..core.types import CXDFunction

                self._example_metadata = []
                for meta in metadata_list:
                    meta["function"] = CXDFunction.from_string(meta["function"])
                    self._example_metadata.append(meta)

            self._index_built = True
            self._cache_checksum = current_checksum
            self._cache_valid = True
            self.optimization_stats["index_loads"] += 1

            return True

        except Exception as e:
            logger.warning(f"Failed to load index from cache: {e}")
            return False

    def _save_index_to_cache(self) -> bool:
        """
        Save index to cache with checksum.

        Returns:
            bool: True if successfully saved
        """
        try:
            # Create cache directory
            self._cache_dir.mkdir(parents=True, exist_ok=True)

            # Save vector store
            if not self.vector_store.save(self._cache_dir):
                logger.warning("Failed to save vector store to cache")
                return False

            # Save metadata
            metadata_file = self._cache_dir / "example_metadata.json"
            import json

            # Serialize metadata with function names as strings
            serializable_metadata = []
            for meta in self._example_metadata:
                serialized_meta = meta.copy()
                serialized_meta["function"] = meta["function"].value
                serializable_metadata.append(serialized_meta)

            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(serializable_metadata, f, indent=2, ensure_ascii=False)

            # Save checksum
            current_checksum = self.example_provider.get_checksum()
            checksum_file = self._cache_dir / "checksum.txt"
            with open(checksum_file, "w") as f:
                f.write(current_checksum)

            # Save cache info
            cache_info = {
                "created_at": time.time(),
                "vector_store_type": type(self.vector_store).__name__,
                "embedding_model": self.embedding_model.model_name,
                "examples_count": len(self._example_metadata),
                "dimension": self.embedding_model.dimension,
            }

            info_file = self._cache_dir / "cache_info.json"
            with open(info_file, "w") as f:
                json.dump(cache_info, f, indent=2)

            self._cache_checksum = current_checksum
            self._cache_valid = True
            self.optimization_stats["index_saves"] += 1

            logger.info(f"Saved semantic index cache to {self._cache_dir}")
            return True

        except Exception as e:
            logger.error(f"Failed to save index to cache: {e}")
            return False

    def _clear_cache(self) -> None:
        """Clear cached index files."""
        try:
            if self._cache_dir.exists():
                import shutil

                shutil.rmtree(self._cache_dir)
                logger.info(f"Cleared cache directory: {self._cache_dir}")
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")

    def _build_index(self) -> None:
        """Build index with optimization timing."""
        build_start = time.time()

        # Call parent build method
        super()._build_index()

        build_time = time.time() - build_start

        # Save to cache if enabled and successful
        if self.enable_cache_persistence and self._index_built:
            cache_start = time.time()
            if self._save_index_to_cache():
                cache_time = time.time() - cache_start
                logger.info(
                    f"Index built in {build_time:.2f}s, cached in {cache_time:.2f}s"
                )
            else:
                logger.warning(f"Index built in {build_time:.2f}s, but caching failed")

    def classify(self, text: str) -> "CXDSequence":
        """Classify with optimization tracking."""
        start_time = time.time()

        # Check if cache is still valid
        if self.enable_cache_persistence and self._cache_valid:
            current_checksum = self.example_provider.get_checksum()
            if current_checksum != self._cache_checksum:
                logger.info("Examples changed, rebuilding index")
                self.rebuild_index()

        # Call parent classify method
        result = super().classify(text)

        # Track optimization time savings (rough estimate)
        processing_time = (time.time() - start_time) * 1000
        if self._cache_valid:
            # Estimate time saved by not rebuilding index
            estimated_rebuild_time = 5000  # 5 seconds average
            self.optimization_stats[
                "optimization_time_saved_ms"
            ] += estimated_rebuild_time

        return result

    def rebuild_index(self) -> None:
        """Rebuild index with cache management."""
        logger.info("Rebuilding optimized semantic index...")

        # Clear cache
        if self.enable_cache_persistence:
            self._clear_cache()

        # Reset state
        self._index_built = False
        self._cache_valid = False
        self._cache_checksum = None
        self.vector_store.clear()
        self._example_metadata.clear()

        # Rebuild
        self._build_index()

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization-specific statistics."""
        return self.optimization_stats.copy()

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics including optimizations."""
        # Get base stats
        base_stats = super().get_performance_stats()

        # Add optimization stats
        opt_stats = self.get_optimization_stats()
        base_stats["optimization"] = opt_stats

        # Calculate cache hit rate
        total_cache_operations = opt_stats["cache_hits"] + opt_stats["cache_misses"]
        if total_cache_operations > 0:
            base_stats["optimization"]["cache_hit_rate"] = (
                opt_stats["cache_hits"] / total_cache_operations
            )

        # Add cache configuration
        base_stats["optimization"][
            "cache_persistence_enabled"
        ] = self.enable_cache_persistence
        base_stats["optimization"]["cache_directory"] = str(self._cache_dir)
        base_stats["optimization"]["cache_valid"] = self._cache_valid

        return base_stats

    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information."""
        info = {
            "cache_enabled": self.enable_cache_persistence,
            "cache_directory": str(self._cache_dir),
            "cache_exists": (
                self._cache_dir.exists() if self.enable_cache_persistence else False
            ),
            "cache_valid": self._cache_valid,
            "current_checksum": self.example_provider.get_checksum(),
            "cached_checksum": self._cache_checksum,
        }

        # Add cache file info if exists
        if self.enable_cache_persistence and self._cache_dir.exists():
            try:
                cache_info_file = self._cache_dir / "cache_info.json"
                if cache_info_file.exists():
                    import json

                    with open(cache_info_file, "r") as f:
                        cache_metadata = json.load(f)
                    info["cache_metadata"] = cache_metadata

                # Add file sizes
                cache_files = list(self._cache_dir.glob("*"))
                info["cache_files"] = {
                    str(f.name): f.stat().st_size for f in cache_files if f.is_file()
                }
                info["total_cache_size_bytes"] = sum(info["cache_files"].values())

            except Exception as e:
                info["cache_info_error"] = str(e)

        return info

    def clear_all_caches(self) -> None:
        """Clear all caches (index and embedding)."""
        # Clear index cache
        self._clear_cache()

        # Clear embedding cache if model supports it
        if hasattr(self.embedding_model, "clear_cache"):
            self.embedding_model.clear_cache()

        # Reset state
        self._cache_valid = False
        self._cache_checksum = None

        logger.info("Cleared all caches")

    def validate_cache_integrity(self) -> Dict[str, Any]:
        """
        Validate cache integrity and return diagnostic information.

        Returns:
            Dict: Cache validation results
        """
        validation = {
            "cache_enabled": self.enable_cache_persistence,
            "cache_directory_exists": (
                self._cache_dir.exists() if self.enable_cache_persistence else False
            ),
            "required_files_present": False,
            "checksum_match": False,
            "vector_store_loadable": False,
            "metadata_loadable": False,
            "errors": [],
        }

        if not self.enable_cache_persistence:
            validation["status"] = "disabled"
            return validation

        try:
            # Check required files
            required_files = [
                "checksum.txt",
                "example_metadata.json",
                "cache_info.json",
            ]
            missing_files = []

            for filename in required_files:
                if not (self._cache_dir / filename).exists():
                    missing_files.append(filename)

            validation["required_files_present"] = len(missing_files) == 0
            if missing_files:
                validation["errors"].append(f"Missing files: {missing_files}")

            # Check checksum
            if (self._cache_dir / "checksum.txt").exists():
                with open(self._cache_dir / "checksum.txt", "r") as f:
                    cached_checksum = f.read().strip()
                current_checksum = self.example_provider.get_checksum()
                validation["checksum_match"] = cached_checksum == current_checksum
                validation["cached_checksum"] = cached_checksum
                validation["current_checksum"] = current_checksum

            # Test vector store loading
            test_vector_store = create_vector_store(
                dimension=self.embedding_model.dimension,
                metric=self.config.algorithms.search.metric.value,
                prefer_faiss=self.config.features.faiss_indexing,
            )
            validation["vector_store_loadable"] = test_vector_store.load(
                self._cache_dir
            )

            # Test metadata loading
            metadata_file = self._cache_dir / "example_metadata.json"
            if metadata_file.exists():
                import json

                with open(metadata_file, "r") as f:
                    json.load(f)  # Just test if it's valid JSON
                validation["metadata_loadable"] = True

            # Overall status
            if (
                validation["required_files_present"]
                and validation["checksum_match"]
                and validation["vector_store_loadable"]
                and validation["metadata_loadable"]
            ):
                validation["status"] = "valid"
            else:
                validation["status"] = "invalid"

        except Exception as e:
            validation["status"] = "error"
            validation["errors"].append(str(e))

        return validation


# Export optimized classifier
__all__ = [
    "OptimizedSemanticCXDClassifier",
]

