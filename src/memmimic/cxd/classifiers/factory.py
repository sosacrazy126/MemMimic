# --- Archivo: src/cxd_classifier/classifiers/factory.py ---

from typing import Optional, Union, Literal

from ..core.config import CXDConfig, create_default_config, create_production_config, create_development_config
from ..core.interfaces import CXDClassifier  # Base interface for all classifiers

# Import all classifier types that the factory could build
from .lexical import LexicalCXDClassifier
from .semantic import SemanticCXDClassifier
from .optimized_semantic import OptimizedSemanticCXDClassifier
from .meta import MetaCXDClassifier
from .optimized_meta import OptimizedMetaCXDClassifier, create_fast_classifier # create_fast_classifier is a useful factory function


# Define the classifier types the factory can create for better type hinting and validation
ClassifierType = Literal[
    "lexical",
    "semantic",
    "optimized_semantic",
    "meta",
    "optimized_meta", # Could be the default or "production"
    "fast",
    "development", # For OptimizedMetaCXDClassifier with development config
    "production"   # For OptimizedMetaCXDClassifier with production config
]


class CXDClassifierFactory:
    """
    Factory for creating instances of different CXD classifier types.
    Allows centralized and configurable creation of classification components.
    """

    @staticmethod
    def create(
        classifier_type: ClassifierType = "optimized_meta",
        config: Optional[CXDConfig] = None,
        **kwargs
    ) -> CXDClassifier:
        """
        Creates and returns an instance of a CXD classifier.

        Args:
            classifier_type: The type of classifier to create.
                Valid options: "lexical", "semantic", "optimized_semantic",
                              "meta", "optimized_meta", "fast",
                              "development", "production".
                Default is "optimized_meta".
            config: An optional CXDConfig instance. If None,
                    an appropriate configuration for the classifier type
                    or a default configuration will be used.
            **kwargs: Additional arguments to pass to the classifier
                      constructor (especially useful for OptimizedMetaCXDClassifier
                      and OptimizedSemanticCXDClassifier).

        Returns:
            An instance of a class that implements the CXDClassifier interface.

        Raises:
            ValueError: If an unknown classifier_type is specified.
        """

        if classifier_type == "lexical":
            effective_config = config or create_default_config()
            return LexicalCXDClassifier(config=effective_config)

        elif classifier_type == "semantic":
            effective_config = config or create_default_config()
            # SemanticCXDClassifier can take embedding_model, example_provider, vector_store
            # If passed in kwargs, they'll be used. Otherwise, it will use its defaults.
            return SemanticCXDClassifier(config=effective_config, **kwargs)

        elif classifier_type == "optimized_semantic":
            effective_config = config or create_default_config()
            # OptimizedSemanticCXDClassifier can also take components and cache flags in kwargs
            return OptimizedSemanticCXDClassifier(config=effective_config, **kwargs)

        elif classifier_type == "meta":
            effective_config = config or create_default_config()
            # MetaCXDClassifier can take lexical_classifier and semantic_classifier in kwargs
            # or will create its own by default.
            return MetaCXDClassifier(config=effective_config, **kwargs)

        elif classifier_type == "optimized_meta":
            effective_config = config or create_default_config()
            # OptimizedMetaCXDClassifier uses kwargs for enable_cache_persistence, rebuild_cache, etc.
            return OptimizedMetaCXDClassifier(config=effective_config, **kwargs)

        elif classifier_type == "fast":
            # create_fast_classifier is a factory function in optimized_meta.py
            # that already handles configuration for speed.
            effective_config = config # Allows user to pass a config if desired, otherwise create_fast_classifier uses its own
            return create_fast_classifier(config=effective_config, **kwargs)

        elif classifier_type == "production":
            effective_config = config or create_production_config()
            # We assume "production" implies OptimizedMetaCXDClassifier with production configuration
            return OptimizedMetaCXDClassifier.create_production_classifier(config=effective_config)


        elif classifier_type == "development":
            effective_config = config or create_development_config()
            # We assume "development" implies OptimizedMetaCXDClassifier with development configuration
            return OptimizedMetaCXDClassifier.create_development_classifier(config=effective_config)

        else:
            raise ValueError(f"Unknown classifier type: '{classifier_type}'. "
                             f"Valid options: 'lexical', 'semantic', 'optimized_semantic', "
                             f"'meta', 'optimized_meta', 'fast', 'production', 'development'.")

# You could also have a module-level convenience function if preferred,
# similar to what you have in src/cxd_classifier/__init__.py, but defined here
# to keep the creation logic together.
# def create_classifier(...) -> CXDClassifier:
# return CXDClassifierFactory.create(...)