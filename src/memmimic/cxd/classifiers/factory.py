"""
Factory

MemMimic module for factory functionality.
"""


# --- Archivo: src/cxd_classifier/classifiers/factory.py ---

from typing import Literal, Optional

from ..core.config import (
    CXDConfig,
    create_default_config,
    create_development_config,
    create_production_config,
)
from ..core.interfaces import (
    CXDClassifier,  # Interfaz base para todos los clasificadores
)

# Importación de todos los tipos de clasificadores que la factoría podría construir
from .lexical import LexicalCXDClassifier
from .meta import MetaCXDClassifier
from .optimized_meta import (  # create_fast_classifier es una función de fábrica útil
    OptimizedMetaCXDClassifier,
    create_fast_classifier,
)
from .optimized_semantic import OptimizedSemanticCXDClassifier
from .semantic import SemanticCXDClassifier

# Definimos los tipos de clasificadores que la factoría puede crear para mejor type hinting y validación
ClassifierType = Literal[
    "lexical",
    "semantic",
    "optimized_semantic",
    "meta",
    "optimized_meta",  # Podría ser el default o "production"
    "fast",
    "development",  # Para OptimizedMetaCXDClassifier con config de desarrollo
    "production",  # Para OptimizedMetaCXDClassifier con config de producción
]


class CXDClassifierFactory:
    """
    Factoría para crear instancias de diferentes tipos de clasificadores CXD.
    Permite una creación centralizada y configurable de los componentes de clasificación.
    """

    @staticmethod
    def create(
        classifier_type: ClassifierType = "optimized_meta",
        config: Optional[CXDConfig] = None,
        **kwargs,
    ) -> CXDClassifier:
        """
        Crea y devuelve una instancia de un clasificador CXD.

        Args:
            classifier_type: El tipo de clasificador a crear.
                Opciones válidas: "lexical", "semantic", "optimized_semantic",
                                 "meta", "optimized_meta", "fast",
                                 "development", "production".
                Por defecto es "optimized_meta".
            config: Una instancia opcional de CXDConfig. Si es None,
                    se usará una configuración apropiada para el tipo de clasificador
                    o una configuración por defecto.
            **kwargs: Argumentos adicionales que se pasarán al constructor
                      del clasificador (especialmente útil para OptimizedMetaCXDClassifier
                      y OptimizedSemanticCXDClassifier).

        Returns:
            Una instancia de una clase que implementa la interfaz CXDClassifier.

        Raises:
            ValueError: Si se especifica un classifier_type desconocido.
        """

        if classifier_type == "lexical":
            effective_config = config or create_default_config()
            return LexicalCXDClassifier(config=effective_config)

        elif classifier_type == "semantic":
            effective_config = config or create_default_config()
            # SemanticCXDClassifier puede tomar embedding_model, example_provider, vector_store
            # Si se pasan en kwargs, se usarán. Si no, usará sus defaults.
            return SemanticCXDClassifier(config=effective_config, **kwargs)

        elif classifier_type == "optimized_semantic":
            effective_config = config or create_default_config()
            # OptimizedSemanticCXDClassifier también puede tomar componentes y flags de caché en kwargs
            return OptimizedSemanticCXDClassifier(config=effective_config, **kwargs)

        elif classifier_type == "meta":
            effective_config = config or create_default_config()
            # MetaCXDClassifier puede tomar lexical_classifier y semantic_classifier en kwargs
            # o creará los suyos por defecto.
            return MetaCXDClassifier(config=effective_config, **kwargs)

        elif classifier_type == "optimized_meta":
            effective_config = config or create_default_config()
            # OptimizedMetaCXDClassifier usa kwargs para enable_cache_persistence, rebuild_cache, etc.
            return OptimizedMetaCXDClassifier(config=effective_config, **kwargs)

        elif classifier_type == "fast":
            # create_fast_classifier es una función de fábrica en optimized_meta.py
            # que ya maneja la configuración para la velocidad.
            effective_config = config  # Permite al usuario pasar una config si quiere, si no create_fast_classifier usa la suya
            return create_fast_classifier(config=effective_config, **kwargs)

        elif classifier_type == "production":
            effective_config = config or create_production_config()
            # Asumimos que "production" implica OptimizedMetaCXDClassifier con configuración de producción
            return OptimizedMetaCXDClassifier.create_production_classifier(
                config=effective_config
            )

        elif classifier_type == "development":
            effective_config = config or create_development_config()
            # Asumimos que "development" implica OptimizedMetaCXDClassifier con configuración de desarrollo
            return OptimizedMetaCXDClassifier.create_development_classifier(
                config=effective_config
            )

        else:
            raise ValueError(
                f"Tipo de clasificador desconocido: '{classifier_type}'. "
                f"Opciones válidas: 'lexical', 'semantic', 'optimized_semantic', "
                f"'meta', 'optimized_meta', 'fast', 'production', 'development'."
            )


# Podrías también tener una función de conveniencia a nivel de módulo si lo prefieres,
# similar a lo que ya tienes en src/cxd_classifier/__init__.py, pero definida aquí
# para mantener la lógica de creación junta.
# def create_classifier(...) -> CXDClassifier:
# return CXDClassifierFactory.create(...)

