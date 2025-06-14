# --- Archivo: src/cxd_classifier/providers/vector_store.py ---
import hashlib
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

import numpy as np

from ..core.interfaces import VectorStore # Asegúrate que esta importación sea correcta

# Optional FAISS import
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None # faiss será None si la importación falla

logger = logging.getLogger(__name__)


class FAISSVectorStore(VectorStore):
    """
    High-performance vector store using FAISS for similarity search.
    Provides sub-millisecond vector search with support for various
    distance metrics and index types.
    """

    def __init__(self,
                 dimension: int, # Este es el parámetro de entrada
                 metric: str = "cosine",
                 index_type: str = "flat",
                 normalize_vectors: bool = True):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is not available. Install with: pip install faiss-cpu or faiss-gpu")

        self._dimension_val: int = dimension  # ASIGNACIÓN AL ATRIBUTO INTERNO
        self.metric: str = metric.lower()
        self.index_type: str = index_type.lower()
        self.normalize_vectors: bool = normalize_vectors

        self.index: 'faiss.Index' = self._create_faiss_index()
        self.metadata_store: List[Dict[str, Any]] = []
        self.stats: Dict[str, Any] = {
            "vectors_added": 0,
            "searches_performed": 0,
            "total_search_time_ms": 0.0,
            "last_search_time_ms": 0.0
        }
        logger.info(f"Initialized FAISS vector store: {self.dimension}D, {self.metric}, {self.index_type}")

    @property
    def dimension(self) -> int:
        """Get vector dimension."""
        return self._dimension_val

    def _create_faiss_index(self) -> 'faiss.Index':
        """Create appropriate FAISS index based on configuration."""
        dim_to_use = self._dimension_val
        index: 'faiss.Index'

        if self.metric == "cosine":
            if self.index_type == "flat":
                index = faiss.IndexFlatIP(dim_to_use)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatIP(dim_to_use)
                nlist = min(100, max(1, dim_to_use // 10 if dim_to_use >= 40 else 4))
                index = faiss.IndexIVFFlat(quantizer, dim_to_use, nlist, faiss.METRIC_INNER_PRODUCT)
            elif self.index_type == "hnsw":
                M = 32
                index = faiss.IndexHNSWFlat(dim_to_use, M, faiss.METRIC_INNER_PRODUCT)
                index.hnsw.efConstruction = 40
                index.hnsw.efSearch = 32
            else:
                raise ValueError(f"Unsupported index type for cosine: {self.index_type}")

        elif self.metric == "euclidean":
            if self.index_type == "flat":
                index = faiss.IndexFlatL2(dim_to_use)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatL2(dim_to_use)
                nlist = min(100, max(1, dim_to_use // 10 if dim_to_use >= 40 else 4))
                index = faiss.IndexIVFFlat(quantizer, dim_to_use, nlist, faiss.METRIC_L2)
            elif self.index_type == "hnsw":
                M = 32
                index = faiss.IndexHNSWFlat(dim_to_use, M, faiss.METRIC_L2)
                index.hnsw.efConstruction = 40
                index.hnsw.efSearch = 32
            else:
                raise ValueError(f"Unsupported index type for euclidean: {self.index_type}")

        elif self.metric == "dot_product":
            if self.index_type == "flat":
                index = faiss.IndexFlatIP(dim_to_use)
            else:
                raise ValueError(f"Dot product currently only supports flat index type for simplicity. "
                                 f"Consider IndexIVFFlat or HNSWFlat with METRIC_INNER_PRODUCT if needed.")
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        return index

    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        if vectors.ndim != 2:
            raise ValueError("Vectors must be a 2D array")
        if vectors.shape[1] != self.dimension: # Usa la propiedad (getter)
            raise ValueError(f"Vector dimension {vectors.shape[1]} doesn't match index dimension {self.dimension}")
        if len(metadata) != vectors.shape[0]:
            raise ValueError("Number of metadata entries must match number of vectors")

        vectors_to_add = vectors.astype(np.float32)

        if self.normalize_vectors and self.metric == "cosine":
            faiss.normalize_L2(vectors_to_add)

        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            n_train_min = getattr(self.index, 'nlist', 1) if hasattr(self.index, 'nlist') else 1
            n_train_min = max(n_train_min, 1)
            if vectors_to_add.shape[0] >= n_train_min:
                self.index.train(vectors_to_add)
            else:
                logger.warning(f"Index type {self.index_type} may require training, but not enough vectors ({vectors_to_add.shape[0]}) provided. "
                               f"Min required: {n_train_min}. This might lead to errors if index is not already trained.")

        self.index.add(vectors_to_add)
        self.metadata_store.extend(metadata)
        self.stats["vectors_added"] += vectors_to_add.shape[0]
        logger.debug(f"Added {vectors_to_add.shape[0]} vectors to FAISS index. Total size: {self.index.ntotal}")

    def search(self, query_vector: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        if self.index.ntotal == 0:
            logger.warning("Search called on an empty FAISS index.")
            return np.array([]), np.array([])
        if query_vector.ndim != 1:
            raise ValueError("Query vector must be 1D")
        if len(query_vector) != self.dimension: # Usa la propiedad (getter)
            raise ValueError(f"Query vector dimension {len(query_vector)} doesn't match index dimension {self.dimension}")

        query = query_vector.astype(np.float32).reshape(1, -1)
        if self.normalize_vectors and self.metric == "cosine":
            faiss.normalize_L2(query)

        effective_k = min(k, self.index.ntotal)
        if effective_k == 0:
            return np.array([]), np.array([])

        similarities, indices = self.index.search(query, effective_k)
        similarities = similarities[0]
        indices = indices[0]

        valid_mask = indices >= 0
        similarities = similarities[valid_mask]
        indices = indices[valid_mask]

        search_time = (time.time() - start_time) * 1000
        self.stats["searches_performed"] += 1
        self.stats["total_search_time_ms"] += search_time
        self.stats["last_search_time_ms"] = search_time
        return similarities, indices

    def get_metadata(self, index: int) -> Dict[str, Any]:
        if 0 <= index < len(self.metadata_store):
            return self.metadata_store[index]
        raise IndexError(f"Index {index} out of range for metadata store (size: {len(self.metadata_store)})")

    def save(self, path: Union[str, Path]) -> bool:
        try:
            save_path = Path(path)
            save_path.mkdir(parents=True, exist_ok=True)
            index_path = save_path / "faiss_index.bin"
            faiss.write_index(self.index, str(index_path))
            metadata_path = save_path / "metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata_store, f)
            config_data = {
                "dimension": self.dimension, # Usa la propiedad (getter)
                "metric": self.metric,
                "index_type": self.index_type,
                "normalize_vectors": self.normalize_vectors,
                "stats": self.stats
            }
            config_path_file = save_path / "config.pkl"
            with open(config_path_file, 'wb') as f:
                pickle.dump(config_data, f)
            logger.info(f"Saved FAISS vector store to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save FAISS vector store: {e}", exc_info=True)
            return False

    def load(self, path: Union[str, Path]) -> bool:
        try:
            load_path = Path(path)
            config_path_file = load_path / "config.pkl"
            if not config_path_file.exists():
                logger.error(f"Config file not found at {config_path_file}")
                return False
            with open(config_path_file, 'rb') as f:
                config_data = pickle.load(f)

            # Verificamos que la dimensión del __init__ coincida con la guardada.
            # Si se quisiera que el archivo dictara la dimensión, habría que cambiar self._dimension_val aquí.
            if config_data["dimension"] != self.dimension: # Usa la propiedad (getter)
                raise ValueError(f"Dimension mismatch: expected {self.dimension} (from init), got {config_data['dimension']} from saved config.")

            index_path = load_path / "faiss_index.bin"
            if not index_path.exists():
                logger.error(f"FAISS index file not found at {index_path}")
                return False
            self.index = faiss.read_index(str(index_path))

            metadata_path = load_path / "metadata.pkl"
            if not metadata_path.exists():
                logger.warning(f"Metadata file not found at {metadata_path}. Initializing empty metadata.")
                self.metadata_store = []
            else:
                with open(metadata_path, 'rb') as f:
                    self.metadata_store = pickle.load(f)

            self.metric = config_data["metric"]
            self.index_type = config_data["index_type"]
            self.normalize_vectors = config_data["normalize_vectors"]
            self.stats = config_data.get("stats", self.stats)
            logger.info(f"Loaded FAISS vector store from {load_path}. Index size: {self.index.ntotal}")
            return True
        except Exception as e:
            logger.error(f"Failed to load FAISS vector store: {e}", exc_info=True)
            return False

    @property
    def size(self) -> int:
        return self.index.ntotal if self.index else 0

    def clear(self) -> None:
        if self.index: # Solo llamar a reset si el índice existe
            self.index.reset()
        self.metadata_store.clear()
        self.stats["vectors_added"] = 0
        # Podrías querer recrear el índice con self._create_faiss_index() si reset no es suficiente
        # o si los parámetros de creación pudieran cambiar. Por ahora, reset es lo estándar.
        logger.info("Cleared FAISS vector store.")

    def get_stats(self) -> Dict[str, Any]:
        stats = self.stats.copy()
        stats.update({
            "size": self.size,
            "dimension": self.dimension, # Usa la propiedad (getter)
            "metric": self.metric,
            "index_type": self.index_type,
            "faiss_available": True
        })
        if self.stats.get("searches_performed", 0) > 0:
            stats["avg_search_time_ms"] = (
                self.stats["total_search_time_ms"] / self.stats["searches_performed"]
            )
        return stats


class NumpyVectorStore(VectorStore):
    """
    Fallback vector store using NumPy for similarity search.
    Provides O(n) search when FAISS is not available. Suitable for
    small to medium datasets (< 10k vectors).
    """
    def __init__(self, dimension: int, metric: str = "cosine"):
        self._dimension_val: int = dimension  # ASIGNACIÓN AL ATRIBUTO INTERNO
        self.metric: str = metric.lower()

        if self.metric not in ["cosine", "euclidean", "dot_product"]:
            raise ValueError(f"Unsupported metric: {self.metric}")

        self.vectors: Optional[np.ndarray] = None
        self.metadata_store: List[Dict[str, Any]] = []
        self.stats: Dict[str, Any] = {
            "vectors_added": 0,
            "searches_performed": 0,
            "total_search_time_ms": 0.0,
            "last_search_time_ms": 0.0
        }
        logger.info(f"Initialized NumPy vector store: {self.dimension}D, {self.metric}") # Usa la propiedad (getter)

    @property
    def dimension(self) -> int:
        """Get vector dimension."""
        return self._dimension_val

    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        if vectors.ndim != 2:
            raise ValueError("Vectors must be a 2D array")
        if vectors.shape[1] != self.dimension: # Usa la propiedad (getter)
            raise ValueError(f"Vector dimension {vectors.shape[1]} doesn't match store dimension {self.dimension}")
        if len(metadata) != vectors.shape[0]:
            raise ValueError("Number of metadata entries must match number of vectors")

        vectors_to_add = vectors.astype(np.float32)
        if self.vectors is None:
            self.vectors = vectors_to_add
        else:
            self.vectors = np.vstack([self.vectors, vectors_to_add])

        self.metadata_store.extend(metadata)
        self.stats["vectors_added"] += vectors_to_add.shape[0]
        logger.debug(f"Added {vectors_to_add.shape[0]} vectors to NumPy store. Total size: {self.size}")

    def search(self, query_vector: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()
        if self.vectors is None or self.size == 0:
            logger.warning("Search called on an empty NumPy store.")
            return np.array([]), np.array([])
        if query_vector.ndim != 1:
            raise ValueError("Query vector must be 1D")
        if len(query_vector) != self.dimension: # Usa la propiedad (getter)
            raise ValueError(f"Query vector dimension {len(query_vector)} doesn't match store dimension {self.dimension}")

        query = query_vector.astype(np.float32)
        similarities: np.ndarray

        if self.metric == "cosine":
            similarities = self._cosine_similarity(query, self.vectors)
        elif self.metric == "euclidean":
            distances = np.linalg.norm(self.vectors - query, axis=1)
            similarities = 1.0 / (1.0 + distances + 1e-8) # Añadido epsilon para evitar división por cero si distancia es -1
        elif self.metric == "dot_product":
            similarities = np.dot(self.vectors, query)
        else:
            # Esta rama es teóricamente inalcanzable debido a la validación en __init__
            raise ValueError(f"Unsupported metric for search: {self.metric}")

        effective_k = min(k, len(similarities))
        if effective_k == 0:
            return np.array([]), np.array([])

        top_indices = np.argsort(similarities)[::-1][:effective_k] # Orden descendente para similitud
        top_similarities = similarities[top_indices]

        search_time = (time.time() - start_time) * 1000
        self.stats["searches_performed"] += 1
        self.stats["total_search_time_ms"] += search_time
        self.stats["last_search_time_ms"] = search_time
        return top_similarities, top_indices

    def _cosine_similarity(self, query: np.ndarray, data_vectors: np.ndarray) -> np.ndarray:
        query_norm = np.linalg.norm(query)
        if query_norm < 1e-8: # query es un vector nulo
            return np.zeros(data_vectors.shape[0]) # Similitud cero con todos

        data_norms = np.linalg.norm(data_vectors, axis=1)
        
        # Crear una máscara para las normas de data_vectors que son casi cero
        zero_norm_mask = data_norms < 1e-8
        # Para las normas casi cero, establecerlas a un valor pequeño para evitar la división por cero,
        # pero la similitud con estos vectores será esencialmente cero de todos modos si el producto punto no es también cero.
        data_norms[zero_norm_mask] = 1e-8 

        dot_products = np.dot(data_vectors, query)
        similarities = dot_products / (query_norm * data_norms)
        
        # Si un vector de datos era nulo, su similitud debería ser 0 (a menos que query también fuera nulo, caso ya manejado)
        similarities[zero_norm_mask] = 0.0
        
        return similarities


    def get_metadata(self, index: int) -> Dict[str, Any]:
        if 0 <= index < len(self.metadata_store):
            return self.metadata_store[index]
        raise IndexError(f"Index {index} out of range for metadata store (size: {len(self.metadata_store)})")

    def save(self, path: Union[str, Path]) -> bool:
        try:
            save_path = Path(path)
            save_path.mkdir(parents=True, exist_ok=True)
            if self.vectors is not None:
                vectors_path = save_path / "vectors.npy"
                np.save(vectors_path, self.vectors)
            metadata_path = save_path / "metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata_store, f)
            config_data = {
                "dimension": self.dimension, # Usa la propiedad (getter)
                "metric": self.metric,
                "stats": self.stats
            }
            config_path_file = save_path / "config.pkl"
            with open(config_path_file, 'wb') as f:
                pickle.dump(config_data, f)
            logger.info(f"Saved NumPy vector store to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save NumPy vector store: {e}", exc_info=True)
            return False

    def load(self, path: Union[str, Path]) -> bool:
        try:
            load_path = Path(path)
            config_path_file = load_path / "config.pkl"
            if not config_path_file.exists():
                logger.error(f"Config file not found at {config_path_file}")
                return False
            with open(config_path_file, 'rb') as f:
                config_data = pickle.load(f)

            if config_data["dimension"] != self.dimension: # Usa la propiedad (getter)
                raise ValueError(f"Dimension mismatch: expected {self.dimension}, got {config_data['dimension']} from saved config.")

            vectors_path = load_path / "vectors.npy"
            if vectors_path.exists():
                self.vectors = np.load(vectors_path)
            else:
                self.vectors = None # Importante si el archivo de vectores no existe

            metadata_path = load_path / "metadata.pkl"
            if not metadata_path.exists():
                logger.warning(f"Metadata file not found at {metadata_path}. Initializing empty metadata.")
                self.metadata_store = []
            else:
                with open(metadata_path, 'rb') as f:
                    self.metadata_store = pickle.load(f)
            
            self.metric = config_data["metric"]
            self.stats = config_data.get("stats", self.stats)
            logger.info(f"Loaded NumPy vector store from {load_path}. Size: {self.size}")
            return True
        except Exception as e:
            logger.error(f"Failed to load NumPy vector store: {e}", exc_info=True)
            return False

    @property
    def size(self) -> int:
        return 0 if self.vectors is None else self.vectors.shape[0]

    def clear(self) -> None:
        self.vectors = None
        self.metadata_store.clear()
        self.stats["vectors_added"] = 0
        logger.info("Cleared NumPy vector store.")

    def get_stats(self) -> Dict[str, Any]:
        stats = self.stats.copy()
        stats.update({
            "size": self.size,
            "dimension": self.dimension, # Usa la propiedad (getter)
            "metric": self.metric,
            "faiss_available": False
        })
        if self.stats.get("searches_performed", 0) > 0:
            stats["avg_search_time_ms"] = (
                self.stats["total_search_time_ms"] / self.stats["searches_performed"]
            )
        return stats

# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_vector_store(dimension: int,
                       metric: str = "cosine",
                       prefer_faiss: bool = True,
                       index_type: str = "flat",
                       **kwargs) -> VectorStore:
    """
    Create the best available vector store for the environment.
    """
    # normalize_vectors es un kwarg común para FAISSVectorStore cuando metric es 'cosine'
    normalize_vectors = kwargs.get('normalize_vectors', True)

    if prefer_faiss and FAISS_AVAILABLE:
        try:
            logger.info(f"Attempting to create FAISSVectorStore (dim={dimension}, metric={metric}, type={index_type}, normalize={normalize_vectors})")
            return FAISSVectorStore(dimension, metric, index_type, normalize_vectors=normalize_vectors)
        except Exception as e:
            logger.warning(f"Failed to create FAISS store, falling back to NumPy: {e}", exc_info=True)

    logger.info(f"Using NumPyVectorStore as FAISS is not preferred, not available, or failed to initialize (dim={dimension}, metric={metric}).")
    return NumpyVectorStore(dimension, metric)


def create_faiss_store(dimension: int, **kwargs) -> FAISSVectorStore:
    """
    Create FAISS vector store (requires FAISS to be available).
    """
    return FAISSVectorStore(dimension, **kwargs)


def create_numpy_store(dimension: int, **kwargs) -> NumpyVectorStore:
    """
    Create NumPy vector store.
    """
    return NumpyVectorStore(dimension, **kwargs)


# Export vector stores
__all__ = [
    "FAISSVectorStore",
    "NumpyVectorStore",
    "create_vector_store",
    "create_faiss_store",
    "create_numpy_store",
    "FAISS_AVAILABLE",
]