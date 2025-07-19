# --- Archivo: src/cxd_classifier/providers/embedding_models.py ---
import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

# Importar la interfaz base
from ..core.interfaces import EmbeddingModel

# Optional imports for SentenceTransformer and PyTorch
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None  # Necesario para que el type hint 'SentenceTransformer' no falle si no está instalado

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)


class SentenceTransformerModel(EmbeddingModel):
    """
    Production embedding model using SentenceTransformers.
    Provides high-quality semantic embeddings using pre-trained transformer models.
    """

    def __init__(
        self,
        model_name_param: str = "all-MiniLM-L6-v2",  # Renombrado para evitar conflicto con la propiedad
        device_param: str = "cpu",  # Renombrado para evitar conflicto si device se vuelve propiedad
        cache_folder: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers not available. Install with: "
                "pip install sentence-transformers"
            )

        self._actual_model_name: str = model_name_param
        self._device_val: str = self._validate_device(device_param)
        self._cache_folder_val: Optional[str] = cache_folder
        self._trust_remote_code_val: bool = trust_remote_code

        logger.info(
            f"Attempting to load SentenceTransformer model: {self._actual_model_name} on device: {self._device_val}"
        )
        start_time = time.time()
        try:
            # self.st_model es la instancia del modelo de la biblioteca sentence_transformers
            self.st_model: SentenceTransformer = SentenceTransformer(
                self._actual_model_name,
                device=self._device_val,
                cache_folder=self._cache_folder_val,
                trust_remote_code=self._trust_remote_code_val,
            )
            load_time = time.time() - start_time
            logger.info(
                f"Successfully loaded SentenceTransformer model '{self._actual_model_name}' in {load_time:.2f}s from cache or download."
            )
        except Exception as e:
            logger.error(
                f"Failed to load SentenceTransformer model '{self._actual_model_name}': {e}",
                exc_info=True,
            )
            raise RuntimeError(f"Failed to load model {self._actual_model_name}: {e}")

        self._dimension_val: int = self.st_model.get_sentence_embedding_dimension()
        self._max_seq_length_val: Optional[int] = getattr(
            self.st_model, "max_seq_length", None
        )

        self.stats: Dict[str, Any] = {
            "total_encodings": 0,
            "total_texts_encoded": 0,
            "total_encoding_time_ms": 0.0,
            "model_load_time_s": load_time,
            "cache_hits": 0,  # In-memory cache hits for this instance
            "cache_misses": 0,
        }
        self._encoding_cache: Dict[str, np.ndarray] = {}
        self._cache_enabled: bool = False
        self._max_cache_size: int = 1000

    def _validate_device(self, device_choice: str) -> str:
        """Validate and adjust device setting."""
        selected_device = device_choice.lower()
        if selected_device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():  # type: ignore
                return "cuda"
            elif TORCH_AVAILABLE and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore
                return "mps"
            return "cpu"
        elif selected_device == "cuda":
            if not (TORCH_AVAILABLE and torch.cuda.is_available()):  # type: ignore
                logger.warning(
                    f"CUDA specified but not available. Falling back to CPU."
                )
                return "cpu"
            return "cuda"
        elif selected_device == "mps":
            if not (TORCH_AVAILABLE and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):  # type: ignore
                logger.warning(f"MPS specified but not available. Falling back to CPU.")
                return "cpu"
            return "mps"
        elif selected_device == "cpu":
            return "cpu"
        else:
            logger.warning(f"Unknown device '{selected_device}', falling back to CPU")
            return "cpu"

    @property
    def model_name(self) -> str:
        """Get the specific SentenceTransformer model name used."""
        return self._actual_model_name

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension_val

    @property
    def max_sequence_length(self) -> Optional[int]:
        """Get maximum sequence length supported by model."""
        return self._max_seq_length_val

    def encode(self, text: str) -> np.ndarray:
        cache_key = ""  # Definir fuera del if para que esté en scope
        if self._cache_enabled:
            cache_key = self._get_cache_key(text)
            if cache_key in self._encoding_cache:
                self.stats["cache_hits"] += 1
                return self._encoding_cache[cache_key].copy()
            self.stats["cache_misses"] += 1

        start_time = time.time()
        try:
            embedding = self.st_model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,  # Normalizar para similitud coseno efectiva
                show_progress_bar=False,
            )
            # Asegurar float32 como es común para embeddings
            embedding_float32 = embedding.astype(np.float32)

            encoding_time = (time.time() - start_time) * 1000
            self.stats["total_encodings"] += 1
            self.stats["total_texts_encoded"] += 1
            self.stats["total_encoding_time_ms"] += encoding_time

            if self._cache_enabled:
                self._cache_embedding(cache_key, embedding_float32)
            return embedding_float32
        except Exception as e:
            logger.error(
                f"Failed to encode text with {self.model_name}: {e}", exc_info=True
            )
            return np.zeros(
                self.dimension, dtype=np.float32
            )  # Usa la propiedad dimension

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty(
                (0, self.dimension), dtype=np.float32
            )  # Usa la propiedad dimension

        start_time = time.time()
        try:
            embeddings = self.st_model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=32,  # Un valor razonable, puede ser configurable
            )
            embeddings_float32 = embeddings.astype(np.float32)

            encoding_time = (time.time() - start_time) * 1000
            self.stats[
                "total_encodings"
            ] += 1  # Una llamada a encode_batch es un "encoding"
            self.stats["total_texts_encoded"] += len(texts)
            self.stats["total_encoding_time_ms"] += encoding_time
            return embeddings_float32
        except Exception as e:
            logger.error(
                f"Failed to encode batch with {self.model_name}: {e}", exc_info=True
            )
            return np.zeros(
                (len(texts), self.dimension), dtype=np.float32
            )  # Usa la propiedad dimension

    def _get_cache_key(self, text: str) -> str:
        # La propiedad self.model_name ya devuelve _actual_model_name
        content = f"{self.model_name}:{text}"
        return hashlib.md5(content.encode("utf-8")).hexdigest()[:16]

    def _cache_embedding(self, key: str, embedding: np.ndarray) -> None:
        if len(self._encoding_cache) >= self._max_cache_size:
            oldest_key = next(iter(self._encoding_cache))
            del self._encoding_cache[oldest_key]
        self._encoding_cache[key] = embedding.copy()  # Guardar una copia

    def enable_cache(self, max_size: int = 1000) -> None:
        self._cache_enabled = True
        self._max_cache_size = max_size
        logger.info(
            f"Enabled in-memory encoding cache (size: {max_size}) for {self.model_name}."
        )

    def disable_cache(self) -> None:
        self._cache_enabled = False
        self._encoding_cache.clear()
        logger.info(
            f"Disabled and cleared in-memory encoding cache for {self.model_name}."
        )

    def clear_cache(self) -> None:
        self._encoding_cache.clear()
        logger.info(f"Cleared in-memory encoding cache for {self.model_name}.")

    def get_stats(self) -> Dict[str, Any]:
        stats_copy = self.stats.copy()
        stats_copy.update(
            {
                "model_name": self.model_name,  # Propiedad
                "device": self._device_val,  # Atributo interno donde se guarda el device validado
                "dimension": self.dimension,  # Propiedad
                "max_sequence_length": self.max_sequence_length,  # Propiedad
                "cache_enabled": self._cache_enabled,
                "cache_size": len(self._encoding_cache),
                "cache_max_size": self._max_cache_size,
                "cache_hit_rate": (
                    (
                        self.stats["cache_hits"]
                        / max(
                            self.stats["cache_hits"] + self.stats["cache_misses"], 1
                        )  # Evitar división por cero
                    )
                    if self._cache_enabled
                    and (self.stats["cache_hits"] + self.stats["cache_misses"]) > 0
                    else 0.0
                ),
            }
        )
        if self.stats["total_encodings"] > 0:
            stats_copy[
                "avg_encoding_time_ms_per_call"
            ] = (  # Tiempo promedio por llamada a encode/encode_batch
                self.stats["total_encoding_time_ms"] / self.stats["total_encodings"]
            )
            if self.stats["total_texts_encoded"] > 0:
                stats_copy[
                    "avg_encoding_time_ms_per_text"
                ] = (  # Tiempo promedio por texto individual
                    self.stats["total_encoding_time_ms"]
                    / self.stats["total_texts_encoded"]
                )
            stats_copy["avg_texts_per_encoding_call"] = (
                self.stats["total_texts_encoded"] / self.stats["total_encodings"]
            )
        return stats_copy


class MockEmbeddingModel(EmbeddingModel):
    """
    Mock embedding model for development and testing.
    Generates deterministic embeddings based on text hashing.
    """

    def __init__(
        self,
        dimension_param: int = 384,  # Renombrado para claridad
        seed: int = 42,
        normalize: bool = True,
    ):
        self._dimension_val: int = dimension_param  # Atributo interno
        self.seed: int = seed
        self.normalize: bool = normalize
        self.rng = np.random.RandomState(seed)
        self.stats: Dict[str, Any] = {
            "total_encodings": 0,
            "total_texts_encoded": 0,
            "total_encoding_time_ms": 0.0,
        }
        logger.info(
            f"Initialized MockEmbeddingModel: {self.dimension}D, seed={self.seed}"
        )  # Usa la propiedad

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension_val

    @property
    def model_name(self) -> str:
        """Get the model name/identifier."""
        # Sobrescribe el default de la interfaz para ser más específico
        return f"MockEmbeddingModel-{self.dimension}D-seed{self.seed}"

    @property
    def max_sequence_length(self) -> Optional[int]:
        """Mock model doesn't have a fixed max sequence length."""
        return None

    def encode(self, text: str) -> np.ndarray:
        start_time = time.time()
        embedding = self._text_to_embedding(text)
        encoding_time = (time.time() - start_time) * 1000
        self.stats["total_encodings"] += 1
        self.stats["total_texts_encoded"] += 1
        self.stats["total_encoding_time_ms"] += encoding_time
        return embedding

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)  # Usa la propiedad

        start_time = time.time()
        embeddings = np.array(
            [self._text_to_embedding(text) for text in texts], dtype=np.float32
        )
        encoding_time = (time.time() - start_time) * 1000
        self.stats["total_encodings"] += 1
        self.stats["total_texts_encoded"] += len(texts)
        self.stats["total_encoding_time_ms"] += encoding_time
        return embeddings

    def _text_to_embedding(self, text: str) -> np.ndarray:
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        hash_seed = int(text_hash[:8], 16) % (
            2**31
        )  # Usar solo una parte del hash para el seed
        local_rng = np.random.RandomState(hash_seed)
        # Generar embedding usando la dimensión correcta (a través de la propiedad)
        embedding = local_rng.normal(0, 1, self.dimension).astype(np.float32)

        # Lógica simplificada para añadir variaciones basadas en el texto
        if self.dimension > 0:  # Evitar error si la dimensión es 0
            embedding[0] = (
                embedding[0] + len(text.split()) / 100.0
            ) % 1.0  # Modulo 1 para mantener en rango similar
        if self.dimension > 1:
            embedding[1] = (embedding[1] + len(text) / 1000.0) % 1.0

        if self.normalize:
            norm = np.linalg.norm(embedding)
            if norm > 1e-8:  # Evitar división por cero
                embedding = embedding / norm
            else:  # Si la norma es cero (o casi), devolver un vector nulo o manejar como se prefiera
                embedding = np.zeros(self.dimension, dtype=np.float32)
        return embedding

    def get_stats(self) -> Dict[str, Any]:
        stats_copy = self.stats.copy()
        stats_copy.update(
            {
                "model_name": self.model_name,  # Propiedad
                "dimension": self.dimension,  # Propiedad
                "seed": self.seed,
                "normalize": self.normalize,
                "is_mock": True,
            }
        )
        if self.stats["total_encodings"] > 0:
            stats_copy["avg_encoding_time_ms_per_call"] = (
                self.stats["total_encoding_time_ms"] / self.stats["total_encodings"]
            )
            if self.stats["total_texts_encoded"] > 0:
                stats_copy["avg_encoding_time_ms_per_text"] = (
                    self.stats["total_encoding_time_ms"]
                    / self.stats["total_texts_encoded"]
                )
            stats_copy["avg_texts_per_encoding_call"] = (
                self.stats["total_texts_encoded"] / self.stats["total_encodings"]
            )
        return stats_copy


class CachedEmbeddingModel(EmbeddingModel):
    """
    Wrapper that adds persistent caching to any embedding model.
    Saves embeddings to disk to avoid recomputing expensive embeddings.
    """

    def __init__(
        self,
        base_model: EmbeddingModel,
        cache_dir_param: Optional[Union[str, Path]] = None,  # Renombrado
        max_cache_size: int = 10000,
    ):
        self.base_model: EmbeddingModel = base_model
        self._max_cache_size_val: int = max_cache_size  # Atributo interno
        self._memory_cache: Dict[str, np.ndarray] = {}
        self._disk_cache_dir: Optional[Path] = None

        if cache_dir_param:
            self._disk_cache_dir = Path(cache_dir_param).expanduser()
            try:
                self._disk_cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info(
                    f"Disk cache for embeddings initialized at: {self._disk_cache_dir}"
                )
            except OSError as e:
                logger.error(
                    f"Could not create disk cache directory {self._disk_cache_dir}: {e}. Disk caching will be disabled."
                )
                self._disk_cache_dir = None

        self.cache_stats: Dict[str, Any] = {  # Renombrado para claridad
            "memory_hits": 0,
            "memory_misses": 0,  # Misses de memoria que intentan disco o codificación
            "disk_hits": 0,
            "disk_misses": 0,  # Misses de disco que fuerzan codificación
            "encodes_due_to_miss": 0,  # Veces que se tuvo que codificar porque no estaba en ninguna caché
        }
        logger.info(
            f"Initialized CachedEmbeddingModel wrapping '{self.base_model.model_name}'. Memory cache size: {self._max_cache_size_val}, Disk cache: {self._disk_cache_dir or 'Disabled'}"
        )

    @property
    def dimension(self) -> int:
        """Get embedding dimension from the base model."""
        return self.base_model.dimension

    @property
    def model_name(self) -> str:
        """Get the model name, indicating it's cached."""
        return f"Cached({self.base_model.model_name})"

    @property
    def max_sequence_length(self) -> Optional[int]:
        """Get max sequence length from the base model."""
        return self.base_model.max_sequence_length

    def encode(self, text: str) -> np.ndarray:
        cache_key = self._get_cache_key(text)

        # 1. Check memory cache
        if cache_key in self._memory_cache:
            self.cache_stats["memory_hits"] += 1
            return self._memory_cache[cache_key].copy()
        self.cache_stats["memory_misses"] += 1

        # 2. Check disk cache (if enabled)
        if self._disk_cache_dir:
            disk_embedding = self._load_from_disk(cache_key)
            if disk_embedding is not None:
                self.cache_stats["disk_hits"] += 1
                self._add_to_memory_cache(
                    cache_key, disk_embedding
                )  # Cargar en memoria también
                return disk_embedding.copy()
            self.cache_stats["disk_misses"] += 1  # Miss de disco

        # 3. Generate new embedding (cache miss on both memory and disk)
        self.cache_stats["encodes_due_to_miss"] += 1
        embedding = self.base_model.encode(text)
        self._add_to_memory_cache(cache_key, embedding)
        if self._disk_cache_dir:
            self._save_to_disk(cache_key, embedding)
        return embedding  # Ya es una copia si encode() del base model devuelve copia, o el original.

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        results = np.empty((len(texts), self.dimension), dtype=np.float32)
        texts_to_encode_indices: List[int] = []
        texts_to_encode_values: List[str] = []

        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            # 1. Check memory cache
            if cache_key in self._memory_cache:
                results[i] = self._memory_cache[cache_key]
                self.cache_stats["memory_hits"] += 1
                continue
            self.cache_stats["memory_misses"] += 1

            # 2. Check disk cache
            if self._disk_cache_dir:
                disk_embedding = self._load_from_disk(cache_key)
                if disk_embedding is not None:
                    results[i] = disk_embedding
                    self._add_to_memory_cache(cache_key, disk_embedding)
                    self.cache_stats["disk_hits"] += 1
                    continue
                self.cache_stats["disk_misses"] += 1

            # 3. Mark for encoding
            texts_to_encode_indices.append(i)
            texts_to_encode_values.append(text)

        if texts_to_encode_values:
            self.cache_stats["encodes_due_to_miss"] += len(texts_to_encode_values)
            new_embeddings = self.base_model.encode_batch(texts_to_encode_values)
            for i, original_idx in enumerate(texts_to_encode_indices):
                embedding = new_embeddings[i]
                results[original_idx] = embedding
                text_for_key = texts_to_encode_values[i]  # o texts[original_idx]
                cache_key = self._get_cache_key(text_for_key)
                self._add_to_memory_cache(cache_key, embedding)
                if self._disk_cache_dir:
                    self._save_to_disk(cache_key, embedding)
        return results

    def _get_cache_key(self, text: str) -> str:
        # Usa la propiedad model_name del modelo base para la clave
        content = f"{self.base_model.model_name}:{text}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def _add_to_memory_cache(self, key: str, embedding: np.ndarray) -> None:
        if len(self._memory_cache) >= self._max_cache_size_val:
            # Simple FIFO, podrías implementar LRU si es necesario
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]
        self._memory_cache[key] = embedding.copy()  # Siempre guardar copia

    def _save_to_disk(self, key: str, embedding: np.ndarray) -> None:
        if not self._disk_cache_dir:
            return
        try:
            cache_file = self._disk_cache_dir / f"{key}.npy"
            np.save(cache_file, embedding)
        except Exception as e:
            logger.warning(
                f"Failed to save embedding '{key}' to disk cache at {self._disk_cache_dir}: {e}"
            )

    def _load_from_disk(self, key: str) -> Optional[np.ndarray]:
        if not self._disk_cache_dir:
            return None
        try:
            cache_file = self._disk_cache_dir / f"{key}.npy"
            if cache_file.exists() and cache_file.is_file():
                return np.load(cache_file)
        except Exception as e:
            logger.warning(
                f"Failed to load embedding '{key}' from disk cache at {self._disk_cache_dir}: {e}"
            )
        return None

    def clear_cache(self) -> None:
        self._memory_cache.clear()
        if self._disk_cache_dir:
            try:
                for cache_file in self._disk_cache_dir.glob("*.npy"):
                    cache_file.unlink(missing_ok=True)  # missing_ok en Python 3.8+
            except Exception as e:
                logger.warning(
                    f"Failed to clear disk cache at {self._disk_cache_dir}: {e}"
                )
        logger.info(f"Cleared cache for {self.model_name}")
        # Resetear contadores de caché también
        for key in self.cache_stats:
            self.cache_stats[key] = 0

    def get_stats(self) -> Dict[str, Any]:
        # Obtener stats del modelo base para tener una visión completa
        # Es importante no llamar a get_stats del base_model recursivamente si este también es un CachedEmbeddingModel.
        # Asumimos que base_model.get_stats() es seguro.
        base_model_stats_summary = {}
        try:
            # Para evitar llamadas recursivas infinitas si se anidan CachedEmbeddingModels (aunque no es lo usual)
            if not isinstance(self.base_model, CachedEmbeddingModel):
                base_model_stats = self.base_model.get_stats()
                base_model_stats_summary = {
                    "base_model_total_encodings": base_model_stats.get(
                        "total_encodings"
                    ),
                    "base_model_avg_encoding_time_ms_per_call": base_model_stats.get(
                        "avg_encoding_time_ms_per_call"
                    ),
                    "base_model_avg_encoding_time_ms_per_text": base_model_stats.get(
                        "avg_encoding_time_ms_per_text"
                    ),
                }
        except Exception as e:
            logger.warning(
                f"Could not retrieve stats from base model {self.base_model.model_name}: {e}"
            )

        # Calcular tasas de acierto
        total_memory_access = (
            self.cache_stats["memory_hits"] + self.cache_stats["memory_misses"]
        )
        memory_hit_rate = (
            (self.cache_stats["memory_hits"] / total_memory_access)
            if total_memory_access > 0
            else 0.0
        )

        total_disk_access = (
            self.cache_stats["disk_hits"] + self.cache_stats["disk_misses"]
        )
        disk_hit_rate = (
            (self.cache_stats["disk_hits"] / total_disk_access)
            if total_disk_access > 0
            else 0.0
        )

        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "base_model_name": self.base_model.model_name,
            "cache_info": {
                "memory_hits": self.cache_stats["memory_hits"],
                "memory_misses": self.cache_stats["memory_misses"],
                "memory_hit_rate": memory_hit_rate,
                "memory_cache_current_size": len(self._memory_cache),
                "memory_cache_max_size": self._max_cache_size_val,
                "disk_cache_enabled": self._disk_cache_dir is not None,
                "disk_cache_path": (
                    str(self._disk_cache_dir) if self._disk_cache_dir else None
                ),
                "disk_hits": self.cache_stats["disk_hits"],
                "disk_misses": self.cache_stats["disk_misses"],
                "disk_hit_rate": disk_hit_rate,
                "total_encodes_due_to_cache_miss": self.cache_stats[
                    "encodes_due_to_miss"
                ],
            },
            "base_model_performance_summary": base_model_stats_summary,
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_embedding_model(model_type: str = "auto", **kwargs) -> EmbeddingModel:
    """
    Create the best available embedding model for the environment.
    Args:
        model_type: Type of model ("auto", "sentence_transformer", "mock")
        **kwargs: Additional arguments for the model.
                  For "sentence_transformer", "model_name" se mapea a "model_name_param",
                                            "device" se mapea a "device_param".
                  Para "mock", "dimension" se mapea a "dimension_param".
    """
    model_name_kwarg = kwargs.pop("model_name", None)
    device_kwarg = kwargs.pop("device", None)  # Extraer 'device' si existe

    if model_type == "auto":
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.info("Auto-selected SentenceTransformer model (production).")
            st_kwargs = {}
            if model_name_kwarg:
                st_kwargs["model_name_param"] = model_name_kwarg
            if device_kwarg:
                st_kwargs["device_param"] = device_kwarg  # <--- AÑADIR MAPEO
            st_kwargs.update(
                kwargs
            )  # Añadir otros kwargs restantes (cache_folder, trust_remote_code)
            return SentenceTransformerModel(**st_kwargs)
        else:
            # ... (lógica de MockEmbeddingModel como estaba) ...
            logger.info(
                "SentenceTransformers not available. Auto-selected MockEmbeddingModel."
            )
            mock_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k in ["dimension_param", "seed", "normalize"]
            }
            if "dimension" in kwargs and "dimension_param" not in mock_kwargs:
                mock_kwargs["dimension_param"] = kwargs["dimension"]
            return MockEmbeddingModel(**mock_kwargs)

    elif model_type == "sentence_transformer":
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "SentenceTransformers library is not installed. "
                "Please install it with: pip install sentence-transformers"
            )
        st_kwargs = {}
        if model_name_kwarg:
            st_kwargs["model_name_param"] = model_name_kwarg
        if device_kwarg:
            st_kwargs["device_param"] = device_kwarg  # <--- AÑADIR MAPEO
        st_kwargs.update(kwargs)
        return SentenceTransformerModel(**st_kwargs)

    elif model_type == "mock":
        # ... (lógica de MockEmbeddingModel como estaba) ...
        mock_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ["dimension_param", "seed", "normalize"]
        }
        if "dimension" in kwargs and "dimension_param" not in mock_kwargs:
            mock_kwargs["dimension_param"] = kwargs["dimension"]
        return MockEmbeddingModel(**mock_kwargs)
    else:
        raise ValueError(
            f"Unknown embedding model type: '{model_type}'. "
            "Valid types are 'auto', 'sentence_transformer', 'mock'."
        )


def create_cached_model(base_model: EmbeddingModel, **kwargs) -> CachedEmbeddingModel:
    """
    Wrap an embedding model with caching.
    Args:
        base_model: Base embedding model to wrap.
        **kwargs: Arguments for CachedEmbeddingModel (e.g., cache_dir_param, max_cache_size).
                  'cache_dir' se mapea a 'cache_dir_param'.
    """
    if "cache_dir" in kwargs and "cache_dir_param" not in kwargs:
        kwargs["cache_dir_param"] = kwargs.pop("cache_dir")
    return CachedEmbeddingModel(base_model, **kwargs)


# Export embedding models
__all__ = [
    "SentenceTransformerModel",
    "MockEmbeddingModel",
    "CachedEmbeddingModel",
    "create_embedding_model",
    "create_cached_model",
    "SENTENCE_TRANSFORMERS_AVAILABLE",
    "TORCH_AVAILABLE",
]
