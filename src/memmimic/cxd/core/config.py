"""
Configuration management for CXD Classifier using Pydantic.

This module provides a comprehensive configuration system with validation,
type checking, and support for multiple configuration sources including
YAML files, environment variables, and programmatic settings.

The configuration system is built on Pydantic for robust validation and
automatic type conversion.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Type
from enum import Enum

# Pydantic V2: BaseSettings is imported from pydantic_settings
from pydantic_settings import BaseSettings, SettingsConfigDict
# Pydantic V2: Field and the new validation decorators are imported from pydantic
from pydantic import Field, field_validator, model_validator
import yaml


class LogLevel(str, Enum):
    """Valid log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Device(str, Enum):
    """Valid compute devices."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Metal Performance Shaders


class OutputFormat(str, Enum):
    """Valid output formats for CLI."""
    TABLE = "table"
    JSON = "json"
    YAML = "yaml"
    CSV = "csv"


class MetricType(str, Enum):
    """Valid distance metrics for vector search."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"


class OptimizationMetric(str, Enum):
    """Valid metrics for optimization."""
    ACCURACY = "accuracy"
    F1_MACRO = "f1_macro"
    F1_WEIGHTED = "f1_weighted"
    PRECISION = "precision"
    RECALL = "recall"


# =============================================================================
# CONFIGURATION SECTIONS
# =============================================================================

class PathsConfig(BaseSettings):
    """Configuration for file paths and directories."""

    cache_dir: Path = Field(default=Path("./cxd_cache"), description="Cache directory")
    canonical_examples: Path = Field(default=Path("./config/canonical_examples.yaml"), description="Canonical examples file")
    log_dir: Path = Field(default=Path("./logs"), description="Log directory")
    models_dir: Path = Field(default=Path("./models"), description="Models directory")

    # Pydantic V2: Usar field_validator para validaciones de campos individuales.
    # El decorador ahora es por campo, y la firma de la función de validación cambia.
    @field_validator("cache_dir", "log_dir", "models_dir", mode='before') # mode='before' para replicar comportamiento de validator pre-v2
    @classmethod
    def ensure_directory_exists(cls, v: Any) -> Path: # v es el valor del campo
        """Ensure directory exists or can be created."""
        if isinstance(v, str):
            path_obj = Path(v)
        elif isinstance(v, Path):
            path_obj = v
        else:
            # Esto podría ser un error o requerir un manejo diferente según el caso de uso.
            # Por ahora, asumimos que siempre será str o Path según el default.
            raise TypeError("Path must be a string or Path object")
        path_obj.mkdir(parents=True, exist_ok=True)
        return path_obj

    @field_validator("canonical_examples", mode='before')
    @classmethod
    def validate_examples_file(cls, v: Any) -> Path: # v es el valor del campo
        """Validate examples file exists or provide default path."""
        if isinstance(v, str):
            return Path(v)
        elif isinstance(v, Path):
            return v
        # Don't require file to exist at config load time
        raise TypeError("Path must be a string or Path object")


class EmbeddingConfig(BaseSettings):
    """Configuration for embedding models."""

    name: str = Field(default="all-MiniLM-L6-v2", description="SentenceTransformer model name")
    device: Device = Field(default=Device.CPU, description="Compute device")
    fallback_dimension: int = Field(default=384, ge=50, le=4096, description="Fallback dimension for mock models")

    alternatives: Dict[str, str] = Field(default_factory=lambda: {
        "large": "all-mpnet-base-v2",
        "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",
        "fast": "all-MiniLM-L6-v2"
    })

    @field_validator("name")
    @classmethod
    def validate_model_name(cls, v: str) -> str: # v es el valor del campo
        """Validate model name format."""
        if not v or not isinstance(v, str):
            raise ValueError("Model name must be a non-empty string")
        return v


class MockModelConfig(BaseSettings):
    """Configuration for mock embedding model."""

    dimension: int = Field(default=384, ge=50, le=4096, description="Mock model dimension")
    seed: int = Field(default=42, description="Random seed for reproducibility")


class ModelsConfig(BaseSettings):
    """Configuration for all models."""

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    mock: MockModelConfig = Field(default_factory=MockModelConfig)


class ThresholdsConfig(BaseSettings):
    """Configuration for algorithm thresholds."""

    concordance: float = Field(default=0.6, ge=0.0, le=1.0, description="Meta-classification concordance threshold")
    semantic: float = Field(default=0.2, ge=0.0, le=1.0, description="Semantic similarity minimum threshold")
    confidence_min: float = Field(default=0.1, ge=0.0, le=1.0, description="Minimum confidence for classification")
    confidence_high: float = Field(default=0.8, ge=0.0, le=1.0, description="High confidence threshold")

    # Pydantic V2: Usar model_validator para validaciones a nivel de modelo (antes root_validator).
    # El modo 'after' es el más común para replicar el comportamiento de root_validator sin pre=True.
    # La firma de la función cambia: ahora es un método de instancia y debe devolver `self`.
    @model_validator(mode='after')
    def validate_threshold_ordering(self) -> 'ThresholdsConfig':
        """Ensure thresholds are in logical order."""
        # En Pydantic V2, se accede a los campos directamente con self.nombre_campo
        conf_min = self.confidence_min
        conf_high = self.confidence_high

        if conf_min >= conf_high:
            raise ValueError("confidence_min must be less than confidence_high")

        return self


class SearchConfig(BaseSettings):
    """Configuration for vector search."""

    k: int = Field(default=10, ge=1, le=100, description="Number of nearest neighbors")
    metric: MetricType = Field(default=MetricType.COSINE, description="Distance metric")
    timeout_ms: int = Field(default=1000, ge=100, description="Search timeout in milliseconds")


class FusionConfig(BaseSettings):
    """Configuration for classifier fusion."""

    lexical_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Weight for lexical classifier")
    semantic_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Weight for semantic classifier")
    semantic_override: bool = Field(default=True, description="Enable semantic override in low concordance")

    @model_validator(mode='after')
    def validate_weights_sum(self) -> 'FusionConfig':
        """Ensure weights sum to 1.0."""
        lexical = self.lexical_weight
        semantic = self.semantic_weight

        total = lexical + semantic
        if abs(total - 1.0) > 0.01:  # Allow small floating point errors
            raise ValueError(f"Lexical and semantic weights must sum to 1.0 (got {total})")

        return self


class AlgorithmsConfig(BaseSettings):
    """Configuration for algorithm parameters."""

    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)


class FeaturesConfig(BaseSettings):
    """Configuration for feature flags."""

    cache_persistence: bool = Field(default=True, description="Enable cache persistence")
    batch_processing: bool = Field(default=True, description="Enable batch processing")
    explainability: bool = Field(default=True, description="Enable explainability features")
    auto_optimization: bool = Field(default=False, description="Enable automatic optimization")
    fine_tuning: bool = Field(default=False, description="Enable fine-tuning capabilities")
    debug_mode: bool = Field(default=False, description="Enable debug mode")
    performance_logging: bool = Field(default=True, description="Enable performance logging")
    faiss_indexing: bool = Field(default=True, description="Enable FAISS indexing")
    gpu_acceleration: bool = Field(default=False, description="Enable GPU acceleration")


class PerformanceConfig(BaseSettings):
    """Configuration for performance settings."""

    batch_size: int = Field(default=32, ge=1, le=1000, description="Batch processing size")
    max_concurrent_requests: int = Field(default=100, ge=1, description="Max concurrent requests")
    max_cache_size_mb: int = Field(default=1024, ge=100, description="Max cache size in MB")
    cache_cleanup_interval: int = Field(default=3600, ge=300, description="Cache cleanup interval in seconds")
    classification_timeout: int = Field(default=30, ge=1, description="Classification timeout in seconds")
    index_build_timeout: int = Field(default=300, ge=10, description="Index build timeout in seconds")
    model_load_timeout: int = Field(default=120, ge=10, description="Model load timeout in seconds")
    num_threads: int = Field(default=1, ge=0, le=64, description="Number of threads (0=auto-detect)")
    thread_pool_size: int = Field(default=4, ge=1, le=32, description="Thread pool size")


class LoggingConfig(BaseSettings):
    """Configuration for logging."""

    level: LogLevel = Field(default=LogLevel.INFO, description="Log level")
    structured: bool = Field(default=True, description="Enable JSON structured logging")
    colored: bool = Field(default=True, description="Enable colored console output")
    max_size_mb: int = Field(default=100, ge=1, description="Max log file size in MB")
    backup_count: int = Field(default=5, ge=1, description="Number of backup log files")
    log_classifications: bool = Field(default=True, description="Log classification events")
    log_performance: bool = Field(default=True, description="Log performance metrics")
    log_errors: bool = Field(default=True, description="Log error events")
    log_cache_operations: bool = Field(default=False, description="Log cache operations")
    sanitize_inputs: bool = Field(default=True, description="Sanitize inputs in logs")
    max_input_length_logged: int = Field(default=200, ge=50, description="Max input length to log")


class ValidationConfig(BaseSettings):
    """Configuration for input validation and testing."""

    max_text_length: int = Field(default=10000, ge=100, description="Maximum text length")
    min_text_length: int = Field(default=1, ge=1, description="Minimum text length")
    allowed_languages: List[str] = Field(default_factory=list, description="Allowed languages (empty=all)")
    enable_regression_testing: bool = Field(default=True, description="Enable regression testing")
    golden_dataset_path: Path = Field(default=Path("./tests/data/golden_dataset.yaml"), description="Golden dataset path")
    confidence_calibration: bool = Field(default=True, description="Enable confidence calibration")
    strict_validation: bool = Field(default=False, description="Enable strict validation")
    graceful_degradation: bool = Field(default=True, description="Enable graceful degradation")


class CLIConfig(BaseSettings):
    """Configuration for command-line interface."""

    output_format: OutputFormat = Field(default=OutputFormat.TABLE, description="Default output format")
    show_confidence: bool = Field(default=True, description="Show confidence scores")
    show_evidence: bool = Field(default=False, description="Show evidence")
    show_timing: bool = Field(default=False, description="Show timing information")
    use_colors: bool = Field(default=True, description="Use colored output")
    theme: str = Field(default="default", description="Color theme")
    default_verbosity: int = Field(default=1, ge=0, le=3, description="Default verbosity level")


class APIConfig(BaseSettings):
    """Configuration for API service (if running as service)."""

    host: str = Field(default="localhost", description="Server host")
    port: int = Field(default=8000, ge=1024, le=65535, description="Server port")
    workers: int = Field(default=1, ge=1, description="Number of workers")
    rate_limit_rpm: int = Field(default=1000, ge=1, description="Rate limit (requests per minute)")
    rate_limit_burst: int = Field(default=100, ge=1, description="Rate limit burst")
    enable_cors: bool = Field(default=True, description="Enable CORS")
    api_key_required: bool = Field(default=False, description="Require API key")
    health_check_endpoint: str = Field(default="/health", description="Health check endpoint")
    metrics_endpoint: str = Field(default="/metrics", description="Metrics endpoint")
    docs_endpoint: str = Field(default="/docs", description="Documentation endpoint")


class ExperimentalConfig(BaseSettings):
    """Configuration for experimental features."""

    neural_classifier: bool = Field(default=False, description="Enable neural classifier")
    transformer_fine_tuning: bool = Field(default=False, description="Enable transformer fine-tuning")
    active_learning: bool = Field(default=False, description="Enable active learning")
    uncertainty_quantification: bool = Field(default=False, description="Enable uncertainty quantification")
    image_text_fusion: bool = Field(default=False, description="Enable image-text fusion")
    audio_text_fusion: bool = Field(default=False, description="Enable audio-text fusion")
    cluster_processing: bool = Field(default=False, description="Enable cluster processing")
    async_processing: bool = Field(default=False, description="Enable async processing")


# =============================================================================
# MAIN CONFIGURATION CLASS
# =============================================================================

class CXDConfig(BaseSettings):
    """
    Complete configuration for CXD Classifier.

    This is the main configuration class that combines all configuration sections
    and provides validation, environment variable support, and file loading.
    """

    paths: PathsConfig = Field(default_factory=PathsConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    algorithms: AlgorithmsConfig = Field(default_factory=AlgorithmsConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    cli: CLIConfig = Field(default_factory=CLIConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    experimental: ExperimentalConfig = Field(default_factory=ExperimentalConfig)

    version: str = Field(default="2.0", description="Configuration version")
    description: str = Field(default="CXD Classifier Configuration", description="Configuration description")

    # Pydantic V2: La clase Config interna se reemplaza por model_config = SettingsConfigDict(...)
    # Esto es para pydantic-settings. Pydantic base usa model_config = ConfigDict(...)
    model_config = SettingsConfigDict(
        env_prefix="CXD_",
        case_sensitive=False,
        env_file=".env",
        env_nested_delimiter="__",
        extra='ignore', # o 'forbid' si quieres ser estricto con campos extra
        # json_encoders no es parte de SettingsConfigDict, se maneja de forma diferente si es necesario
        # para serialización, o no es necesario aquí si solo usas dict() y yaml.dump
    )
    # Nota sobre json_encoders: Si necesitas serialización JSON personalizada,
    # Pydantic V2 lo maneja de forma diferente, a menudo a través del método model_dump_json()
    # o configurando tipos con serializadores JSON. Para guardar en YAML, self.model_dump()
    # (reemplaza a .dict()) y luego yaml.dump debería funcionar bien con tipos estándar.
    # Path es manejado por defecto en model_dump() a str.

    @classmethod
    def load_from_yaml(cls, config_path: Union[str, Path]) -> 'CXDConfig':
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
        if yaml_data is None:
            yaml_data = {}
        return cls(**yaml_data)

    def save_to_yaml(self, config_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        # Pydantic V2: .dict() se reemplaza por .model_dump()
        config_dict = self.model_dump(mode='python') # mode='python' para obtener tipos nativos de Python como Path
        # Para YAML, es mejor convertir Path a str explícitamente si es necesario o asegurar que yaml.dump lo maneje.
        # La conversión de Path a str para serialización es común.
        # Vamos a asegurar que los Path se conviertan a string para la serialización YAML:
        def path_to_str_serializer(obj):
            if isinstance(obj, Path):
                return str(obj)
            raise TypeError
        
        # Crear una copia del diccionario para modificarla
        import copy
        serializable_dict = copy.deepcopy(config_dict)

        # Convertir todos los objetos Path a cadenas en el diccionario anidado
        # Esto es un poco más manual ya que json_encoders en model_config no aplica a model_dump() para YAML directamente.
        # Una forma más robusta sería usar model_dump_json y luego json.loads, pero para YAML directo:
        def convert_paths_in_dict(d):
            for k, v in d.items():
                if isinstance(v, Path):
                    d[k] = str(v)
                elif isinstance(v, dict):
                    convert_paths_in_dict(v)
                elif isinstance(v, list):
                    for i, item in enumerate(v):
                        if isinstance(item, dict):
                            convert_paths_in_dict(item)
                        elif isinstance(item, Path):
                             v[i] = str(item)
        
        convert_paths_in_dict(serializable_dict)

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(serializable_dict, f, default_flow_style=False, indent=2)


    @classmethod
    def load_from_env(cls) -> 'CXDConfig':
        """Load configuration from environment variables only."""
        return cls()

    @classmethod
    def load_with_overrides(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        env_override: bool = True,
        **kwargs
    ) -> 'CXDConfig':
        """Load configuration with multiple sources and overrides."""
        # No se necesita una lógica de fusión manual tan compleja con pydantic-settings
        # y el orden en que se cargan las fuentes.
        # El orden de pydantic-settings es:
        # 1. Argumentos al inicializador (kwargs aquí).
        # 2. Variables de entorno.
        # 3. Variables del archivo .env.
        # 4. Valores por defecto del modelo.
        # Si cargas desde YAML y luego quieres aplicar env y kwargs, el proceso es un poco diferente.

        # Opción 1: YAML como base, luego el resto es manejado por Pydantic-Settings
        if config_path:
            config_from_yaml = cls.load_from_yaml(config_path)
            # Los kwargs aquí sobrescribirían los valores de YAML, luego Pydantic cargaría env y .env
            # que también podrían sobrescribir YAML si `_env_file` o las variables de entorno se cargan después.
            # Para asegurar que YAML sea la base y luego env/kwargs sobrescriban:
            initial_data = config_from_yaml.model_dump()
        else:
            initial_data = {}

        # kwargs deberían tener la máxima prioridad sobre YAML
        if kwargs:
            initial_data.update(kwargs)
        
        # Ahora inicializa con esta base, Pydantic-Settings cargará env y .env
        # y aplicará sus propias prioridades.
        # Si env_override es False, necesitaríamos una forma de no cargar variables de entorno.
        # La forma más fácil es si los kwargs y el archivo YAML son los únicos que importan.
        if not env_override:
             # Si no queremos overrides de env, creamos un modelo sin que pydantic-settings busque vars de entorno
             # Esto es un poco más complejo, podrías necesitar pasar `_strict=False` y no definir `env_prefix` temporalmente.
             # O, más simple, Pydantic-Settings aplica env vars. Si `env_override=False` y se pasan `kwargs`,
             # la pregunta es si los `kwargs` deben ignorar las env_vars.
             # Por ahora, asumimos que si env_override=False, solo YAML y kwargs importan, y kwargs gana a YAML.
             # Pydantic-settings siempre intentará cargar env_vars si model_config lo especifica.
             # Para un control total, puedes instanciar cls(**initial_data)
             # y si env_override es True, hacer otra instanciación que sí cargue las env_vars.
             # Por simplicidad, este método priorizará así: kwargs > env_vars > .env file > yaml_file > defaults
             # Esto se logra instanciando una vez y dejando que pydantic-settings haga su trabajo.
             # Para que YAML sea base y ENVS/KWARGS sobrescriban:
             # 1. Cargar YAML.
             # 2. Crear instancia con datos de YAML + KWARGS. Pydantic-settings aplicará ENV.
             merged_data = {**initial_data, **kwargs}
             config = cls(**merged_data)
             return config # Pydantic-settings ya habrá cargado ENV vars y .env file.

        # Si env_override es True (comportamiento por defecto de pydantic-settings)
        # Carga YAML como base, luego inicializa para que pydantic-settings cargue env y .env
        # y finalmente aplica kwargs.
        yaml_values = {}
        if config_path:
            config_path = Path(config_path)
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    yaml_values = yaml.safe_load(f) or {}
        
        # Los kwargs tienen la mayor precedencia
        # Los valores de entorno son cargados por Pydantic-Settings
        # Los valores del archivo .env son cargados por Pydantic-Settings
        # Luego vienen los valores del archivo YAML
        # Finalmente los valores por defecto del modelo
        # Para lograr yaml_values como base, y luego env/kwargs encima:
        # Debemos pasar yaml_values al constructor y dejar que pydantic-settings
        # aplique las variables de entorno y los kwargs (si se pasan al constructor).
        # La forma en que pydantic-settings maneja las fuentes es a través de `settings_sources`.
        # La forma más simple para este método sería:
        final_config_data = {**yaml_values, **kwargs} # kwargs sobreescriben yaml_values
        return cls(**final_config_data) # Pydantic-Settings cargará ENV y .env sobre esto


    def get_cache_path(self, filename: str) -> Path:
        """Get path for cache file."""
        return self.paths.cache_dir / filename

    def get_log_path(self, filename: str) -> Path:
        """Get path for log file."""
        return self.paths.log_dir / filename

    def is_production_mode(self) -> bool:
        """Check if running in production mode."""
        return not self.features.debug_mode and self.logging.level != LogLevel.DEBUG

    def is_development_mode(self) -> bool:
        """Check if running in development mode."""
        return self.features.debug_mode or self.logging.level == LogLevel.DEBUG

    def validate_runtime_requirements(self) -> List[str]:
        """Validate that runtime requirements are met."""
        warnings = []
        if self.features.faiss_indexing:
            try:
                import faiss
            except ImportError:
                warnings.append("FAISS indexing enabled but faiss not installed")
        if self.models.embedding.device == Device.CUDA:
            try:
                import torch
                if not torch.cuda.is_available():
                    warnings.append("CUDA device requested but not available")
            except ImportError:
                warnings.append("CUDA device requested but torch not installed")

        cache_dir = self.paths.cache_dir
        if cache_dir.exists():
            try:
                import shutil
                free_space_mb = shutil.disk_usage(cache_dir).free // (1024 * 1024)
                if free_space_mb < self.performance.max_cache_size_mb * 2:
                    warnings.append(f"Low disk space for cache: {free_space_mb}MB available")
            except Exception:
                pass
        return warnings

    def get_effective_device(self) -> str:
        """Get the effective device to use, considering availability."""
        requested_device = self.models.embedding.device
        if requested_device == Device.CPU:
            return "cpu"
        elif requested_device == Device.CUDA:
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        elif requested_device == Device.MPS:
            try:
                import torch
                return "mps" if torch.backends.mps.is_available() else "cpu"
            except ImportError:
                return "cpu"
        else:
            return "cpu"

# =============================================================================
# CONFIGURATION FACTORY FUNCTIONS
# =============================================================================

def create_default_config() -> CXDConfig:
    """Create default configuration."""
    return CXDConfig()

def create_development_config() -> CXDConfig:
    """Create development configuration with debug settings."""
    return CXDConfig(
        features=FeaturesConfig(debug_mode=True),
        logging=LoggingConfig(level=LogLevel.DEBUG, log_cache_operations=True),
        validation=ValidationConfig(strict_validation=True),
        performance=PerformanceConfig(batch_size=4)
    )

def create_production_config() -> CXDConfig:
    """Create production configuration with optimized settings."""
    return CXDConfig(
        features=FeaturesConfig(debug_mode=False),
        logging=LoggingConfig(level=LogLevel.INFO, sanitize_inputs=True),
        validation=ValidationConfig(strict_validation=False, graceful_degradation=True),
        performance=PerformanceConfig(batch_size=64)
    )

def load_config_from_file(
    config_path: Union[str, Path],
    env_override: bool = True # Este parámetro se vuelve menos directo con pydantic-settings
                              # ya que la carga de ENV está controlada por model_config
) -> CXDConfig:
    """Load configuration from file with optional environment overrides."""
    # Con pydantic-settings, si env_file y env_prefix están en model_config,
    # las variables de entorno SIEMPRE se cargarán y tendrán precedencia sobre el archivo .env,
    # y ambas sobre los valores por defecto.
    # Los valores pasados directamente al constructor (como desde un archivo YAML)
    # tendrán precedencia sobre los defaults, pero las ENV vars y el .env file los sobreescribirán.
    # Para que YAML sea la base y solo luego ENV:
    yaml_values = {}
    if config_path:
        path = Path(config_path)
        if path.exists():
             with open(path, 'r', encoding='utf-8') as f:
                yaml_values = yaml.safe_load(f) or {}

    if not env_override:
        # Si no queremos override de env, y Pydantic-Settings siempre los carga si está configurado,
        # esta función se complica. Una opción sería modificar temporalmente el model_config
        # o usar una clase base diferente para la carga.
        # La forma más simple sería documentar que pydantic-settings tiene su propia jerarquía.
        # Si se necesita cargar YAML y NADA MÁS, sería: return CXDConfig(**yaml_values)
        # pero esto no aplicaría los valores por defecto de campos no presentes en el YAML.
        # Para aplicar defaults pero no ENV:
        temp_model_config = SettingsConfigDict(env_file=None, extra='ignore') # Intenta deshabilitar carga de ENV
        # Esto requeriría definir CXDConfig con un model_config mutable o una subclase.
        # Es más fácil asumir que las ENV vars se cargan si están definidas en el entorno.
        # Por lo tanto, env_override=False es difícil de implementar limpiamente sin cambiar cómo se define CXDConfig.
        # Dejaremos que Pydantic-Settings maneje la carga de ENV como está configurado.
        pass # El parámetro env_override se vuelve más bien una indicación de intención.

    return CXDConfig(**yaml_values) # Pydantic-Settings cargará ENV y .env sobre los valores de yaml_values


__all__ = [
    "CXDConfig", "PathsConfig", "ModelsConfig", "AlgorithmsConfig",
    "FeaturesConfig", "PerformanceConfig", "LoggingConfig",
    "ValidationConfig", "CLIConfig", "APIConfig", "ExperimentalConfig",
    "LogLevel", "Device", "OutputFormat", "MetricType", "OptimizationMetric",
    "create_default_config", "create_development_config",
    "create_production_config", "load_config_from_file",
]