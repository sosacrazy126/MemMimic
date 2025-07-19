"""
Canonical examples providers for CXD Classifier.

This module implements providers for loading and managing canonical examples
from various sources (YAML files, JSON files, etc.) that serve as training
data for semantic classification.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from ..core.canonical import CanonicalExample
from ..core.interfaces import CanonicalExampleProvider
from ..core.types import CXDFunction


class YamlExampleProvider(CanonicalExampleProvider):
    """
    Provider for loading canonical examples from YAML files.

    Supports the rich YAML structure defined in the vision document
    with metadata, quality scores, and categorization.
    """

    def __init__(
        self,
        yaml_path: Union[str, Path],
        encoding: str = "utf-8",
        cache_parsed: bool = True,
    ):
        """
        Initialize YAML example provider.

        Args:
            yaml_path: Path to YAML file with canonical examples
            encoding: File encoding (default: utf-8)
            cache_parsed: Whether to cache parsed examples in memory
        """
        self.yaml_path = Path(yaml_path)
        self.encoding = encoding
        self.cache_parsed = cache_parsed

        # Cache
        self._cached_examples: Optional[Dict[CXDFunction, List[CanonicalExample]]] = (
            None
        )
        self._cached_checksum: Optional[str] = None
        self._last_load_time: Optional[float] = None

        # Validation
        self._validate_file_exists()

    def _validate_file_exists(self) -> None:
        """Validate that YAML file exists."""
        if not self.yaml_path.exists():
            raise FileNotFoundError(
                f"Canonical examples file not found: {self.yaml_path}"
            )

        if not self.yaml_path.is_file():
            raise ValueError(f"Path is not a file: {self.yaml_path}")

    def load_examples(self) -> Dict[CXDFunction, List[CanonicalExample]]:
        """
        Load canonical examples organized by CXD function.

        Returns:
            Dict: Mapping from CXDFunction to list of examples

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            yaml.YAMLError: If YAML is invalid
            ValueError: If examples structure is invalid
        """
        # Check cache validity
        if self._is_cache_valid():
            return self._cached_examples

        # Load and parse YAML
        with open(self.yaml_path, "r", encoding=self.encoding) as f:
            yaml_data = yaml.safe_load(f)

        if not yaml_data:
            raise ValueError(f"Empty or invalid YAML file: {self.yaml_path}")

        # Parse examples from YAML structure
        examples_by_function = self._parse_yaml_structure(yaml_data)

        # Cache if enabled
        if self.cache_parsed:
            self._cached_examples = examples_by_function
            self._cached_checksum = self.get_checksum()
            self._last_load_time = time.time()

        return examples_by_function

    def _is_cache_valid(self) -> bool:
        """Check if cached examples are still valid."""
        if not self.cache_parsed or self._cached_examples is None:
            return False

        # Check if file has been modified
        current_checksum = self.get_checksum()
        return current_checksum == self._cached_checksum

    def _parse_yaml_structure(
        self, yaml_data: Dict[str, Any]
    ) -> Dict[CXDFunction, List[CanonicalExample]]:
        """
        Parse YAML structure into canonical examples.

        Expected YAML structure:
        ```yaml
        version: "1.2"
        description: "Ejemplos canónicos para clasificador CXD"
        examples:
          CONTROL:
            - text: "Buscar información en la base de datos"
              id: "ctrl_001"
              tags: ["search", "database"]
              category: "search"
              quality_score: 0.9
              created_by: "admin"
              last_modified: "2025-05-20"
          CONTEXT:
            - text: "Esta conversación se relaciona con trabajo anterior"
              id: "ctx_001"
              ...
        ```

        Args:
            yaml_data: Parsed YAML data

        Returns:
            Dict: Examples organized by function
        """
        examples_by_function = {}

        # Validate top-level structure
        if "examples" not in yaml_data:
            raise ValueError("YAML must contain 'examples' section")

        examples_section = yaml_data["examples"]
        if not isinstance(examples_section, dict):
            raise ValueError("'examples' section must be a dictionary")

        # Parse examples for each function
        for function_name, examples_list in examples_section.items():
            try:
                function = CXDFunction.from_string(function_name)
            except ValueError:
                raise ValueError(f"Invalid CXD function in YAML: {function_name}")

            if not isinstance(examples_list, list):
                raise ValueError(f"Examples for {function_name} must be a list")

            # Parse individual examples
            parsed_examples = []
            for i, example_data in enumerate(examples_list):
                try:
                    example = self._parse_single_example(example_data, function, i)
                    parsed_examples.append(example)
                except Exception as e:
                    raise ValueError(
                        f"Error parsing example {i} for {function_name}: {e}"
                    )

            examples_by_function[function] = parsed_examples

        return examples_by_function

    def _parse_single_example(
        self, example_data: Dict[str, Any], function: CXDFunction, index: int
    ) -> CanonicalExample:
        """
        Parse a single example from YAML data.

        Args:
            example_data: Example data from YAML
            function: CXD function for this example
            index: Index in the list (for auto-generating IDs)

        Returns:
            CanonicalExample: Parsed example
        """
        # Validate required fields
        if "text" not in example_data:
            raise ValueError("Example must have 'text' field")

        text = example_data["text"]
        if not text or not isinstance(text, str):
            raise ValueError("'text' field must be a non-empty string")

        # Parse optional fields with defaults
        example_id = example_data.get("id", f"{function.value.lower()}_{index:03d}")
        tags = example_data.get("tags", [])
        category = example_data.get("category", "")
        quality_score = example_data.get("quality_score", 1.0)
        created_by = example_data.get("created_by", "system")
        last_modified = example_data.get("last_modified", "")
        metadata = example_data.get("metadata", {})

        # Validate types
        if not isinstance(tags, list):
            tags = [str(tags)] if tags else []

        if not isinstance(quality_score, (int, float)):
            quality_score = 1.0

        # Create example
        return CanonicalExample(
            text=text,
            function=function,
            id=example_id,
            tags=tags,
            category=category,
            quality_score=float(quality_score),
            created_by=created_by,
            last_modified=last_modified,
            metadata=metadata,
        )

    def get_checksum(self) -> str:
        """
        Get checksum/hash of YAML file for cache invalidation.

        Returns:
            str: SHA-256 hash of file content
        """
        try:
            with open(self.yaml_path, "rb") as f:
                file_content = f.read()
            return hashlib.sha256(file_content).hexdigest()[:16]  # First 16 chars
        except Exception:
            # If we can't read file, return timestamp-based hash
            stat = self.yaml_path.stat()
            content = f"{self.yaml_path}:{stat.st_mtime}:{stat.st_size}"
            return hashlib.sha256(content.encode()).hexdigest()[:16]

    def reload_examples(self) -> Dict[CXDFunction, List[CanonicalExample]]:
        """
        Force reload examples from file.

        Returns:
            Dict: Freshly loaded examples
        """
        # Clear cache
        self._cached_examples = None
        self._cached_checksum = None
        self._last_load_time = None

        # Reload
        return self.load_examples()

    def save_examples(
        self, examples_by_function: Dict[CXDFunction, List[CanonicalExample]]
    ) -> None:
        """
        Save examples back to YAML file.

        Args:
            examples_by_function: Examples to save
        """
        # Build YAML structure
        yaml_structure = {
            "version": "1.2",
            "description": f"Canonical examples for CXD classifier - Updated {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "examples": {},
        }

        for function, examples in examples_by_function.items():
            yaml_structure["examples"][function.value] = [
                example.to_dict() for example in examples
            ]

        # Write to file
        with open(self.yaml_path, "w", encoding=self.encoding) as f:
            yaml.dump(
                yaml_structure,
                f,
                default_flow_style=False,
                indent=2,
                ensure_ascii=False,
            )

        # Invalidate cache
        self._cached_examples = None
        self._cached_checksum = None

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata from YAML file.

        Returns:
            Dict: Metadata including version, description, etc.
        """
        with open(self.yaml_path, "r", encoding=self.encoding) as f:
            yaml_data = yaml.safe_load(f)

        return {
            "version": yaml_data.get("version", "unknown"),
            "description": yaml_data.get("description", ""),
            "file_path": str(self.yaml_path),
            "file_size_bytes": self.yaml_path.stat().st_size,
            "last_modified": time.ctime(self.yaml_path.stat().st_mtime),
            "checksum": self.get_checksum(),
        }


class JsonExampleProvider(CanonicalExampleProvider):
    """
    Provider for loading canonical examples from JSON files.

    Alternative to YAML provider for environments where JSON is preferred.
    """

    def __init__(self, json_path: Union[str, Path], encoding: str = "utf-8"):
        """
        Initialize JSON example provider.

        Args:
            json_path: Path to JSON file with canonical examples
            encoding: File encoding
        """
        self.json_path = Path(json_path)
        self.encoding = encoding

        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.json_path}")

    def load_examples(self) -> Dict[CXDFunction, List[CanonicalExample]]:
        """Load examples from JSON file."""
        with open(self.json_path, "r", encoding=self.encoding) as f:
            json_data = json.load(f)

        examples_by_function = {}

        for function_name, examples_list in json_data.get("examples", {}).items():
            function = CXDFunction.from_string(function_name)

            parsed_examples = []
            for example_data in examples_list:
                example = CanonicalExample.from_dict(example_data)
                example.function = function  # Ensure function is set correctly
                parsed_examples.append(example)

            examples_by_function[function] = parsed_examples

        return examples_by_function

    def get_checksum(self) -> str:
        """Get checksum of JSON file."""
        with open(self.json_path, "rb") as f:
            content = f.read()
        return hashlib.sha256(content).hexdigest()[:16]


class InMemoryExampleProvider(CanonicalExampleProvider):
    """
    Provider for canonical examples stored in memory.

    Useful for testing or programmatic example management.
    """

    def __init__(
        self,
        examples_by_function: Optional[
            Dict[CXDFunction, List[CanonicalExample]]
        ] = None,
    ):
        """
        Initialize in-memory provider.

        Args:
            examples_by_function: Initial examples to store
        """
        self._examples = examples_by_function or {}
        self._checksum = self._calculate_checksum()

    def load_examples(self) -> Dict[CXDFunction, List[CanonicalExample]]:
        """Get examples from memory."""
        return self._examples.copy()

    def get_checksum(self) -> str:
        """Get checksum based on current examples."""
        return self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calculate checksum based on examples content."""
        content = ""
        for function in sorted(self._examples.keys(), key=lambda f: f.value):
            for example in sorted(self._examples[function], key=lambda e: e.id):
                content += f"{function.value}:{example.id}:{example.text}:{example.quality_score}"

        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def add_example(self, function: CXDFunction, example: CanonicalExample) -> None:
        """
        Add an example to memory.

        Args:
            function: CXD function
            example: Example to add
        """
        if function not in self._examples:
            self._examples[function] = []

        self._examples[function].append(example)
        self._checksum = self._calculate_checksum()

    def remove_example(self, function: CXDFunction, example_id: str) -> bool:
        """
        Remove an example from memory.

        Args:
            function: CXD function
            example_id: ID of example to remove

        Returns:
            bool: True if example was removed
        """
        if function not in self._examples:
            return False

        original_length = len(self._examples[function])
        self._examples[function] = [
            ex for ex in self._examples[function] if ex.id != example_id
        ]

        if len(self._examples[function]) < original_length:
            self._checksum = self._calculate_checksum()
            return True

        return False


class CompositeExampleProvider(CanonicalExampleProvider):
    """
    Provider that combines examples from multiple sources.

    Useful for loading examples from multiple files or combining
    file-based examples with programmatic examples.
    """

    def __init__(self, providers: List[CanonicalExampleProvider]):
        """
        Initialize composite provider.

        Args:
            providers: List of providers to combine
        """
        self.providers = providers
        if not providers:
            raise ValueError("At least one provider must be specified")

    def load_examples(self) -> Dict[CXDFunction, List[CanonicalExample]]:
        """Load and merge examples from all providers."""
        combined_examples = {}

        for provider in self.providers:
            provider_examples = provider.load_examples()

            for function, examples in provider_examples.items():
                if function not in combined_examples:
                    combined_examples[function] = []

                combined_examples[function].extend(examples)

        # Remove duplicates based on ID
        for function, examples in combined_examples.items():
            seen_ids = set()
            unique_examples = []

            for example in examples:
                if example.id not in seen_ids:
                    unique_examples.append(example)
                    seen_ids.add(example.id)

            combined_examples[function] = unique_examples

        return combined_examples

    def get_checksum(self) -> str:
        """Get combined checksum from all providers."""
        checksums = [provider.get_checksum() for provider in self.providers]
        combined_content = ":".join(sorted(checksums))
        return hashlib.sha256(combined_content.encode()).hexdigest()[:16]


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_yaml_provider(yaml_path: Union[str, Path], **kwargs) -> YamlExampleProvider:
    """
    Create YAML example provider.

    Args:
        yaml_path: Path to YAML file
        **kwargs: Additional arguments for YamlExampleProvider

    Returns:
        YamlExampleProvider: Configured provider
    """
    return YamlExampleProvider(yaml_path, **kwargs)


def create_json_provider(json_path: Union[str, Path], **kwargs) -> JsonExampleProvider:
    """
    Create JSON example provider.

    Args:
        json_path: Path to JSON file
        **kwargs: Additional arguments for JsonExampleProvider

    Returns:
        JsonExampleProvider: Configured provider
    """
    return JsonExampleProvider(json_path, **kwargs)


def create_default_provider(
    config_dir: Union[str, Path] = "./config",
) -> CanonicalExampleProvider:
    """
    Create default example provider based on available files.

    Args:
        config_dir: Directory to search for example files

    Returns:
        CanonicalExampleProvider: Best available provider

    Raises:
        FileNotFoundError: If no example files are found
    """
    config_dir = Path(config_dir)

    # Look for YAML files first (preferred)
    yaml_files = list(config_dir.glob("canonical_examples*.yaml")) + list(
        config_dir.glob("examples*.yaml")
    )
    if yaml_files:
        return YamlExampleProvider(yaml_files[0])

    # Look for JSON files
    json_files = list(config_dir.glob("canonical_examples*.json")) + list(
        config_dir.glob("examples*.json")
    )
    if json_files:
        return JsonExampleProvider(json_files[0])

    raise FileNotFoundError(f"No canonical examples files found in {config_dir}")


# Export providers
__all__ = [
    "YamlExampleProvider",
    "JsonExampleProvider",
    "InMemoryExampleProvider",
    "CompositeExampleProvider",
    "create_yaml_provider",
    "create_json_provider",
    "create_default_provider",
]
