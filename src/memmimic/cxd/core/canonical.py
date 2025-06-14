"""
Canonical examples data structures for CXD Classifier.

This module defines the data structures used for managing canonical examples
that serve as training/reference data for the CXD classification system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib

from .types import CXDFunction


@dataclass
class CanonicalExample:
    """
    A canonical example for CXD classification.
    
    Represents a single training/reference example with rich metadata
    for use in semantic classification and system validation.
    """
    
    text: str
    function: CXDFunction
    id: str
    tags: List[str] = field(default_factory=list)
    category: str = ""
    quality_score: float = 1.0
    created_by: str = "system"
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize fields after initialization."""
        # Validate quality score bounds
        self.quality_score = max(0.0, min(1.0, self.quality_score))
        
        # Ensure tags is a list
        if not isinstance(self.tags, list):
            self.tags = [str(self.tags)] if self.tags else []
        
        # Generate ID if not provided
        if not self.id:
            self.id = self.generate_id()
        
        # Set default category based on function if not provided
        if not self.category:
            self.category = self._default_category_for_function()
    
    def __str__(self) -> str:
        """String representation showing function and text preview."""
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"{self.function.value}: {text_preview}"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"CanonicalExample(id='{self.id}', function={self.function.name}, "
                f"category='{self.category}', quality={self.quality_score})")
    
    @property
    def word_count(self) -> int:
        """Get word count of the text."""
        return len(self.text.split())
    
    @property
    def char_count(self) -> int:
        """Get character count of the text."""
        return len(self.text)
    
    @property
    def is_high_quality(self, threshold: float = 0.8) -> bool:
        """Check if example is considered high quality."""
        return self.quality_score >= threshold
    
    def generate_id(self) -> str:
        """
        Generate a unique ID based on function, text, and category.
        
        Returns:
            str: Generated ID
        """
        # Create deterministic ID from content
        content = f"{self.function.value}:{self.category}:{self.text}"
        hash_obj = hashlib.md5(content.encode('utf-8'))
        return f"{self.function.value.lower()}_{hash_obj.hexdigest()[:8]}"
    
    def _default_category_for_function(self) -> str:
        """Get default category based on function."""
        defaults = {
            CXDFunction.CONTROL: "general",
            CXDFunction.CONTEXT: "relation", 
            CXDFunction.DATA: "processing"
        }
        return defaults.get(self.function, "general")
    
    def add_tag(self, tag: str) -> None:
        """
        Add a tag if not already present.
        
        Args:
            tag: Tag to add
        """
        if tag not in self.tags:
            self.tags.append(tag)
    
    def remove_tag(self, tag: str) -> bool:
        """
        Remove a tag.
        
        Args:
            tag: Tag to remove
            
        Returns:
            bool: True if tag was removed, False if not found
        """
        if tag in self.tags:
            self.tags.remove(tag)
            return True
        return False
    
    def has_tag(self, tag: str) -> bool:
        """Check if example has a specific tag."""
        return tag in self.tags
    
    def update_quality_score(self, new_score: float, method: str = "replace") -> None:
        """
        Update quality score using different methods.
        
        Args:
            new_score: New quality score
            method: Update method ("replace", "average", "max", "min")
        """
        new_score = max(0.0, min(1.0, new_score))
        
        if method == "replace":
            self.quality_score = new_score
        elif method == "average":
            self.quality_score = (self.quality_score + new_score) / 2
        elif method == "max":
            self.quality_score = max(self.quality_score, new_score)
        elif method == "min":
            self.quality_score = min(self.quality_score, new_score)
        
        # Update modification timestamp
        self.last_modified = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dict: Serializable dictionary representation
        """
        return {
            "text": self.text,
            "function": self.function.value,
            "id": self.id,
            "tags": self.tags,
            "category": self.category,
            "quality_score": self.quality_score,
            "created_by": self.created_by,
            "last_modified": self.last_modified,
            "metadata": self.metadata,
            "word_count": self.word_count,
            "char_count": self.char_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CanonicalExample':
        """
        Create CanonicalExample from dictionary.
        
        Args:
            data: Dictionary with example data
            
        Returns:
            CanonicalExample: Reconstructed example instance
        """
        # Handle both string and enum function values
        function_value = data["function"]
        if isinstance(function_value, str):
            function = CXDFunction.from_string(function_value)
        else:
            function = function_value
        
        return cls(
            text=data["text"],
            function=function,
            id=data.get("id", ""),
            tags=data.get("tags", []),
            category=data.get("category", ""),
            quality_score=data.get("quality_score", 1.0),
            created_by=data.get("created_by", "system"),
            last_modified=data.get("last_modified", ""),
            metadata=data.get("metadata", {})
        )
    
    def similarity_score(self, other: 'CanonicalExample') -> float:
        """
        Calculate similarity score with another example.
        
        Args:
            other: Other canonical example
            
        Returns:
            float: Similarity score (0.0 to 1.0)
        """
        score = 0.0
        
        # Function match (40% weight)
        if self.function == other.function:
            score += 0.4
        
        # Category match (20% weight)
        if self.category == other.category:
            score += 0.2
        
        # Tag overlap (20% weight)
        if self.tags and other.tags:
            common_tags = set(self.tags) & set(other.tags)
            all_tags = set(self.tags) | set(other.tags)
            tag_similarity = len(common_tags) / len(all_tags)
            score += 0.2 * tag_similarity
        
        # Text similarity (20% weight) - simple Jaccard similarity
        words1 = set(self.text.lower().split())
        words2 = set(other.text.lower().split())
        if words1 and words2:
            common_words = words1 & words2
            all_words = words1 | words2
            text_similarity = len(common_words) / len(all_words)
            score += 0.2 * text_similarity
        
        return score


@dataclass
class CanonicalExampleSet:
    """
    A collection of canonical examples with metadata and statistics.
    
    Provides functionality for managing, analyzing, and validating
    sets of canonical examples.
    """
    
    examples: List[CanonicalExample]
    version: str = "1.0"
    description: str = ""
    total_examples: int = field(init=False)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived fields after initialization."""
        self.total_examples = len(self.examples)
    
    def __len__(self) -> int:
        """Number of examples in set."""
        return len(self.examples)
    
    def __iter__(self):
        """Iterate over examples."""
        return iter(self.examples)
    
    def __getitem__(self, index: int) -> CanonicalExample:
        """Get example by index."""
        return self.examples[index]
    
    @property
    def function_counts(self) -> Dict[CXDFunction, int]:
        """Get count of examples per function."""
        counts = {}
        for example in self.examples:
            counts[example.function] = counts.get(example.function, 0) + 1
        return counts
    
    @property
    def category_counts(self) -> Dict[str, int]:
        """Get count of examples per category."""
        counts = {}
        for example in self.examples:
            counts[example.category] = counts.get(example.category, 0) + 1
        return counts
    
    @property
    def average_quality(self) -> float:
        """Calculate average quality score."""
        if not self.examples:
            return 0.0
        return sum(ex.quality_score for ex in self.examples) / len(self.examples)
    
    @property
    def high_quality_examples(self, threshold: float = 0.8) -> List[CanonicalExample]:
        """Get high quality examples."""
        return [ex for ex in self.examples if ex.is_high_quality(threshold)]
    
    def get_examples_by_function(self, function: CXDFunction) -> List[CanonicalExample]:
        """
        Get examples for specific function.
        
        Args:
            function: CXD function to filter by
            
        Returns:
            List[CanonicalExample]: Examples for function
        """
        return [ex for ex in self.examples if ex.function == function]
    
    def get_examples_by_category(self, category: str) -> List[CanonicalExample]:
        """
        Get examples for specific category.
        
        Args:
            category: Category to filter by
            
        Returns:
            List[CanonicalExample]: Examples in category
        """
        return [ex for ex in self.examples if ex.category == category]
    
    def get_examples_by_tag(self, tag: str) -> List[CanonicalExample]:
        """
        Get examples containing specific tag.
        
        Args:
            tag: Tag to search for
            
        Returns:
            List[CanonicalExample]: Examples with tag
        """
        return [ex for ex in self.examples if ex.has_tag(tag)]
    
    def filter_by_quality(self, min_quality: float) -> 'CanonicalExampleSet':
        """
        Create new set with examples above quality threshold.
        
        Args:
            min_quality: Minimum quality score
            
        Returns:
            CanonicalExampleSet: Filtered example set
        """
        filtered_examples = [ex for ex in self.examples if ex.quality_score >= min_quality]
        return CanonicalExampleSet(
            examples=filtered_examples,
            version=self.version,
            description=f"Filtered (quality >= {min_quality}): {self.description}",
            metadata=self.metadata.copy()
        )
    
    def add_example(self, example: CanonicalExample) -> None:
        """
        Add an example to the set.
        
        Args:
            example: Example to add
        """
        self.examples.append(example)
        self.total_examples = len(self.examples)
    
    def remove_example(self, example_id: str) -> bool:
        """
        Remove example by ID.
        
        Args:
            example_id: ID of example to remove
            
        Returns:
            bool: True if example was removed, False if not found
        """
        original_length = len(self.examples)
        self.examples = [ex for ex in self.examples if ex.id != example_id]
        self.total_examples = len(self.examples)
        return len(self.examples) < original_length
    
    def find_by_id(self, example_id: str) -> Optional[CanonicalExample]:
        """
        Find example by ID.
        
        Args:
            example_id: ID to search for
            
        Returns:
            CanonicalExample: Found example or None
        """
        for example in self.examples:
            if example.id == example_id:
                return example
        return None
    
    def find_duplicates(self, similarity_threshold: float = 0.9) -> List[tuple[CanonicalExample, CanonicalExample, float]]:
        """
        Find potentially duplicate examples.
        
        Args:
            similarity_threshold: Threshold for considering examples duplicates
            
        Returns:
            List[Tuple]: List of (example1, example2, similarity_score) tuples
        """
        duplicates = []
        
        for i, ex1 in enumerate(self.examples):
            for ex2 in self.examples[i+1:]:
                similarity = ex1.similarity_score(ex2)
                if similarity >= similarity_threshold:
                    duplicates.append((ex1, ex2, similarity))
        
        return duplicates
    
    def validate_set(self) -> Dict[str, Any]:
        """
        Validate the example set and return validation report.
        
        Returns:
            Dict: Validation report with warnings and statistics
        """
        report = {
            "total_examples": len(self.examples),
            "function_distribution": self.function_counts,
            "category_distribution": self.category_counts,
            "average_quality": self.average_quality,
            "warnings": [],
            "errors": []
        }
        
        # Check function balance
        function_counts = self.function_counts
        if function_counts:
            min_count = min(function_counts.values())
            max_count = max(function_counts.values())
            if max_count > min_count * 2:  # More than 2x imbalance
                report["warnings"].append(f"Function imbalance: {function_counts}")
        
        # Check for low quality examples
        low_quality = [ex for ex in self.examples if ex.quality_score < 0.5]
        if low_quality:
            report["warnings"].append(f"{len(low_quality)} low quality examples (< 0.5)")
        
        # Check for duplicates
        duplicates = self.find_duplicates(0.95)
        if duplicates:
            report["warnings"].append(f"{len(duplicates)} potential duplicate pairs")
        
        # Check for missing categories
        no_category = [ex for ex in self.examples if not ex.category]
        if no_category:
            report["warnings"].append(f"{len(no_category)} examples without category")
        
        return report
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dict: Serializable dictionary representation
        """
        return {
            "version": self.version,
            "description": self.description,
            "total_examples": self.total_examples,
            "examples": [ex.to_dict() for ex in self.examples],
            "metadata": self.metadata,
            "statistics": {
                "function_counts": {f.value: count for f, count in self.function_counts.items()},
                "category_counts": self.category_counts,
                "average_quality": self.average_quality
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CanonicalExampleSet':
        """
        Create CanonicalExampleSet from dictionary.
        
        Args:
            data: Dictionary with example set data
            
        Returns:
            CanonicalExampleSet: Reconstructed example set
        """
        examples = [CanonicalExample.from_dict(ex_data) for ex_data in data["examples"]]
        
        return cls(
            examples=examples,
            version=data.get("version", "1.0"),
            description=data.get("description", ""),
            metadata=data.get("metadata", {})
        )


# Export canonical example types
__all__ = [
    "CanonicalExample",
    "CanonicalExampleSet",
]
