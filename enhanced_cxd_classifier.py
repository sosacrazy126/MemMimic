#!/usr/bin/env python3
"""
Enhanced CXD Classification System (Dependency-Free)

This module provides sophisticated CXD classification without ML dependencies.
Extracts the best patterns from the original lexical classifier and adds
confidence scoring, structural analysis, and comprehensive evidence tracking.

Philosophy: Ruthless simplicity - maximum accuracy with zero external dependencies.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# CORE TYPES (Simplified from original)
# =============================================================================

class CXDFunction(Enum):
    """Cognitive function categories"""
    CONTROL = "CONTROL"
    CONTEXT = "CONTEXT"
    DATA = "DATA"


@dataclass
class ClassificationResult:
    """Result of CXD classification"""
    function: str  # "CONTROL", "CONTEXT", "DATA", or "unknown"
    confidence: float  # 0.0 to 1.0
    evidence: List[str]  # Human-readable evidence
    scores: Dict[str, float]  # Scores for each function


# =============================================================================
# ENHANCED CXD CLASSIFIER
# =============================================================================

class EnhancedCXDClassifier:
    """
    Enhanced CXD classifier using multi-layered pattern analysis.

    Combines:
    1. Regex patterns (bilingual: English + Spanish)
    2. Weighted keywords
    3. Linguistic indicators
    4. Structural analysis
    5. Confidence thresholds

    No external dependencies - pure Python with regex.
    """

    def __init__(self):
        """Initialize classifier with patterns and keywords"""
        self.patterns = self._build_patterns()
        self.keywords = self._build_keywords()
        self.indicators = self._build_indicators()

        # Confidence thresholds
        self.MIN_CONFIDENCE = 0.30  # Below this = "unknown"
        self.HIGH_CONFIDENCE = 0.70  # Above this = very confident

    def _build_patterns(self) -> Dict[str, List[Tuple[str, float, str]]]:
        """
        Build regex patterns for each CXD function.

        Returns:
            Dict mapping function -> [(pattern, confidence, category), ...]
        """
        return {
            "CONTROL": [
                # Search and retrieval (0.9 confidence)
                (r"\b(search|find|locate|look\s+for|retrieve)\b.*\b(information|data|documents|results)\b",
                 0.9, "search"),
                (r"\b(buscar|encontrar|localizar|rastrear)\b.*\b(información|datos|documentos|resultados)\b",
                 0.9, "search"),

                # Filter and selection (0.85)
                (r"\b(filter|select|exclude|narrow\s+down|refine)\b.*\b(results|data|options|choices)\b",
                 0.85, "filter"),
                (r"\b(filtrar|seleccionar|cribar|excluir)\b.*\b(resultados|información|datos|opciones)\b",
                 0.85, "filter"),

                # Control and management (0.80)
                (r"\b(control|direct|manage|coordinate|supervise)\b.*\b(process|flow|system|resources)\b",
                 0.80, "control"),
                (r"\b(controlar|dirigir|gestionar|coordinar|supervisar)\b.*\b(proceso|flujo|sistema|recursos)\b",
                 0.80, "control"),

                # Decision making (0.85)
                (r"\b(decide|determine|resolve|choose|establish)\b.*\b(what|which|how|strategy|action)\b",
                 0.85, "decision"),
                (r"\b(decidir|determinar|resolver|elegir|establecer)\b.*\b(qué|cuál|cómo|estrategia|acción)\b",
                 0.85, "decision"),
            ],

            "CONTEXT": [
                # Relationships (0.90)
                (r"\b(relates?|connects?|links?|associates?)\b.*\b(to|with|previous|prior|earlier|work)\b",
                 0.90, "relation"),
                (r"\b(relaciona|conecta|vincula|asocia)\b.*\b(con|anterior|previo|experiencias|trabajo)\b",
                 0.90, "relation"),

                # References (0.85)
                (r"\b(reference|mention|refer\s+to|cite)\b.*\b(previous|prior|earlier|discussion|conversation)\b",
                 0.85, "reference"),
                (r"\b(referencia|menciona|alude|cita)\b.*\b(anterior|previo|discusión|conversación)\b",
                 0.85, "reference"),

                # Contextualization (0.80)
                (r"\b(situate|contextualize|frame|place)\b.*\b(context|framework|situation|environment)\b",
                 0.80, "contextualization"),
                (r"\b(situar|contextualizar|enmarcar|ubicar)\b.*\b(marco|contexto|situación|ambiente)\b",
                 0.80, "contextualization"),

                # Historical/memory (0.85)
                (r"\b(remember|recall|continue|based\s+on)\b.*\b(conversation|discussion|topic|thread)\b",
                 0.85, "memory"),
                (r"\b(recordar|retomar|continuar|basándome)\b.*\b(conversación|discusión|tema|hilo)\b",
                 0.85, "memory"),
            ],

            "DATA": [
                # Processing (0.90)
                (r"\b(process|analyze|examine|compute|calculate)\b.*\b(information|data|results|patterns)\b",
                 0.90, "processing"),
                (r"\b(procesar|analizar|examinar|computar|calcular)\b.*\b(información|datos|resultados|patrones)\b",
                 0.90, "processing"),

                # Transformation (0.85)
                (r"\b(transform|convert|modify|adapt|reformat)\b.*\b(data|format|structure)\b",
                 0.85, "transformation"),
                (r"\b(transformar|convertir|modificar|adaptar|reformatear)\b.*\b(datos|formato|estructura)\b",
                 0.85, "transformation"),

                # Generation (0.80)
                (r"\b(generate|create|produce|elaborate|build)\b.*\b(synthesis|summary|report|visualization)\b",
                 0.80, "generation"),
                (r"\b(generar|crear|producir|elaborar|construir)\b.*\b(síntesis|resumen|reporte|visualización)\b",
                 0.80, "generation"),

                # Extraction (0.85)
                (r"\b(extract|derive|obtain|get|deduce)\b.*\b(insights|conclusions|patterns|trends)\b",
                 0.85, "extraction"),
                (r"\b(extraer|derivar|obtener|conseguir|deducir)\b.*\b(insights|conclusiones|patrones|tendencias)\b",
                 0.85, "extraction"),

                # Organization (0.80)
                (r"\b(organize|structure|classify|categorize|order)\b.*\b(information|data|content)\b",
                 0.80, "organization"),
                (r"\b(organizar|estructurar|clasificar|categorizar|ordenar)\b.*\b(información|datos|contenido)\b",
                 0.80, "organization"),
            ]
        }

    def _build_keywords(self) -> Dict[str, Dict[str, float]]:
        """
        Build keyword dictionaries with confidence weights.

        Keywords have lower weight (0.6-0.8) than patterns.
        Returns:
            Dict mapping function -> {keyword: weight, ...}
        """
        return {
            "CONTROL": {
                # Search (0.8)
                "search": 0.8, "buscar": 0.8, "find": 0.8, "encontrar": 0.8,
                "locate": 0.7, "localizar": 0.7, "track": 0.7, "rastrear": 0.7,
                # Filter (0.8)
                "filter": 0.8, "filtrar": 0.8, "select": 0.8, "seleccionar": 0.8,
                "exclude": 0.7, "excluir": 0.7, "screen": 0.7, "cribar": 0.7,
                # Control (0.8)
                "control": 0.8, "controlar": 0.8, "manage": 0.8, "gestionar": 0.8,
                "direct": 0.7, "dirigir": 0.7, "coordinate": 0.7, "coordinar": 0.7,
                # Decision (0.8)
                "decide": 0.8, "decidir": 0.8, "determine": 0.8, "determinar": 0.8,
                "choose": 0.8, "elegir": 0.8, "resolve": 0.7, "resolver": 0.7,
            },

            "CONTEXT": {
                # Relation (0.8)
                "relate": 0.8, "relacionar": 0.8, "connect": 0.8, "conectar": 0.8,
                "link": 0.7, "vincular": 0.7, "associate": 0.8, "asociar": 0.8,
                # Reference (0.8)
                "reference": 0.8, "referenciar": 0.8, "mention": 0.7, "mencionar": 0.7,
                "cite": 0.7, "citar": 0.7, "allude": 0.6, "aludir": 0.6,
                # Context (0.8)
                "context": 0.8, "contexto": 0.8, "framework": 0.7, "marco": 0.7,
                "situation": 0.7, "situación": 0.7, "environment": 0.6, "ambiente": 0.6,
                # Memory (0.8)
                "remember": 0.8, "recordar": 0.8, "previous": 0.7, "anterior": 0.7,
                "prior": 0.7, "previo": 0.7, "conversation": 0.8, "conversación": 0.8,
            },

            "DATA": {
                # Processing (0.8)
                "process": 0.8, "procesar": 0.8, "analyze": 0.8, "analizar": 0.8,
                "examine": 0.7, "examinar": 0.7, "compute": 0.8, "computar": 0.8,
                # Transformation (0.8)
                "transform": 0.8, "transformar": 0.8, "convert": 0.8, "convertir": 0.8,
                "modify": 0.7, "modificar": 0.7, "adapt": 0.7, "adaptar": 0.7,
                # Generation (0.8)
                "generate": 0.8, "generar": 0.8, "create": 0.7, "crear": 0.7,
                "produce": 0.7, "producir": 0.7, "elaborate": 0.7, "elaborar": 0.7,
                # Extraction (0.8)
                "extract": 0.8, "extraer": 0.8, "derive": 0.7, "derivar": 0.7,
                "obtain": 0.6, "obtener": 0.6, "deduce": 0.7, "deducir": 0.7,
                # Organization (0.8)
                "organize": 0.8, "organizar": 0.8, "structure": 0.8, "estructurar": 0.8,
                "classify": 0.7, "clasificar": 0.7, "categorize": 0.7, "categorizar": 0.7,
            }
        }

    def _build_indicators(self) -> Dict[str, List[str]]:
        """
        Build linguistic indicators (lower weight: 0.3)

        Returns:
            Dict mapping function -> [indicator, ...]
        """
        return {
            "CONTROL": [
                # Question words
                "what", "qué", "which", "cuál", "how", "cómo", "where", "dónde",
                # Imperatives
                "need", "necesito", "want", "quiero", "should", "debe", "must", "hay que",
            ],

            "CONTEXT": [
                # Temporal
                "before", "antes", "after", "después", "during", "durante",
                "while", "mientras", "previously", "anteriormente", "prior", "previamente",
                # Contextual connectors
                "furthermore", "además", "also", "también", "likewise", "asimismo",
            ],

            "DATA": [
                # Data indicators
                "data", "datos", "information", "información", "results", "resultados",
                "metrics", "métricas", "statistics", "estadísticas", "numbers", "números",
                # Process indicators
                "then", "entonces", "therefore", "por lo tanto", "thus", "así",
            ]
        }

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify text into CONTROL, CONTEXT, DATA, or unknown.

        Args:
            text: Text to classify

        Returns:
            ClassificationResult with function, confidence, evidence, and scores
        """
        if not text or not text.strip():
            return ClassificationResult(
                function="unknown",
                confidence=0.0,
                evidence=["Empty text"],
                scores={"CONTROL": 0.0, "CONTEXT": 0.0, "DATA": 0.0}
            )

        # Normalize text
        normalized = self._normalize_text(text)

        # Analyze each function
        function_scores = {}
        function_evidence = {}

        for function in ["CONTROL", "CONTEXT", "DATA"]:
            score, evidence = self._analyze_function(normalized, function)
            function_scores[function] = score
            function_evidence[function] = evidence

        # Determine best match
        best_function = max(function_scores, key=function_scores.get)
        best_score = function_scores[best_function]

        # Check if confidence meets minimum threshold
        if best_score < self.MIN_CONFIDENCE:
            return ClassificationResult(
                function="unknown",
                confidence=best_score,
                evidence=[f"All scores below threshold ({self.MIN_CONFIDENCE})"] +
                         function_evidence.get(best_function, [])[:3],
                scores=function_scores
            )

        return ClassificationResult(
            function=best_function,
            confidence=best_score,
            evidence=function_evidence[best_function],
            scores=function_scores
        )

    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching"""
        # Lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _analyze_function(self, text: str, function: str) -> Tuple[float, List[str]]:
        """
        Analyze text for a specific CXD function.

        Args:
            text: Normalized text
            function: "CONTROL", "CONTEXT", or "DATA"

        Returns:
            (score, evidence)
        """
        total_score = 0.0
        evidence = []
        match_count = 0

        # 1. Pattern matching (highest weight)
        patterns = self.patterns.get(function, [])
        for pattern, confidence, category in patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if matches:
                # Each pattern match adds to score
                pattern_score = confidence
                total_score += pattern_score
                match_count += 1

                # Add first match as evidence
                match_text = matches[0].group()[:50]
                evidence.append(f"Pattern ({category}): '{match_text}...'")

        # 2. Keyword matching (medium weight - 70% of keyword confidence)
        keywords = self.keywords.get(function, {})
        for keyword, keyword_conf in keywords.items():
            if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                keyword_score = keyword_conf * 0.7  # Reduce keyword weight
                total_score += keyword_score
                match_count += 1
                evidence.append(f"Keyword: '{keyword}'")

        # 3. Indicator matching (lowest weight: 0.3)
        indicators = self.indicators.get(function, [])
        for indicator in indicators:
            if re.search(r'\b' + re.escape(indicator) + r'\b', text, re.IGNORECASE):
                total_score += 0.3
                match_count += 1
                evidence.append(f"Indicator: '{indicator}'")

        # 4. Structural analysis (bonus)
        structural_bonus, structural_evidence = self._structural_analysis(text, function)
        if structural_bonus > 0:
            total_score += structural_bonus
            evidence.extend(structural_evidence)

        # 5. Normalize score with diminishing returns
        if match_count > 0:
            # Average with diminishing returns for many matches
            normalized_score = total_score / (1 + match_count * 0.1)
            # Cap at 95% to avoid overconfidence
            normalized_score = min(normalized_score, 0.95)
        else:
            normalized_score = 0.0

        return normalized_score, evidence

    def _structural_analysis(self, text: str, function: str) -> Tuple[float, List[str]]:
        """
        Analyze text structure for additional signals.

        Args:
            text: Normalized text
            function: CXD function

        Returns:
            (bonus_score, evidence)
        """
        bonus = 0.0
        evidence = []

        # Question marks → CONTROL (search/decision)
        if function == "CONTROL" and '?' in text:
            bonus += 0.2
            evidence.append("Structure: Contains question")

        # "because", "why" → CONTEXT
        if function == "CONTEXT":
            if re.search(r'\b(because|why|debido|porque|razón)\b', text, re.IGNORECASE):
                bonus += 0.2
                evidence.append("Structure: Explanatory language")

        # Numbers/measurements → DATA
        if function == "DATA":
            # Check for numbers
            if re.search(r'\b\d+\b', text):
                bonus += 0.15
                evidence.append("Structure: Contains numbers")
            # Check for code/technical symbols
            if re.search(r'[{}()\[\]<>=]', text):
                bonus += 0.15
                evidence.append("Structure: Technical/code-like")

        return bonus, evidence


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def classify_cxd(text: str) -> Tuple[str, float]:
    """
    Convenience function for quick classification.

    Args:
        text: Text to classify

    Returns:
        (function, confidence) where function is "CONTROL", "CONTEXT", "DATA", or "unknown"
    """
    classifier = EnhancedCXDClassifier()
    result = classifier.classify(text)
    return result.function, result.confidence


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "EnhancedCXDClassifier",
    "ClassificationResult",
    "CXDFunction",
    "classify_cxd",
]
