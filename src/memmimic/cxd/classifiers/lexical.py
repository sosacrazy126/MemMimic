"""
Lexical CXD Classifier implementation.

This module implements the lexical CXD classifier that uses pattern matching
and keyword analysis to identify cognitive functions in text.
"""

import re
import time
from typing import Dict, List, Set, Optional, Any
import logging

from ..core.interfaces import CXDClassifier
from ..core.types import CXDFunction, ExecutionState, CXDTag, CXDSequence
from ..core.config import CXDConfig

logger = logging.getLogger(__name__)


class LexicalCXDClassifier(CXDClassifier):
    """
    Lexical classifier for CXD functions using pattern matching.
    
    Analyzes text using predefined patterns, keywords, and linguistic markers
    to identify Control, Context, and Data cognitive functions.
    """
    
    def __init__(self, config: Optional[CXDConfig] = None):
        """
        Initialize lexical classifier.
        
        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or CXDConfig()
        
        # Patterns and keywords for each function
        self.patterns = self._build_patterns()
        self.keywords = self._build_keywords()
        self.indicators = self._build_indicators()
        
        # Configuration parameters
        self.min_confidence = self.config.algorithms.thresholds.confidence_min
        self.high_confidence = self.config.algorithms.thresholds.confidence_high
        
        # Statistics
        self.stats = {
            "total_classifications": 0,
            "processing_times": [],
            "function_detections": {func: 0 for func in CXDFunction},
            "pattern_matches": {},
            "keyword_matches": {}
        }
        
        logger.info("Initialized LexicalCXDClassifier with pattern-based analysis")
    
    def _build_patterns(self) -> Dict[CXDFunction, List[Dict[str, Any]]]:
        """
        Build regex patterns for each CXD function.
        
        Returns:
            Dict: Patterns organized by function with confidence weights
        """
        return {
            CXDFunction.CONTROL: [
                # Search and retrieval patterns
                {
                    "pattern": r"\b(buscar|encontrar|localizar|rastrear|explorar)\b.*\b(información|datos|elementos|documentos|resultados)\b",
                    "confidence": 0.9,
                    "category": "search",
                    "description": "Search for information patterns"
                },
                {
                    "pattern": r"\b(search|find|locate|look\s+for|retrieve)\b.*\b(information|data|documents|results)\b",
                    "confidence": 0.9,
                    "category": "search",
                    "description": "English search patterns"
                },
                
                # Filter and selection patterns
                {
                    "pattern": r"\b(filtrar|seleccionar|cribar|excluir|depurar)\b.*\b(resultados|información|datos|opciones)\b",
                    "confidence": 0.85,
                    "category": "filter",
                    "description": "Filter and selection patterns"
                },
                {
                    "pattern": r"\b(filter|select|exclude|narrow\s+down|refine)\b.*\b(results|data|options|choices)\b",
                    "confidence": 0.85,
                    "category": "filter", 
                    "description": "English filter patterns"
                },
                
                # Control and management patterns
                {
                    "pattern": r"\b(controlar|dirigir|gestionar|coordinar|supervisar)\b.*\b(proceso|flujo|sistema|recursos)\b",
                    "confidence": 0.8,
                    "category": "control",
                    "description": "Control and management patterns"
                },
                {
                    "pattern": r"\b(control|direct|manage|coordinate|supervise)\b.*\b(process|flow|system|resources)\b",
                    "confidence": 0.8,
                    "category": "control",
                    "description": "English control patterns"
                },
                
                # Decision patterns
                {
                    "pattern": r"\b(decidir|determinar|resolver|elegir|establecer)\b.*\b(qué|cuál|cómo|estrategia|acción)\b",
                    "confidence": 0.85,
                    "category": "decision",
                    "description": "Decision making patterns"
                },
                {
                    "pattern": r"\b(decide|determine|resolve|choose|establish)\b.*\b(what|which|how|strategy|action)\b",
                    "confidence": 0.85,
                    "category": "decision",
                    "description": "English decision patterns"
                }
            ],
            
            CXDFunction.CONTEXT: [
                # Relationship patterns
                {
                    "pattern": r"\b(relaciona|conecta|vincula|asocia)\b.*\b(con|anterior|previo|experiencias|trabajo)\b",
                    "confidence": 0.9,
                    "category": "relation",
                    "description": "Relationship establishment patterns"
                },
                {
                    "pattern": r"\b(relates?|connects?|links?|associates?)\b.*\b(to|with|previous|prior|earlier|work)\b",
                    "confidence": 0.9,
                    "category": "relation",
                    "description": "English relationship patterns"
                },
                
                # Reference patterns
                {
                    "pattern": r"\b(referencia|menciona|alude|cita)\b.*\b(anterior|previo|discusión|conversación|sesión)\b",
                    "confidence": 0.85,
                    "category": "reference",
                    "description": "Reference to previous content"
                },
                {
                    "pattern": r"\b(reference|mention|refer\s+to|cite)\b.*\b(previous|prior|earlier|discussion|conversation)\b",
                    "confidence": 0.85,
                    "category": "reference",
                    "description": "English reference patterns"
                },
                
                # Contextual situating patterns
                {
                    "pattern": r"\b(situar|contextualizar|enmarcar|ubicar)\b.*\b(marco|contexto|situación|ambiente)\b",
                    "confidence": 0.8,
                    "category": "contextualization",
                    "description": "Contextual situating patterns"
                },
                {
                    "pattern": r"\b(situate|contextualize|frame|place)\b.*\b(context|framework|situation|environment)\b",
                    "confidence": 0.8,
                    "category": "contextualization",
                    "description": "English contextualization patterns"
                },
                
                # Historical/memory patterns
                {
                    "pattern": r"\b(recordar|retomar|continuar|basándome)\b.*\b(conversación|discusión|tema|hilo)\b",
                    "confidence": 0.85,
                    "category": "memory",
                    "description": "Historical reference patterns"
                },
                {
                    "pattern": r"\b(remember|recall|continue|based\s+on)\b.*\b(conversation|discussion|topic|thread)\b",
                    "confidence": 0.85,
                    "category": "memory",
                    "description": "English memory patterns"
                }
            ],
            
            CXDFunction.DATA: [
                # Processing patterns
                {
                    "pattern": r"\b(procesar|analizar|examinar|computar|calcular)\b.*\b(información|datos|resultados|patrones)\b",
                    "confidence": 0.9,
                    "category": "processing",
                    "description": "Data processing patterns"
                },
                {
                    "pattern": r"\b(process|analyze|examine|compute|calculate)\b.*\b(information|data|results|patterns)\b",
                    "confidence": 0.9,
                    "category": "processing",
                    "description": "English processing patterns"
                },
                
                # Transformation patterns
                {
                    "pattern": r"\b(transformar|convertir|modificar|adaptar|reformatear)\b.*\b(datos|formato|estructura)\b",
                    "confidence": 0.85,
                    "category": "transformation",
                    "description": "Data transformation patterns"
                },
                {
                    "pattern": r"\b(transform|convert|modify|adapt|reformat)\b.*\b(data|format|structure)\b",
                    "confidence": 0.85,
                    "category": "transformation",
                    "description": "English transformation patterns"
                },
                
                # Generation patterns
                {
                    "pattern": r"\b(generar|crear|producir|elaborar|construir)\b.*\b(síntesis|resumen|reporte|visualización)\b",
                    "confidence": 0.8,
                    "category": "generation",
                    "description": "Content generation patterns"
                },
                {
                    "pattern": r"\b(generate|create|produce|elaborate|build)\b.*\b(synthesis|summary|report|visualization)\b",
                    "confidence": 0.8,
                    "category": "generation",
                    "description": "English generation patterns"
                },
                
                # Extraction patterns
                {
                    "pattern": r"\b(extraer|derivar|obtener|conseguir|deducir)\b.*\b(insights|conclusiones|patrones|tendencias)\b",
                    "confidence": 0.85,
                    "category": "extraction",
                    "description": "Information extraction patterns"
                },
                {
                    "pattern": r"\b(extract|derive|obtain|get|deduce)\b.*\b(insights|conclusions|patterns|trends)\b",
                    "confidence": 0.85,
                    "category": "extraction",
                    "description": "English extraction patterns"
                },
                
                # Organization patterns
                {
                    "pattern": r"\b(organizar|estructurar|clasificar|categorizar|ordenar)\b.*\b(información|datos|contenido)\b",
                    "confidence": 0.8,
                    "category": "organization",
                    "description": "Data organization patterns"
                },
                {
                    "pattern": r"\b(organize|structure|classify|categorize|order)\b.*\b(information|data|content)\b",
                    "confidence": 0.8,
                    "category": "organization",
                    "description": "English organization patterns"
                }
            ]
        }
    
    def _build_keywords(self) -> Dict[CXDFunction, Dict[str, float]]:
        """
        Build keyword dictionaries with confidence weights.
        
        Returns:
            Dict: Keywords organized by function with confidence weights
        """
        return {
            CXDFunction.CONTROL: {
                # Search keywords
                "buscar": 0.8, "search": 0.8, "encontrar": 0.8, "find": 0.8,
                "localizar": 0.7, "locate": 0.7, "rastrear": 0.7, "track": 0.7,
                
                # Filter keywords
                "filtrar": 0.8, "filter": 0.8, "seleccionar": 0.8, "select": 0.8,
                "cribar": 0.7, "screen": 0.7, "excluir": 0.7, "exclude": 0.7,
                
                # Control keywords
                "controlar": 0.8, "control": 0.8, "dirigir": 0.7, "direct": 0.7,
                "gestionar": 0.8, "manage": 0.8, "coordinar": 0.7, "coordinate": 0.7,
                
                # Decision keywords
                "decidir": 0.8, "decide": 0.8, "determinar": 0.8, "determine": 0.8,
                "resolver": 0.7, "resolve": 0.7, "elegir": 0.8, "choose": 0.8
            },
            
            CXDFunction.CONTEXT: {
                # Relation keywords
                "relacionar": 0.8, "relate": 0.8, "conectar": 0.8, "connect": 0.8,
                "vincular": 0.7, "link": 0.7, "asociar": 0.8, "associate": 0.8,
                
                # Reference keywords
                "referenciar": 0.8, "reference": 0.8, "mencionar": 0.7, "mention": 0.7,
                "citar": 0.7, "cite": 0.7, "aludir": 0.6, "allude": 0.6,
                
                # Context keywords
                "contexto": 0.8, "context": 0.8, "situación": 0.7, "situation": 0.7,
                "marco": 0.7, "framework": 0.7, "ambiente": 0.6, "environment": 0.6,
                
                # Memory keywords
                "recordar": 0.8, "remember": 0.8, "anterior": 0.7, "previous": 0.7,
                "previo": 0.7, "prior": 0.7, "conversación": 0.8, "conversation": 0.8
            },
            
            CXDFunction.DATA: {
                # Processing keywords
                "procesar": 0.8, "process": 0.8, "analizar": 0.8, "analyze": 0.8,
                "examinar": 0.7, "examine": 0.7, "computar": 0.8, "compute": 0.8,
                
                # Transformation keywords
                "transformar": 0.8, "transform": 0.8, "convertir": 0.8, "convert": 0.8,
                "modificar": 0.7, "modify": 0.7, "adaptar": 0.7, "adapt": 0.7,
                
                # Generation keywords
                "generar": 0.8, "generate": 0.8, "crear": 0.7, "create": 0.7,
                "producir": 0.7, "produce": 0.7, "elaborar": 0.7, "elaborate": 0.7,
                
                # Extraction keywords
                "extraer": 0.8, "extract": 0.8, "derivar": 0.7, "derive": 0.7,
                "obtener": 0.6, "obtain": 0.6, "deducir": 0.7, "deduce": 0.7,
                
                # Organization keywords
                "organizar": 0.8, "organize": 0.8, "estructurar": 0.8, "structure": 0.8,
                "clasificar": 0.7, "classify": 0.7, "categorizar": 0.7, "categorize": 0.7
            }
        }
    
    def _build_indicators(self) -> Dict[CXDFunction, List[str]]:
        """
        Build linguistic indicators for each function.
        
        Returns:
            Dict: Linguistic indicators by function
        """
        return {
            CXDFunction.CONTROL: [
                # Question words indicating search/decision
                "qué", "what", "cuál", "which", "cómo", "how", "dónde", "where",
                # Imperative indicators
                "necesito", "need", "quiero", "want", "debe", "should", "hay que", "must"
            ],
            
            CXDFunction.CONTEXT: [
                # Temporal references
                "antes", "before", "después", "after", "durante", "during", "mientras", "while",
                "anteriormente", "previously", "previamente", "prior",
                # Contextual connectors
                "además", "furthermore", "también", "also", "asimismo", "likewise"
            ],
            
            CXDFunction.DATA: [
                # Data-related indicators
                "datos", "data", "información", "information", "resultados", "results",
                "métricas", "metrics", "estadísticas", "statistics", "números", "numbers",
                # Process indicators
                "entonces", "then", "por lo tanto", "therefore", "así", "thus"
            ]
        }
    
    def classify(self, text: str) -> CXDSequence:
        """
        Classify text using lexical pattern matching.
        
        Args:
            text: Input text to classify
            
        Returns:
            CXDSequence: Classified CXD sequence
        """
        start_time = time.time()
        
        if not text or not text.strip():
            return CXDSequence([])
        
        # Normalize text
        normalized_text = self._normalize_text(text)
        
        # Analyze each function
        function_scores = {}
        evidence_by_function = {}
        
        for function in CXDFunction:
            score, evidence = self._analyze_function(normalized_text, function)
            if score >= self.min_confidence:
                function_scores[function] = score
                evidence_by_function[function] = evidence
        
        # Create CXD tags
        tags = self._create_tags(function_scores, evidence_by_function)
        
        # Update statistics
        processing_time = (time.time() - start_time) * 1000
        self._update_stats(function_scores, processing_time)
        
        return CXDSequence(tags)
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for pattern matching.
        
        Args:
            text: Input text
            
        Returns:
            str: Normalized text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Optional: Remove punctuation for keyword matching
        # Keep punctuation for pattern matching
        
        return text
    
    def _analyze_function(self, text: str, function: CXDFunction) -> tuple[float, List[str]]:
        """
        Analyze text for a specific CXD function.
        
        Args:
            text: Normalized text
            function: CXD function to analyze
            
        Returns:
            Tuple[float, List[str]]: Confidence score and evidence list
        """
        total_score = 0.0
        evidence = []
        match_count = 0
        
        # Pattern matching
        patterns = self.patterns.get(function, [])
        for pattern_info in patterns:
            pattern = pattern_info["pattern"]
            confidence = pattern_info["confidence"]
            category = pattern_info["category"]
            
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                total_score += confidence
                match_count += 1
                evidence.append(f"Pattern match ({category}): '{match.group()}'")
                
                # Update pattern statistics
                pattern_key = f"{function.value}:{category}"
                self.stats["pattern_matches"][pattern_key] = (
                    self.stats["pattern_matches"].get(pattern_key, 0) + 1
                )
        
        # Keyword matching
        keywords = self.keywords.get(function, {})
        for keyword, keyword_confidence in keywords.items():
            if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                total_score += keyword_confidence * 0.7  # Keywords have lower weight
                match_count += 1
                evidence.append(f"Keyword match: '{keyword}'")
                
                # Update keyword statistics
                keyword_key = f"{function.value}:{keyword}"
                self.stats["keyword_matches"][keyword_key] = (
                    self.stats["keyword_matches"].get(keyword_key, 0) + 1
                )
        
        # Indicator matching (lower weight)
        indicators = self.indicators.get(function, [])
        for indicator in indicators:
            if re.search(r'\b' + re.escape(indicator) + r'\b', text, re.IGNORECASE):
                total_score += 0.3  # Indicators have low weight
                evidence.append(f"Indicator: '{indicator}'")
        
        # Normalize score (prevent over-confidence)
        if match_count > 0:
            # Average score with diminishing returns for multiple matches
            normalized_score = total_score / (1 + match_count * 0.1)
            normalized_score = min(normalized_score, 0.95)  # Cap at 95%
        else:
            normalized_score = 0.0
        
        return normalized_score, evidence
    
    def _create_tags(self, function_scores: Dict[CXDFunction, float], 
                    evidence_by_function: Dict[CXDFunction, List[str]]) -> List[CXDTag]:
        """
        Create CXD tags from function scores.
        
        Args:
            function_scores: Confidence scores by function
            evidence_by_function: Evidence by function
            
        Returns:
            List[CXDTag]: Created tags
        """
        tags = []
        
        for function, confidence in function_scores.items():
            # Determine execution state based on confidence
            if confidence >= self.high_confidence:
                state = ExecutionState.SUCCESS
            elif confidence >= 0.5:
                state = ExecutionState.PARTIAL
            elif confidence >= 0.3:
                state = ExecutionState.UNCERTAIN
            else:
                state = ExecutionState.FAILURE
            
            # Create tag
            tag = CXDTag(
                function=function,
                state=state,
                confidence=confidence,
                evidence=evidence_by_function.get(function, [])
            )
            
            tags.append(tag)
        
        # Sort by confidence (highest first)
        tags.sort(key=lambda t: t.confidence, reverse=True)
        
        return tags
    
    def _update_stats(self, function_scores: Dict[CXDFunction, float], processing_time: float) -> None:
        """Update classification statistics."""
        self.stats["total_classifications"] += 1
        self.stats["processing_times"].append(processing_time)
        
        for function in function_scores:
            self.stats["function_detections"][function] += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the classifier.
        
        Returns:
            Dict: Performance statistics
        """
        stats = self.stats.copy()
        
        if self.stats["processing_times"]:
            times = self.stats["processing_times"]
            stats.update({
                "avg_processing_time_ms": sum(times) / len(times),
                "max_processing_time_ms": max(times),
                "min_processing_time_ms": min(times),
                "total_processing_time_ms": sum(times)
            })
        
        # Add pattern and keyword statistics
        stats["total_pattern_matches"] = sum(self.stats["pattern_matches"].values())
        stats["total_keyword_matches"] = sum(self.stats["keyword_matches"].values())
        stats["unique_patterns_matched"] = len(self.stats["pattern_matches"])
        stats["unique_keywords_matched"] = len(self.stats["keyword_matches"])
        
        # Function detection rates
        total_classifications = self.stats["total_classifications"]
        if total_classifications > 0:
            detection_rates = {}
            for function, count in self.stats["function_detections"].items():
                detection_rates[f"{function.value}_detection_rate"] = count / total_classifications
            stats.update(detection_rates)
        
        return stats
    
    def get_pattern_info(self) -> Dict[str, Any]:
        """
        Get information about patterns and keywords.
        
        Returns:
            Dict: Pattern and keyword information
        """
        info = {
            "total_patterns": sum(len(patterns) for patterns in self.patterns.values()),
            "total_keywords": sum(len(keywords) for keywords in self.keywords.values()),
            "total_indicators": sum(len(indicators) for indicators in self.indicators.values()),
            "patterns_by_function": {
                func.value: len(patterns) for func, patterns in self.patterns.items()
            },
            "keywords_by_function": {
                func.value: len(keywords) for func, keywords in self.keywords.items()
            }
        }
        
        return info
    
    def explain_classification(self, text: str) -> Dict[str, Any]:
        """
        Provide detailed explanation of classification process.
        
        Args:
            text: Input text to explain
            
        Returns:
            Dict: Detailed explanation
        """
        normalized_text = self._normalize_text(text)
        explanation = {
            "input_text": text,
            "normalized_text": normalized_text,
            "function_analysis": {}
        }
        
        for function in CXDFunction:
            score, evidence = self._analyze_function(normalized_text, function)
            explanation["function_analysis"][function.value] = {
                "confidence_score": score,
                "evidence": evidence,
                "meets_threshold": score >= self.min_confidence
            }
        
        return explanation


# Export lexical classifier
__all__ = [
    "LexicalCXDClassifier",
]
