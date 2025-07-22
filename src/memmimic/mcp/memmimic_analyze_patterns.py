#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemMimic - Memory Pattern Analysis Tool
Analyze patterns in memory usage and content evolution
Part of the MemMimic cognitive memory system
"""

import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

# Ensure UTF-8 output
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add MemMimic to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
memmimic_src = os.path.join(current_dir, "..", "..")
sys.path.insert(0, memmimic_src)


def get_memory_store():
    """Get the MemMimic memory store instance"""
    try:
        from memmimic.memory.storage import AMMSStorage

        # Use MemMimic memory database - standardized path
        db_path = os.path.join(memmimic_src, "..", "memmimic.db")
        if not os.path.exists(db_path):
            # Fallback to legacy path
            db_path = os.path.join(
                memmimic_src, "..", "..", "clay", "claude_mcp_enhanced_memories.db"
            )

        return AMMSStorage(db_path)
    except Exception as e:
        print(f"❌ Error accessing memory store: {e}", file=sys.stderr)
        return None


def init_cxd_classifier():
    """Initialize CXD classifier for pattern analysis"""
    try:
        from memmimic.cxd.classifiers.optimized_meta import create_optimized_classifier
        from memmimic.cxd.core.config import CXDConfig

        config = CXDConfig()
        classifier = create_optimized_classifier(config=config)
        return classifier
    except Exception as e:
        print(f"⚠️ CXD classifier unavailable: {e}", file=sys.stderr)
        return None


def classify_memory_batch(classifier, memories):
    """Classify multiple memories for pattern analysis"""
    if not classifier:
        return {}

    classifications = {}
    for memory in memories:
        try:
            result = classifier.classify_detailed(text=memory.content)
            classifications[memory.id] = {
                "function": (
                    result.final_sequence.dominant_function.value
                    if result.final_sequence.dominant_function
                    else "UNKNOWN"
                ),
                "confidence": result.final_confidence or 0.0,
                "pattern": (
                    result.final_sequence.pattern
                    if result.final_sequence
                    else "UNKNOWN"
                ),
                "concordance": getattr(result, "concordance_score", 0.0),
            }
        except Exception as e:
            classifications[memory.id] = {
                "function": "UNKNOWN",
                "confidence": 0.0,
                "pattern": "UNKNOWN",
                "concordance": 0.0,
            }

    return classifications


def analyze_temporal_patterns(memories, classifications):
    """Analyze how memory patterns change over time"""
    time_buckets = defaultdict(list)

    for memory in memories:
        try:
            # Parse creation time
            created = datetime.fromisoformat(memory.created_at.replace("Z", "+00:00"))

            # Group by time periods
            now = datetime.now().replace(tzinfo=created.tzinfo)
            hours_ago = (now - created).total_seconds() / 3600

            if hours_ago <= 1:
                bucket = "last_1h"
            elif hours_ago <= 6:
                bucket = "last_6h"
            elif hours_ago <= 24:
                bucket = "last_24h"
            elif hours_ago <= 24 * 7:
                bucket = f"last_{int(hours_ago / 24)}d"
            else:
                bucket = "older"

            memory_data = {
                "memory": memory,
                "classification": classifications.get(
                    memory.id, {"function": "UNKNOWN"}
                ),
            }
            time_buckets[bucket].append(memory_data)

        except Exception as e:
            continue

    return time_buckets


def analyze_content_patterns(memories, classifications):
    """Analyze content patterns and cognitive themes"""
    patterns = {
        "functions": Counter(),
        "types": Counter(),
        "confidence_distribution": [],
        "content_themes": defaultdict(list),
        "complexity_scores": [],
        "cognitive_evolution": [],
    }

    for memory in memories:
        # Memory type distribution
        patterns["types"][memory.memory_type] += 1

        # CXD function distribution
        classification = classifications.get(memory.id, {"function": "UNKNOWN"})
        patterns["functions"][classification["function"]] += 1

        # Confidence and cognitive quality
        confidence = classification.get("confidence", 0.0)
        patterns["confidence_distribution"].append(confidence)

        # Content complexity analysis
        content_length = len(memory.content)
        patterns["complexity_scores"].append(content_length)

        # Enhanced theme extraction
        content_lower = memory.content.lower()
        themes = []

        # MemMimic-specific themes
        if any(
            word in content_lower for word in ["memmimic", "contextual memory", "cmi"]
        ):
            themes.append("memmimic_core")
        if any(
            word in content_lower
            for word in ["cxd", "classifier", "cognitive function"]
        ):
            themes.append("cxd_system")
        if any(word in content_lower for word in ["tale", "narrative", "story"]):
            themes.append("tales_system")
        if any(
            word in content_lower
            for word in ["memory", "memoria", "remember", "recall"]
        ):
            themes.append("memory_operations")
        if any(word in content_lower for word in ["pattern", "analysis", "insight"]):
            themes.append("pattern_analysis")
        if any(
            word in content_lower for word in ["refactor", "architecture", "structure"]
        ):
            themes.append("system_evolution")
        if any(
            word in content_lower for word in ["collaboration", "sprooket", "claude"]
        ):
            themes.append("collaboration")
        if any(word in content_lower for word in ["error", "bug", "fix", "debug"]):
            themes.append("troubleshooting")
        if any(word in content_lower for word in ["test", "pytest", "validation"]):
            themes.append("testing")

        if not themes:
            themes.append("general")

        for theme in themes:
            patterns["content_themes"][theme].append(memory.id)

        # Track cognitive evolution
        patterns["cognitive_evolution"].append(
            {
                "id": memory.id,
                "function": classification["function"],
                "confidence": confidence,
                "complexity": content_length,
                "timestamp": memory.created_at,
            }
        )

    return patterns


def analyze_usage_patterns(memories):
    """Analyze memory access and usage patterns"""
    patterns = {
        "access_frequency": defaultdict(int),
        "memory_age_distribution": [],
        "type_evolution": defaultdict(list),
        "usage_efficiency": {},
    }

    now = datetime.now()

    for memory in memories:
        try:
            created = datetime.fromisoformat(memory.created_at.replace("Z", "+00:00"))
            age_hours = (
                now.replace(tzinfo=created.tzinfo) - created
            ).total_seconds() / 3600

            patterns["memory_age_distribution"].append(age_hours)
            patterns["access_frequency"][memory.access_count] += 1
            patterns["type_evolution"][memory.memory_type].append(age_hours)

        except Exception as e:
            continue

    # Calculate usage efficiency metrics
    if patterns["access_frequency"]:
        total_accesses = sum(
            count * freq for count, freq in patterns["access_frequency"].items()
        )
        total_memories = len(memories)
        patterns["usage_efficiency"] = {
            "avg_access_per_memory": (
                total_accesses / total_memories if total_memories > 0 else 0
            ),
            "memory_utilization": (
                len([c for c in patterns["access_frequency"].keys() if c > 0])
                / total_memories
                if total_memories > 0
                else 0
            ),
        }

    return patterns


def generate_insights(
    memories, classifications, temporal_patterns, content_patterns, usage_patterns
):
    """Generate actionable insights from pattern analysis"""
    insights = []
    total_memories = len(memories)

    # Cognitive function insights
    func_dist = content_patterns["functions"]
    if func_dist:
        dominant_function = func_dist.most_common(1)[0]
        insights.append(
            f"🧠 Función cognitiva dominante: {dominant_function[0]} ({dominant_function[1]}/{total_memories} memorias)"
        )

        # Cognitive balance analysis
        if len(func_dist) >= 3:
            control = func_dist.get("CONTROL", 0)
            context = func_dist.get("CONTEXT", 0)
            data = func_dist.get("DATA", 0)

            if max(control, context, data) > total_memories * 0.6:
                insights.append(
                    "⚖️ Desequilibrio cognitivo detectado - considera diversificar funciones"
                )
            else:
                insights.append(
                    "✅ Balance cognitivo saludable entre Control/Context/Data"
                )

    # Temporal activity insights
    recent_memories = sum(
        len(bucket)
        for bucket_name, bucket in temporal_patterns.items()
        if any(period in bucket_name for period in ["1h", "6h", "24h"])
    )
    if recent_memories > 0:
        recent_percentage = (recent_memories / total_memories) * 100
        insights.append(
            f"⏰ Actividad reciente: {recent_memories} memorias ({recent_percentage:.1f}%) en últimas 24h"
        )

    # Content quality insights
    if content_patterns["confidence_distribution"]:
        avg_confidence = sum(content_patterns["confidence_distribution"]) / len(
            content_patterns["confidence_distribution"]
        )
        high_confidence_count = sum(
            1 for c in content_patterns["confidence_distribution"] if c > 0.7
        )
        insights.append(
            f"🎯 Confianza promedio: {avg_confidence:.2f} ({high_confidence_count} alta confianza)"
        )

    # Complexity insights
    if content_patterns["complexity_scores"]:
        avg_length = sum(content_patterns["complexity_scores"]) / len(
            content_patterns["complexity_scores"]
        )
        insights.append(
            f"📏 Complejidad promedio: {avg_length:.0f} caracteres por memoria"
        )

    # Theme insights
    themes = content_patterns["content_themes"]
    if themes:
        top_theme = max(themes.items(), key=lambda x: len(x[1]))
        insights.append(
            f"🏷️ Tema dominante: {top_theme[0]} ({len(top_theme[1])} memorias)"
        )

    # Usage efficiency insights
    efficiency = usage_patterns.get("usage_efficiency", {})
    if efficiency:
        avg_access = efficiency.get("avg_access_per_memory", 0)
        utilization = efficiency.get("memory_utilization", 0) * 100
        insights.append(
            f"📊 Eficiencia: {avg_access:.1f} accesos/memoria, {utilization:.1f}% utilización"
        )

    return insights


def generate_recommendations(
    insights, content_patterns, temporal_patterns, usage_patterns
):
    """Generate specific recommendations based on patterns"""
    recommendations = []

    # Function balance recommendations
    func_dist = content_patterns["functions"]
    if func_dist:
        total = sum(func_dist.values())
        control_pct = func_dist.get("CONTROL", 0) / total * 100
        context_pct = func_dist.get("CONTEXT", 0) / total * 100
        data_pct = func_dist.get("DATA", 0) / total * 100

        if control_pct > 50:
            recommendations.append(
                "🎛️ Alto Control - considera más actividades de análisis (DATA) y contextualización"
            )
        elif data_pct > 50:
            recommendations.append(
                "📊 Alto Data - balancear con más búsqueda (CONTROL) y referencias contextuales"
            )
        elif context_pct > 50:
            recommendations.append(
                "🔗 Alto Context - añadir más procesamiento directo y gestión de información"
            )

    # Usage recommendations
    efficiency = usage_patterns.get("usage_efficiency", {})
    if efficiency:
        utilization = efficiency.get("memory_utilization", 0)
        if utilization < 0.3:
            recommendations.append(
                "💤 Baja utilización - revisar y activar memorias infrautilizadas"
            )
        elif utilization > 0.8:
            recommendations.append(
                "🔥 Alta utilización - excelente aprovechamiento del sistema"
            )

    # Quality recommendations
    if content_patterns["confidence_distribution"]:
        low_conf_count = sum(
            1 for c in content_patterns["confidence_distribution"] if c < 0.5
        )
        if low_conf_count > len(content_patterns["confidence_distribution"]) * 0.3:
            recommendations.append(
                "🔍 Muchas memorias baja confianza - revisar calidad de contenido"
            )

    # Temporal recommendations
    recent_activity = sum(
        len(bucket)
        for name, bucket in temporal_patterns.items()
        if any(period in name for period in ["1h", "6h", "24h"])
    )
    total_memories = sum(len(bucket) for bucket in temporal_patterns.values())

    if recent_activity < total_memories * 0.2:
        recommendations.append(
            "📈 Baja actividad reciente - considerar usar memoria más frecuentemente"
        )

    return recommendations


def format_analysis_output(
    memories,
    classifications,
    temporal_patterns,
    content_patterns,
    usage_patterns,
    insights,
    recommendations,
):
    """Format the complete pattern analysis output"""
    lines = []

    # Header
    lines.append("🧠 MEMMIMIC - ANÁLISIS DE PATRONES DE MEMORIA")
    lines.append("=" * 60)
    lines.append(f"📊 Dataset: {len(memories)} memorias analizadas")
    lines.append(
        f"🔬 Clasificador CXD: {'✅ Activo' if classifications else '❌ No disponible'}"
    )
    lines.append("")

    # Key insights
    lines.append("💡 INSIGHTS PRINCIPALES:")
    for insight in insights:
        lines.append(f"   {insight}")
    lines.append("")

    # Function distribution with visual bars
    lines.append("🧠 DISTRIBUCIÓN COGNITIVA (CXD):")
    func_dist = content_patterns["functions"]
    if func_dist:
        total = sum(func_dist.values())
        for func, count in func_dist.most_common():
            percentage = (count / total) * 100 if total > 0 else 0
            bar_length = int(percentage / 5)
            bar = "█" * bar_length + "░" * (20 - bar_length)

            func_icons = {
                "CONTROL": "🎛️",
                "CONTEXT": "🔗",
                "DATA": "📊",
                "UNKNOWN": "❓",
            }
            icon = func_icons.get(func, "🔹")
            lines.append(f"   {icon} {func:<8} [{bar}] {count:3d} ({percentage:5.1f}%)")
    lines.append("")

    # Memory types
    lines.append("📂 TIPOS DE MEMORIA:")
    type_dist = content_patterns["types"]
    for memory_type, count in type_dist.most_common():
        percentage = (count / len(memories)) * 100 if len(memories) > 0 else 0
        lines.append(f"   📝 {memory_type:<15} {count:3d} ({percentage:5.1f}%)")
    lines.append("")

    # Temporal patterns
    lines.append("⏰ ACTIVIDAD TEMPORAL:")
    sorted_buckets = sorted(
        temporal_patterns.keys(),
        key=lambda x: (
            0 if "h" in x else 1 if "d" in x else 2,
            (
                int(x.split("_")[1].replace("h", "").replace("d", ""))
                if x != "older"
                else 999
            ),
        ),
    )

    for bucket in sorted_buckets:
        bucket_memories = temporal_patterns[bucket]
        if bucket_memories:
            bucket_functions = Counter(
                item["classification"]["function"] for item in bucket_memories
            )
            lines.append(f"   ⏱️ {bucket:<10} {len(bucket_memories):3d} memorias")
    lines.append("")

    # Content themes
    lines.append("🏷️ TEMAS DE CONTENIDO:")
    themes = content_patterns["content_themes"]
    theme_icons = {
        "memmimic_core": "🧠",
        "cxd_system": "🎯",
        "tales_system": "📖",
        "memory_operations": "💾",
        "pattern_analysis": "📊",
        "system_evolution": "🔄",
        "collaboration": "🤝",
        "troubleshooting": "🔧",
        "testing": "🧪",
        "general": "📋",
    }

    for theme, memory_ids in sorted(
        themes.items(), key=lambda x: len(x[1]), reverse=True
    ):
        count = len(memory_ids)
        percentage = (count / len(memories)) * 100 if len(memories) > 0 else 0
        icon = theme_icons.get(theme, "🔹")
        lines.append(f"   {icon} {theme:<20} {count:3d} ({percentage:5.1f}%)")
    lines.append("")

    # Quality metrics
    lines.append("🎯 MÉTRICAS DE CALIDAD:")
    if content_patterns["confidence_distribution"]:
        confidences = content_patterns["confidence_distribution"]
        avg_conf = sum(confidences) / len(confidences)
        high_conf_count = sum(1 for c in confidences if c > 0.7)
        medium_conf_count = sum(1 for c in confidences if 0.5 <= c <= 0.7)
        low_conf_count = sum(1 for c in confidences if c < 0.5)

        lines.append(f"   📈 Confianza promedio:    {avg_conf:.3f}")
        lines.append(f"   🟢 Alta confianza (>0.7): {high_conf_count} memorias")
        lines.append(f"   🟡 Media confianza:       {medium_conf_count} memorias")
        lines.append(f"   🔴 Baja confianza (<0.5): {low_conf_count} memorias")

    efficiency = usage_patterns.get("usage_efficiency", {})
    if efficiency:
        avg_access = efficiency.get("avg_access_per_memory", 0)
        utilization = efficiency.get("memory_utilization", 0) * 100
        lines.append(f"   🔄 Accesos promedio:      {avg_access:.1f} por memoria")
        lines.append(f"   📊 Utilización:           {utilization:.1f}%")
    lines.append("")

    # Recommendations
    if recommendations:
        lines.append("🎯 RECOMENDACIONES:")
        for rec in recommendations:
            lines.append(f"   {rec}")
        lines.append("")

    lines.append("✅ Análisis de patrones completado")

    return "\n".join(lines)


def main():
    try:
        # Initialize memory store
        memory_store = get_memory_store()
        if not memory_store:
            print("❌ No se pudo acceder al sistema de memoria")
            sys.exit(1)

        # Get memories for analysis
        all_memories = memory_store.get_all()
        if not all_memories:
            print("📭 No hay memorias para analizar")
            sys.exit(0)

        # Limit for performance (analyze last 100 memories)
        memories = all_memories[-100:] if len(all_memories) > 100 else all_memories

        # Initialize CXD classifier
        cxd_classifier = init_cxd_classifier()

        # Classify memories
        classifications = {}
        if cxd_classifier:
            classifications = classify_memory_batch(cxd_classifier, memories)

        # Perform pattern analysis
        temporal_patterns = analyze_temporal_patterns(memories, classifications)
        content_patterns = analyze_content_patterns(memories, classifications)
        usage_patterns = analyze_usage_patterns(memories)

        # Generate insights and recommendations
        insights = generate_insights(
            memories,
            classifications,
            temporal_patterns,
            content_patterns,
            usage_patterns,
        )
        recommendations = generate_recommendations(
            insights, content_patterns, temporal_patterns, usage_patterns
        )

        # Format and output results
        output = format_analysis_output(
            memories,
            classifications,
            temporal_patterns,
            content_patterns,
            usage_patterns,
            insights,
            recommendations,
        )

        print(output)

    except Exception as e:
        print(f"❌ Error en análisis de patrones: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

