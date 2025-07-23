"""
Enhanced Triggers - Biological Reflex Implementation

Transforms traditional MCP tools into biological nervous system triggers
with internal intelligence processing and <5ms response times.

Core Triggers:
- NervousSystemRemember: Enhanced memory storage with internal quality gates
- NervousSystemRecall: Intelligent search with relationship mapping  
- NervousSystemThink: Contextual processing with Socratic guidance
- NervousSystemAnalyze: Pattern analysis with predictive insights

All triggers maintain 100% external compatibility while adding internal intelligence.
"""

from .remember import NervousSystemRemember
from .recall import NervousSystemRecall
from .think import NervousSystemThink
from .analyze import NervousSystemAnalyze

__all__ = [
    'NervousSystemRemember',
    'NervousSystemRecall', 
    'NervousSystemThink',
    'NervousSystemAnalyze'
]

__version__ = "2.0.0"