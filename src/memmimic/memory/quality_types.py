#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemMimic Quality Types - Shared types for memory quality control system
"""

from typing import List
from datetime import datetime

from .storage.amms_storage import Memory


class MemoryQualityResult:
    """Result of memory quality assessment"""
    
    def __init__(
        self,
        approved: bool,
        reason: str,
        confidence: float,
        duplicates: List[Memory] = None,
        suggested_content: str = None,
        auto_decision: bool = False
    ):
        self.approved = approved
        self.reason = reason
        self.confidence = confidence
        self.duplicates = duplicates or []
        self.suggested_content = suggested_content
        self.auto_decision = auto_decision
        self.timestamp = datetime.now()