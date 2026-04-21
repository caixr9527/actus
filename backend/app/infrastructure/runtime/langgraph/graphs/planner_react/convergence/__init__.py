#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 解耦：收敛判定模块。"""

from .judge import ConvergenceJudge, ConvergenceEvaluationResult
from .general_convergence import GeneralConvergenceJudge, GeneralConvergenceResult
from .research_convergence import ResearchConvergenceJudge, ResearchConvergenceResult
from .web_reading_convergence import WebReadingConvergenceJudge, WebReadingConvergenceResult

__all__ = [
    "ConvergenceJudge",
    "ConvergenceEvaluationResult",
    "GeneralConvergenceJudge",
    "GeneralConvergenceResult",
    "ResearchConvergenceJudge",
    "ResearchConvergenceResult",
    "WebReadingConvergenceJudge",
    "WebReadingConvergenceResult",
]
