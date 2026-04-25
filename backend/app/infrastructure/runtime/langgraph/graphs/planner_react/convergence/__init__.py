#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 解耦：收敛判定模块。"""

from .contracts import (
    ConvergenceDecision,
    IterationConvergenceContext,
    MaxIterationConvergenceContext,
)
from .judge import ConvergenceJudge
from .general_convergence import GeneralConvergenceJudge
from .research_convergence import ResearchConvergenceJudge
from .web_reading_convergence import WebReadingConvergenceJudge

__all__ = [
    "ConvergenceDecision",
    "IterationConvergenceContext",
    "MaxIterationConvergenceContext",
    "ConvergenceJudge",
    "GeneralConvergenceJudge",
    "ResearchConvergenceJudge",
    "WebReadingConvergenceJudge",
]
