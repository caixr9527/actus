#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 解耦：收敛判定模块。"""

from .contracts import (
    ConvergenceDecision,
    IterationConvergenceContext,
    MaxIterationConvergenceContext,
)
from .judge import (
    FileFactConvergenceJudge,
    GeneralConvergenceJudge,
    ResearchConvergenceJudge,
    WebReadingConvergenceJudge,
)

__all__ = [
    "ConvergenceDecision",
    "IterationConvergenceContext",
    "MaxIterationConvergenceContext",
    "FileFactConvergenceJudge",
    "GeneralConvergenceJudge",
    "ResearchConvergenceJudge",
    "WebReadingConvergenceJudge",
]
