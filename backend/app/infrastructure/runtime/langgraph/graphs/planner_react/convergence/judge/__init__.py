#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""收敛判定器集合。"""

from .file_fact_convergence import FileFactConvergenceJudge
from .general_convergence import GeneralConvergenceJudge
from .research_convergence import ResearchConvergenceJudge
from .web_reading_convergence import WebReadingConvergenceJudge

__all__ = [
    "FileFactConvergenceJudge",
    "GeneralConvergenceJudge",
    "ResearchConvergenceJudge",
    "WebReadingConvergenceJudge",
]
