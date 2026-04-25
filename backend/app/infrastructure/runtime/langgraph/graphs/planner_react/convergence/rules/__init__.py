#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""收敛规则集合。"""

from .file_fact_rule import FileFactConvergenceRule
from .general_file_observation_rule import GeneralFileObservationConvergenceRule
from .loop_break_rule import LoopBreakConvergenceRule
from .research_rule import ResearchConvergenceRule
from .web_reading_rule import WebReadingConvergenceRule

__all__ = [
    "FileFactConvergenceRule",
    "GeneralFileObservationConvergenceRule",
    "LoopBreakConvergenceRule",
    "ResearchConvergenceRule",
    "WebReadingConvergenceRule",
]
