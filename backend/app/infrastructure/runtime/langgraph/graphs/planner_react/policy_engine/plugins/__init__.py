#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 解耦：策略插件集合。"""

from .convergence_plugin import run_convergence_plugin
from .effects_plugin import run_effects_plugin
from .executor_plugin import run_executor_plugin
from .guard_plugin import run_guard_plugin
from .rewrite_plugin import RewriteDecision, run_rewrite_plugin

__all__ = [
    "run_guard_plugin",
    "run_executor_plugin",
    "run_effects_plugin",
    "run_convergence_plugin",
    "run_rewrite_plugin",
    "RewriteDecision",
]
