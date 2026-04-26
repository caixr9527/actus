#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 解耦：策略插件集合。"""

from .effects_plugin import (
    run_effects_plugin,
    run_preinvoke_effects_plugin,
    run_rewrite_effects_plugin,
)
from .executor_plugin import run_executor_plugin

__all__ = [
    "run_executor_plugin",
    "run_effects_plugin",
    "run_preinvoke_effects_plugin",
    "run_rewrite_effects_plugin",
]
