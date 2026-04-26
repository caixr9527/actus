#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Workspace Runtime capabilities。"""

from .browser_capability import WorkspaceBrowserCapability
from .file_capability import WorkspaceFileCapability
from .research_capability import WorkspaceResearchCapability
from .shell_capability import WorkspaceShellCapability

__all__ = [
    "WorkspaceBrowserCapability",
    "WorkspaceFileCapability",
    "WorkspaceResearchCapability",
    "WorkspaceShellCapability",
]
