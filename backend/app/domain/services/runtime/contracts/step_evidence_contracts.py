#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""步骤执行证据共享契约。"""

# 执行模型可能返回“自然语言正文 + JSON”。自然语言正文不直接作为最终出口，
# 但需要用稳定前缀沉淀到 facts_learned，供 summary 异常时兜底。
STEP_DRAFT_FACT_PREFIX = "当前步骤生成的正文草稿："
