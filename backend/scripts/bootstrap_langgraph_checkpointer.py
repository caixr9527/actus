#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""显式初始化 LangGraph checkpoint schema。"""

import asyncio

from app.infrastructure.runtime import get_langgraph_checkpointer


async def main() -> None:
    checkpointer = get_langgraph_checkpointer()
    await checkpointer.init()
    try:
        # 与应用启动阶段保持一致，显式执行 schema 初始化。
        # 保留这个脚本是为了支持部署前单独预热/排障，而不是依赖隐式副作用。
        await checkpointer.ensure_schema()
    finally:
        await checkpointer.close()


if __name__ == "__main__":
    asyncio.run(main())
