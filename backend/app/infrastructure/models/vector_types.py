#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""面向 asyncpg 的 pgvector 类型封装。"""

from pgvector.sqlalchemy.vector import VECTOR


class AsyncpgVector(VECTOR):
    """与 asyncpg pgvector binary codec 对齐的向量类型。

    说明：
    1. 仓库已在 asyncpg 连接上注册 `pgvector.asyncpg.register_vector`；
    2. 该 codec 期望收到原始 list / numpy.ndarray，而不是 `'[1,2,3]'` 形式的文本；
    3. 原始 `pgvector.sqlalchemy.VECTOR.bind_processor()` 会先把值转成文本，
       与 asyncpg binary codec 叠加后会触发类型转换错误；
    4. 因此这里显式返回原始值，让 asyncpg codec 负责最终编码。
    """

    cache_ok = True

    def bind_processor(self, dialect):
        def process(value):
            return value

        return process
