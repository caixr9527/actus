#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/22
@Author : caixiaorong01@outlook.com
@File   : skill_graph_registry.py
"""
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type

from pydantic import BaseModel

from app.domain.external import LLM


SkillGraphFactory = Callable[[LLM], Any]


@dataclass(frozen=True)
class SkillRuntimePolicy:
    """Skill 运行时策略。

    说明：
    - capability_dependencies: Skill 声明的能力依赖，用于后续能力调度与审计。
    - tool_strategy: 工具使用策略（none/local_only/all）。
    - max_iterations: Skill 图内部最大迭代次数，避免异常循环。
    """

    capability_dependencies: Tuple[str, ...] = ()
    tool_strategy: str = "none"
    max_iterations: int = 1


@dataclass(frozen=True)
class SkillDefinition:
    """Skill 定义（BE-LG-10 V1）。"""

    skill_id: str
    version: str
    description: str
    input_schema: Type[BaseModel]
    output_schema: Type[BaseModel]
    runtime_policy: SkillRuntimePolicy
    graph_factory: SkillGraphFactory


def _normalize_version(version: str) -> str:
    normalized = version.strip()
    if not normalized:
        raise ValueError("Skill 版本号不能为空")
    return normalized


def _version_sort_key(version: str) -> Tuple[int, ...]:
    normalized = _normalize_version(version).lower()
    if normalized.startswith("v"):
        normalized = normalized[1:]
    parts = [part for part in re.split(r"[^\d]+", normalized) if part]
    if not parts:
        return (0,)
    return tuple(int(part) for part in parts)


class SkillGraphRuntime:
    """Skill 运行时执行器。

    职责：
    1. 按 skill_id/version 解析 SkillDefinition；
    2. 校验输入与输出 schema；
    3. 缓存已编译 Skill 子图，避免重复编译。
    """

    def __init__(self, registry: "SkillGraphRegistry", llm: LLM) -> None:
        self._registry = registry
        self._llm = llm
        self._compiled_graphs: Dict[Tuple[str, str], Any] = {}

    async def execute_skill(
            self,
            *,
            skill_id: str,
            payload: Dict[str, Any],
            version: Optional[str] = None,
    ) -> BaseModel:
        definition = self._registry.resolve(skill_id=skill_id, version=version)
        validated_input = definition.input_schema.model_validate(payload)
        graph = self._get_or_build_graph(definition)
        if not hasattr(graph, "ainvoke"):
            raise ValueError(f"Skill[{definition.skill_id}@{definition.version}]子图缺少ainvoke方法")

        raw_output = await graph.ainvoke(validated_input.model_dump(mode="json"))
        if not isinstance(raw_output, dict):
            raise ValueError(
                f"Skill[{definition.skill_id}@{definition.version}]输出格式错误，期望dict，实际为{type(raw_output).__name__}"
            )
        return definition.output_schema.model_validate(raw_output)

    def _get_or_build_graph(self, definition: SkillDefinition) -> Any:
        cache_key = (definition.skill_id, definition.version)
        graph = self._compiled_graphs.get(cache_key)
        if graph is not None:
            return graph
        built = definition.graph_factory(self._llm)
        self._compiled_graphs[cache_key] = built
        return built


class SkillGraphRegistry:
    """Skill 子图注册表。"""

    def __init__(self) -> None:
        self._definitions: Dict[str, Dict[str, SkillDefinition]] = {}

    def register(self, definition: SkillDefinition) -> None:
        skill_id = definition.skill_id.strip()
        if not skill_id:
            raise ValueError("Skill ID 不能为空")
        version = _normalize_version(definition.version)
        versions = self._definitions.setdefault(skill_id, {})
        if version in versions:
            raise ValueError(f"Skill[{skill_id}]版本[{version}]已存在，禁止重复注册")
        versions[version] = definition

    def register_many(self, definitions: Iterable[SkillDefinition]) -> None:
        for definition in definitions:
            self.register(definition)

    def list_skill_ids(self) -> List[str]:
        return sorted(self._definitions.keys())

    def list_versions(self, skill_id: str) -> List[str]:
        versions = self._definitions.get(skill_id, {})
        return sorted(versions.keys(), key=_version_sort_key)

    def resolve(self, *, skill_id: str, version: Optional[str] = None) -> SkillDefinition:
        versions = self._definitions.get(skill_id)
        if not versions:
            raise ValueError(f"Skill[{skill_id}]不存在，请检查注册表配置")

        if version is not None:
            normalized = _normalize_version(version)
            definition = versions.get(normalized)
            if definition is None:
                raise ValueError(f"Skill[{skill_id}]版本[{normalized}]不存在")
            return definition

        latest_version = max(versions.keys(), key=_version_sort_key)
        return versions[latest_version]

    def create_runtime(self, llm: LLM) -> SkillGraphRuntime:
        return SkillGraphRuntime(registry=self, llm=llm)
