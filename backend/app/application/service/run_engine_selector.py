#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/20 13:10
@Author : caixiaorong01@outlook.com
@File   : run_engine_selector.py
"""
import logging
from typing import Callable, Dict

from app.application.service.artifact_ledger_service import ArtifactLedgerService
from app.application.service.artifact_revision_projector import ArtifactRevisionProjector
from app.application.service.data_retention_policy_service import DataRetentionPolicyService
from app.application.service.evidence_digest_projector import EvidenceDigestProjector
from app.application.service.evidence_fact_assembler import EvidenceFactAssembler
from app.application.service.evidence_result_handle_resolver import EvidenceResultHandleResolver
from app.application.service.evidence_runtime_context_provider import EvidenceRuntimeContextProvider
from app.application.service.evidence_ledger_service import EvidenceLedgerService
from app.application.service.runtime_access_control_service import RuntimeAccessControlService
from app.application.service.sandbox_fact_document_input_projector import SandboxFactDocumentInputProjector
from app.application.service.sandbox_fact_ledger_service import SandboxFactLedgerService
from app.domain.external import LLM, JSONParser, Browser, Sandbox, SearchEngine, FileStorage
from app.domain.models import AgentConfig, MCPConfig
from app.domain.repositories import IUnitOfWork
from app.domain.services.runtime import RunEngine
from app.domain.services.runtime.contracts.evidence_runtime_ports import EvidenceStepReconcilerPort
from app.domain.services.runtime.contracts.sandbox_capability_profile_ports import RuntimeToolSnapshotRecorderPort
from app.domain.services.runtime.contracts.sandbox_fact_ports import SandboxFactProjectionContextBuilderPort
from app.domain.services.runtime.contracts.sandbox_fact_ports import RuntimeToolEventPersistencePort
from app.domain.services.workspace_runtime.context import RuntimeContextService
from app.domain.services.runtime.stage_llm import REQUIRED_STAGE_LLM_NAMES
from app.domain.services.tools import MCPTool, A2ATool, ToolRuntimeAdapter, CapabilityBuildContext
from app.domain.services.workspace_runtime import WorkspaceRuntimeService
from app.infrastructure.runtime.langgraph import LangGraphRunEngine, get_langgraph_checkpointer
from core.config import get_settings

logger = logging.getLogger(__name__)


def _maybe_clone_llm_for_stage(
        llm: LLM,
        *,
        model_name: str | None,
        max_tokens: int | None,
) -> LLM | None:
    clone_with_overrides = getattr(llm, "clone_with_overrides", None)
    if not callable(clone_with_overrides):
        return None
    return clone_with_overrides(
        model_name=model_name,
        max_tokens=max_tokens,
    )


def _build_stage_llms(llm: LLM) -> Dict[str, LLM]:
    stage_specs: Dict[str, Dict[str, str | int | None]] = {
        # todo 临时使用 LLM 模型名称及max_tokens，后续会支持自定义，不要对这处进行修改【重点】
        "router": {
            "model_name": llm.model_name,
            "max_tokens": getattr(llm, "max_tokens", 2048),
        },
        "planner": {
            "model_name": llm.model_name,
            "max_tokens": getattr(llm, "max_tokens", 4096),
        },
        "executor": {
            "model_name": llm.model_name,
            "max_tokens": getattr(llm, "max_tokens", 8192),
        },
        "replan": {
            "model_name": llm.model_name,
            "max_tokens": getattr(llm, "max_tokens", 4096),
        },
        "summary": {
            "model_name": llm.model_name,
            "max_tokens": getattr(llm, "max_tokens", 4096),
        },
    }
    stage_llms: Dict[str, LLM] = {}
    for stage_name in REQUIRED_STAGE_LLM_NAMES:
        spec = stage_specs[stage_name]
        stage_llm = _maybe_clone_llm_for_stage(
            llm,
            model_name=spec["model_name"],
            max_tokens=spec["max_tokens"],
        )
        stage_llms[stage_name] = stage_llm or llm
    return stage_llms


async def build_run_engine(
        llm: LLM,
        agent_config: AgentConfig,
        session_id: str,
        file_storage: FileStorage,
        uow_factory: Callable[[], IUnitOfWork],
        json_parser: JSONParser,
        browser: Browser,
        sandbox: Sandbox,
        search_engine: SearchEngine,
        mcp_tool: MCPTool,
        a2a_tool: A2ATool,
        workspace_runtime_service: WorkspaceRuntimeService,
        mcp_config: MCPConfig | None = None,
        user_id: str | None = None,
        tool_runtime_adapter: ToolRuntimeAdapter | None = None,
        runtime_tool_snapshot_recorder: RuntimeToolSnapshotRecorderPort | None = None,
        sandbox_fact_context_builder: SandboxFactProjectionContextBuilderPort | None = None,
        evidence_step_reconciler: EvidenceStepReconcilerPort | None = None,
        runtime_tool_event_persistence: RuntimeToolEventPersistencePort | None = None,
) -> RunEngine:
    """根据配置选择运行时引擎（BE-LG-12 起仅支持 LangGraph）。"""
    settings = get_settings()
    engine_kind = settings.agent_runtime_engine.strip().lower()

    if engine_kind != "langgraph":
        raise ValueError(f"不支持的运行时引擎配置: {engine_kind}，当前仅支持 langgraph")
    if tool_runtime_adapter is None:
        raise ValueError("tool_runtime_adapter 不能为空")
    if runtime_tool_snapshot_recorder is None:
        raise ValueError("runtime_tool_snapshot_recorder 不能为空")
    if user_id is None:
        raise ValueError("user_id 不能为空")

    logger.info("启用 LangGraph 运行时引擎")
    runtime_tools_with_snapshot = tool_runtime_adapter.build_runtime_tools_with_snapshot(
        capability_context=CapabilityBuildContext(
            sandbox=sandbox,
            browser=browser,
            search_engine=search_engine,
            workspace_runtime_service=workspace_runtime_service,
            mcp_tool=mcp_tool,
            a2a_tool=a2a_tool,
            mcp_config=mcp_config,
            user_id=user_id,
        ),
        mcp_tool=mcp_tool,
        mcp_config=mcp_config,
        a2a_tool=a2a_tool,
    )
    await runtime_tool_snapshot_recorder.record_runtime_tool_snapshot(
        user_id=user_id,
        session_id=session_id,
        snapshot=runtime_tools_with_snapshot.snapshot,
    )
    # P0: 单步骤工具循环先做硬上限收口，避免错误回路被配置值无限放大。
    max_tool_iterations = max(1, min(int(agent_config.max_iterations), 20))
    evidence_digest_projector = EvidenceDigestProjector(uow_factory=uow_factory)
    evidence_ledger_service = evidence_step_reconciler or EvidenceLedgerService(
        uow_factory=uow_factory,
        assembler=EvidenceFactAssembler(),
        step_projection=evidence_digest_projector,
    )
    return LangGraphRunEngine(
        session_id=session_id,
        stage_llms=_build_stage_llms(llm),
        file_storage=file_storage,
        user_id=user_id,
        uow_factory=uow_factory,
        runtime_tools=runtime_tools_with_snapshot.runtime_tools,
        runtime_context_service=RuntimeContextService(
            workspace_runtime_service=workspace_runtime_service,
            evidence_context_provider=EvidenceRuntimeContextProvider(
                ledger_service=evidence_ledger_service,
                projector=evidence_digest_projector,
            ),
        ),
        evidence_result_handle_resolver=EvidenceResultHandleResolver(uow_factory=uow_factory),
        evidence_step_reconciler=evidence_ledger_service,
        runtime_tool_event_persistence=runtime_tool_event_persistence,
        max_tool_iterations=max_tool_iterations,
        checkpointer=get_langgraph_checkpointer().get_checkpointer(),
        data_retention_policy_service=DataRetentionPolicyService(),
        sandbox_fact_document_projector=SandboxFactDocumentInputProjector(
            ledger_service=SandboxFactLedgerService(uow_factory=uow_factory),
            artifact_revision_projector=ArtifactRevisionProjector(
                ledger_service=ArtifactLedgerService(uow_factory=uow_factory),
            ),
        ) if sandbox_fact_context_builder is not None else None,
        sandbox_fact_context_builder=sandbox_fact_context_builder,
        access_control_service=RuntimeAccessControlService(uow_factory=uow_factory),
    )
