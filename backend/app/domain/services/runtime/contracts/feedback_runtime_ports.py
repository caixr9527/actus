#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Feedback runtime 接入端口。"""

from __future__ import annotations

from typing import Protocol

from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.services.runtime.contracts.artifact_governance_contract import (
    ArtifactRevisionResolveCommand,
    ArtifactRevisionResolveResult,
)
from app.domain.services.runtime.contracts.feedback_contract import (
    FeedbackGapResult,
    FeedbackResolutionCommand,
    FeedbackScopeKind,
    FeedbackSnapshotResult,
    FeedbackSnapshotStage,
    FeedbackWriteResult,
    QualityFeedbackCommand,
    RuntimeFeedbackCommand,
    UserFeedbackCommand,
)


class FeedbackRecorderPort(Protocol):
    """runtime 主链路写入 feedback 的统一端口。"""

    async def record_user_feedback(self, command: UserFeedbackCommand) -> FeedbackWriteResult:
        ...

    async def record_runtime_feedback(self, command: RuntimeFeedbackCommand) -> FeedbackWriteResult:
        ...

    async def record_quality_feedback(self, command: QualityFeedbackCommand) -> FeedbackWriteResult:
        ...

    async def resolve_feedback(self, command: FeedbackResolutionCommand) -> FeedbackWriteResult:
        ...


class FeedbackSnapshotProviderPort(Protocol):
    """为 runtime/stage 消费方构建 feedback snapshot 的只读端口。"""

    async def build_snapshot(
            self,
            *,
            access_scope: AccessScopeResult,
            stage: FeedbackSnapshotStage,
            feedback_scope_kind: FeedbackScopeKind,
            requested_scope_id: str | None = None,
            runtime_gaps: list[FeedbackGapResult] | None = None,
    ) -> FeedbackSnapshotResult:
        ...


class FeedbackEventProjectorPort(Protocol):
    """FeedbackRecord 成功写入或 lifecycle 更新后的统一投影端口。"""

    async def project_record_written(self, record_ref: FeedbackWriteResult) -> None:
        ...

    async def project_resolution_updated(self, record_ref: FeedbackWriteResult) -> None:
        ...


class RuntimeFeedbackGapSinkPort(Protocol):
    """当前执行上下文内的 transient feedback gap 收集端口。"""

    def append_feedback_gap(self, gap: FeedbackGapResult) -> None:
        ...

    def get_feedback_gaps(self) -> list[FeedbackGapResult]:
        ...


class ArtifactRevisionResolverPort(Protocol):
    """Feedback 侧依赖的 artifact revision 解析端口。"""

    async def resolve(self, command: ArtifactRevisionResolveCommand) -> ArtifactRevisionResolveResult:
        ...
