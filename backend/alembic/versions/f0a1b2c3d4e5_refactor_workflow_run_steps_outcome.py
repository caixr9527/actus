"""refactor workflow run steps outcome

Revision ID: f0a1b2c3d4e5
Revises: cb806608ad1c
Create Date: 2026-04-01 16:20:00.000000

"""
import hashlib
from typing import Any, Dict, List, Optional, Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "f0a1b2c3d4e5"
down_revision: Union[str, Sequence[str], None] = "cb806608ad1c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _normalize_step_text(value: Any, fallback: str = "") -> str:
    normalized_value = str(value or "").strip()
    return normalized_value or str(fallback or "").strip()


def _build_objective_key_source(title: str, description: str) -> str:
    normalized_parts: List[str] = []
    for raw_item in [title, description]:
        normalized_item = _normalize_step_text(raw_item)
        if normalized_item and normalized_item not in normalized_parts:
            normalized_parts.append(normalized_item)
    if not normalized_parts:
        return "empty-step"
    return " | ".join(normalized_parts)


def _build_objective_key(title: str, description: str) -> str:
    source_text = _build_objective_key_source(title=title, description=description)
    return hashlib.md5(source_text.encode("utf-8")).hexdigest()[:16]


def _normalize_success_criteria(raw: Any, *, default_description: str) -> List[str]:
    if not isinstance(raw, list):
        return [default_description]
    normalized_items: List[str] = []
    for item in raw:
        normalized_item = _normalize_step_text(item)
        if normalized_item and normalized_item not in normalized_items:
            normalized_items.append(normalized_item)
    return normalized_items or [default_description]


def _normalize_step_outcome(
        raw_outcome: Any,
        *,
        status: str,
        result: Any,
        success: Optional[bool],
        attachments: Any,
) -> Optional[Dict[str, Any]]:
    if isinstance(raw_outcome, dict):
        return raw_outcome

    normalized_status = _normalize_step_text(status).lower()
    normalized_result = _normalize_step_text(result)
    normalized_attachments = []
    if isinstance(attachments, list):
        for item in attachments:
            normalized_item = _normalize_step_text(item)
            if normalized_item and normalized_item not in normalized_attachments:
                normalized_attachments.append(normalized_item)

    if (
            normalized_status not in {"completed", "failed"}
            and not normalized_result
            and not normalized_attachments
            and success is not True
    ):
        return None

    return {
        "done": bool(success) if success is not None else normalized_status == "completed",
        "summary": normalized_result,
        "produced_artifacts": normalized_attachments,
        "blockers": [],
        "facts_learned": [],
        "open_questions": [],
        "next_hint": None,
        "reused_from_run_id": None,
        "reused_from_step_id": None,
    }


def _normalize_step_payload(payload: Any, fallback_description: str = "未命名步骤") -> Dict[str, Any]:
    step_payload = dict(payload or {}) if isinstance(payload, dict) else {}
    description = _normalize_step_text(
        step_payload.get("description"),
        fallback=_normalize_step_text(step_payload.get("title"), fallback=fallback_description),
    )
    title = _normalize_step_text(step_payload.get("title"), fallback=description)
    normalized_payload: Dict[str, Any] = {
        "id": _normalize_step_text(step_payload.get("id")),
        "title": title,
        "description": description,
        "objective_key": _normalize_step_text(
            step_payload.get("objective_key"),
            fallback=_build_objective_key(title=title, description=description),
        ),
        "success_criteria": _normalize_success_criteria(
            step_payload.get("success_criteria"),
            default_description=description,
        ),
        "status": _normalize_step_text(step_payload.get("status"), fallback="pending"),
    }
    outcome = _normalize_step_outcome(
        step_payload.get("outcome"),
        status=normalized_payload["status"],
        result=step_payload.get("result"),
        success=step_payload.get("success") if isinstance(step_payload.get("success"), bool) else None,
        attachments=step_payload.get("attachments"),
    )
    if outcome is not None:
        normalized_payload["outcome"] = outcome
    if step_payload.get("error") is not None:
        normalized_payload["error"] = step_payload.get("error")
    return normalized_payload


def upgrade() -> None:
    """Upgrade schema."""
    op.execute(
        """
        CREATE OR REPLACE FUNCTION _workflow_step_objective_source(title text, description text)
        RETURNS text
        LANGUAGE SQL
        IMMUTABLE
        AS $$
            SELECT CASE
                WHEN normalized_title = '' AND normalized_description = '' THEN 'empty-step'
                WHEN normalized_title = '' THEN normalized_description
                WHEN normalized_description = '' OR normalized_description = normalized_title THEN normalized_title
                ELSE normalized_title || ' | ' || normalized_description
            END
            FROM (
                SELECT
                    BTRIM(COALESCE(title, '')) AS normalized_title,
                    BTRIM(COALESCE(description, '')) AS normalized_description
            ) AS normalized
        $$;
        """
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION _workflow_step_objective_key(title text, description text)
        RETURNS text
        LANGUAGE SQL
        IMMUTABLE
        AS $$
            SELECT SUBSTRING(MD5(_workflow_step_objective_source(title, description)) FROM 1 FOR 16)
        $$;
        """
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION _workflow_step_outcome(
            existing_outcome jsonb,
            step_status text,
            legacy_result text,
            legacy_success boolean,
            legacy_attachments jsonb
        )
        RETURNS jsonb
        LANGUAGE SQL
        IMMUTABLE
        AS $$
            SELECT CASE
                WHEN jsonb_typeof(existing_outcome) = 'object' THEN existing_outcome
                WHEN
                    COALESCE(NULLIF(BTRIM(legacy_result), ''), '') <> ''
                    OR COALESCE(
                        jsonb_array_length(
                            CASE
                                WHEN jsonb_typeof(legacy_attachments) = 'array' THEN legacy_attachments
                                ELSE '[]'::jsonb
                            END
                        ),
                        0
                    ) > 0
                    OR COALESCE(legacy_success, false)
                    OR COALESCE(NULLIF(BTRIM(step_status), ''), '') IN ('completed', 'failed')
                THEN jsonb_build_object(
                    'done', COALESCE(legacy_success, COALESCE(NULLIF(BTRIM(step_status), ''), '') = 'completed'),
                    'summary', COALESCE(legacy_result, ''),
                    'produced_artifacts',
                        CASE
                            WHEN jsonb_typeof(legacy_attachments) = 'array' THEN legacy_attachments
                            ELSE '[]'::jsonb
                        END,
                    'blockers', '[]'::jsonb,
                    'facts_learned', '[]'::jsonb,
                    'open_questions', '[]'::jsonb,
                    'next_hint', NULL,
                    'reused_from_run_id', NULL,
                    'reused_from_step_id', NULL
                )
                ELSE NULL
            END
        $$;
        """
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION _workflow_normalize_step_payload(
            step_payload jsonb,
            fallback_description text DEFAULT '未命名步骤'
        )
        RETURNS jsonb
        LANGUAGE SQL
        IMMUTABLE
        AS $$
            SELECT jsonb_strip_nulls(
                jsonb_build_object(
                    'id', COALESCE(step_payload ->> 'id', ''),
                    'title', normalized_title,
                    'description', normalized_description,
                    'objective_key',
                        COALESCE(
                            NULLIF(BTRIM(step_payload ->> 'objective_key'), ''),
                            _workflow_step_objective_key(normalized_title, normalized_description)
                        ),
                    'success_criteria',
                        CASE
                            WHEN jsonb_typeof(step_payload -> 'success_criteria') = 'array'
                                AND jsonb_array_length(step_payload -> 'success_criteria') > 0
                            THEN step_payload -> 'success_criteria'
                            ELSE jsonb_build_array(normalized_description)
                        END,
                    'status', COALESCE(NULLIF(BTRIM(step_payload ->> 'status'), ''), 'pending'),
                    'outcome',
                        _workflow_step_outcome(
                            step_payload -> 'outcome',
                            step_payload ->> 'status',
                            step_payload ->> 'result',
                            CASE
                                WHEN step_payload ? 'success' THEN COALESCE((step_payload ->> 'success')::boolean, false)
                                ELSE NULL
                            END,
                            step_payload -> 'attachments'
                        ),
                    'error', step_payload -> 'error'
                )
            )
            FROM (
                SELECT
                    COALESCE(
                        NULLIF(BTRIM(step_payload ->> 'description'), ''),
                        NULLIF(BTRIM(step_payload ->> 'title'), ''),
                        fallback_description
                    ) AS normalized_description,
                    COALESCE(
                        NULLIF(BTRIM(step_payload ->> 'title'), ''),
                        NULLIF(BTRIM(step_payload ->> 'description'), ''),
                        fallback_description
                    ) AS normalized_title
            ) AS normalized
        $$;
        """
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION _workflow_normalize_step_state_payload(
            step_state_payload jsonb,
            fallback_description text DEFAULT '未命名步骤'
        )
        RETURNS jsonb
        LANGUAGE SQL
        IMMUTABLE
        AS $$
            SELECT jsonb_strip_nulls(
                jsonb_build_object(
                    'step_id', COALESCE(step_state_payload ->> 'step_id', ''),
                    'step_index', COALESCE(step_state_payload -> 'step_index', to_jsonb(0)),
                    'title', normalized_title,
                    'description', normalized_description,
                    'objective_key',
                        COALESCE(
                            NULLIF(BTRIM(step_state_payload ->> 'objective_key'), ''),
                            _workflow_step_objective_key(normalized_title, normalized_description)
                        ),
                    'success_criteria',
                        CASE
                            WHEN jsonb_typeof(step_state_payload -> 'success_criteria') = 'array'
                                AND jsonb_array_length(step_state_payload -> 'success_criteria') > 0
                            THEN step_state_payload -> 'success_criteria'
                            ELSE jsonb_build_array(normalized_description)
                        END,
                    'status', COALESCE(NULLIF(BTRIM(step_state_payload ->> 'status'), ''), 'pending'),
                    'outcome',
                        _workflow_step_outcome(
                            step_state_payload -> 'outcome',
                            step_state_payload ->> 'status',
                            step_state_payload ->> 'result',
                            CASE
                                WHEN step_state_payload ? 'success'
                                THEN COALESCE((step_state_payload ->> 'success')::boolean, false)
                                ELSE NULL
                            END,
                            step_state_payload -> 'attachments'
                        ),
                    'error', step_state_payload -> 'error'
                )
            )
            FROM (
                SELECT
                    COALESCE(
                        NULLIF(BTRIM(step_state_payload ->> 'description'), ''),
                        NULLIF(BTRIM(step_state_payload ->> 'title'), ''),
                        fallback_description
                    ) AS normalized_description,
                    COALESCE(
                        NULLIF(BTRIM(step_state_payload ->> 'title'), ''),
                        NULLIF(BTRIM(step_state_payload ->> 'description'), ''),
                        fallback_description
                    ) AS normalized_title
            ) AS normalized
        $$;
        """
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION _workflow_normalize_graph_state_payload(graph_state jsonb)
        RETURNS jsonb
        LANGUAGE SQL
        IMMUTABLE
        AS $$
            SELECT CASE
                WHEN jsonb_typeof(graph_state) <> 'object' THEN graph_state
                ELSE graph_state
                    || CASE
                        WHEN jsonb_typeof(graph_state -> 'plan') = 'object'
                        THEN jsonb_build_object(
                            'plan',
                            (graph_state -> 'plan')
                            || jsonb_build_object(
                                'steps',
                                COALESCE(
                                    (
                                        SELECT jsonb_agg(
                                            _workflow_normalize_step_payload(step_item.value)
                                            ORDER BY step_item.ordinality
                                        )
                                        FROM jsonb_array_elements(
                                            COALESCE(graph_state -> 'plan' -> 'steps', '[]'::jsonb)
                                        ) WITH ORDINALITY AS step_item(value, ordinality)
                                    ),
                                    '[]'::jsonb
                                )
                            )
                        )
                        ELSE '{}'::jsonb
                    END
                    || CASE
                        WHEN jsonb_typeof(graph_state -> 'last_executed_step') = 'object'
                        THEN jsonb_build_object(
                            'last_executed_step',
                            _workflow_normalize_step_payload(graph_state -> 'last_executed_step')
                        )
                        ELSE '{}'::jsonb
                    END
                    || CASE
                        WHEN jsonb_typeof(graph_state -> 'step_states') = 'array'
                        THEN jsonb_build_object(
                            'step_states',
                            COALESCE(
                                (
                                    SELECT jsonb_agg(
                                        _workflow_normalize_step_state_payload(step_state_item.value)
                                        ORDER BY step_state_item.ordinality
                                    )
                                    FROM jsonb_array_elements(graph_state -> 'step_states')
                                    WITH ORDINALITY AS step_state_item(value, ordinality)
                                ),
                                '[]'::jsonb
                            )
                        )
                        ELSE '{}'::jsonb
                    END
            END
        $$;
        """
    )
    op.add_column(
        "workflow_run_steps",
        sa.Column("title", sa.Text(), server_default=sa.text("''::text"), nullable=False),
    )
    op.add_column(
        "workflow_run_steps",
        sa.Column("objective_key", sa.String(length=255), server_default=sa.text("''::character varying"), nullable=False),
    )
    op.add_column(
        "workflow_run_steps",
        sa.Column(
            "success_criteria",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    )
    op.add_column(
        "workflow_run_steps",
        sa.Column("outcome", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )

    # 将旧字段一次性投影到新结构，完成后删除旧列，不保留兼容层。
    op.execute(
        """
        UPDATE workflow_run_steps
        SET
            title = COALESCE(NULLIF(BTRIM(description), ''), '未命名步骤'),
            objective_key = _workflow_step_objective_key(
                COALESCE(NULLIF(BTRIM(description), ''), '未命名步骤'),
                COALESCE(NULLIF(BTRIM(description), ''), '未命名步骤')
            ),
            success_criteria = jsonb_build_array(COALESCE(NULLIF(BTRIM(description), ''), '未命名步骤')),
            outcome = _workflow_step_outcome(
                NULL,
                status,
                result,
                success,
                attachments
            )
        """
    )
    op.execute(
        """
        UPDATE workflow_run_events
        SET event_payload = jsonb_set(
            event_payload,
            '{step}',
            _workflow_normalize_step_payload(event_payload -> 'step'),
            false
        )
        WHERE COALESCE(NULLIF(event_payload ->> 'type', ''), event_type) = 'step'
            AND jsonb_typeof(event_payload -> 'step') = 'object'
        """
    )
    op.execute(
        """
        UPDATE workflow_run_events
        SET event_payload = jsonb_set(
            event_payload,
            '{plan}',
            (event_payload -> 'plan')
            || jsonb_build_object(
                'steps',
                COALESCE(
                    (
                        SELECT jsonb_agg(
                            _workflow_normalize_step_payload(step_item.value)
                            ORDER BY step_item.ordinality
                        )
                        FROM jsonb_array_elements(
                            COALESCE(event_payload -> 'plan' -> 'steps', '[]'::jsonb)
                        ) WITH ORDINALITY AS step_item(value, ordinality)
                    ),
                    '[]'::jsonb
                )
            ),
            false
        )
        WHERE COALESCE(NULLIF(event_payload ->> 'type', ''), event_type) = 'plan'
            AND jsonb_typeof(event_payload -> 'plan') = 'object'
        """
    )
    op.execute(
        """
        UPDATE workflow_runs
        SET runtime_metadata = jsonb_set(
            runtime_metadata,
            '{graph_state_contract,graph_state}',
            _workflow_normalize_graph_state_payload(
                runtime_metadata #> '{graph_state_contract,graph_state}'
            ),
            false
        )
        WHERE jsonb_typeof(runtime_metadata #> '{graph_state_contract,graph_state}') = 'object'
        """
    )

    op.drop_column("workflow_run_steps", "attachments")
    op.drop_column("workflow_run_steps", "success")
    op.drop_column("workflow_run_steps", "result")
    op.execute("DROP FUNCTION _workflow_normalize_graph_state_payload(jsonb)")
    op.execute("DROP FUNCTION _workflow_normalize_step_state_payload(jsonb, text)")
    op.execute("DROP FUNCTION _workflow_normalize_step_payload(jsonb, text)")
    op.execute("DROP FUNCTION _workflow_step_outcome(jsonb, text, text, boolean, jsonb)")
    op.execute("DROP FUNCTION _workflow_step_objective_key(text, text)")
    op.execute("DROP FUNCTION _workflow_step_objective_source(text, text)")


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column(
        "workflow_run_steps",
        sa.Column(
            "attachments",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    )
    op.add_column(
        "workflow_run_steps",
        sa.Column("success", sa.Boolean(), server_default=sa.text("false"), nullable=False),
    )
    op.add_column("workflow_run_steps", sa.Column("result", sa.Text(), nullable=True))

    op.execute(
        """
        UPDATE workflow_run_steps
        SET
            result = outcome ->> 'summary',
            success = COALESCE((outcome ->> 'done')::boolean, false),
            attachments = COALESCE(outcome -> 'produced_artifacts', '[]'::jsonb)
        """
    )

    op.drop_column("workflow_run_steps", "outcome")
    op.drop_column("workflow_run_steps", "success_criteria")
    op.drop_column("workflow_run_steps", "objective_key")
    op.drop_column("workflow_run_steps", "title")
