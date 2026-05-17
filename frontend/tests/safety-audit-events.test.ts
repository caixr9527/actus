import assert from 'node:assert/strict'
import test from 'node:test'

import { collectSafetyAuditSummaryItems } from '../src/lib/safety-audit-events'
import type { RuntimeEventMeta, SSEEventData } from '../src/lib/api/types'

function runtime(overrides: Partial<RuntimeEventMeta> = {}): RuntimeEventMeta {
  return {
    session_id: 'session-1',
    run_id: 'run-1',
    status_after_event: null,
    current_step_id: 'step-1',
    source_event_id: 'evt-1',
    cursor_event_id: 'evt-1',
    durability: 'persistent',
    visibility: 'hidden',
    ...overrides,
  }
}

test('collectSafetyAuditSummaryItems should only consume safety_audit events', () => {
  const events: SSEEventData[] = [
    {
      type: 'tool',
      data: {
        runtime: runtime({ visibility: 'timeline' }),
        event_id: 'tool-1',
        name: 'shell',
        function: 'shell_execute',
        args: { command: 'rm -rf /*' },
        step_id: 'step-1',
        tool_call_id: 'call-1',
        tool_name: 'shell',
        function_name: 'shell_execute',
        function_args: { command: 'rm -rf /*' },
        function_result: {
          success: false,
          message: '安全策略阻断：禁止执行 rm -rf /*',
          data: {
            audit_id: 'fake-audit-from-tool',
            decision: 'block',
            risk_level: 'critical',
            reason_code: 'blocked_by_policy',
          },
        },
        runtime_metadata: {
          safety_hint: {
            audit_id: 'fake-audit-from-runtime-metadata',
            decision: 'block',
          },
        },
        status: 'called',
      },
    },
  ]

  assert.deepEqual(collectSafetyAuditSummaryItems(events), [])
})

test('collectSafetyAuditSummaryItems should surface only material safety audit refs', () => {
  const events: SSEEventData[] = [
    {
      type: 'safety_audit',
      data: {
        runtime: runtime(),
        event_id: 'audit-evt-1',
        payload: {
          audit_refs: [
            {
              audit_id: 'audit-low-allow',
              decision: 'allow',
              risk_level: 'low',
              reason_code: 'allow',
              step_id: 'step-1',
              tool_call_id: 'call-1',
              function_name: 'read_file',
            },
            {
              audit_id: 'audit-block',
              decision: 'block',
              risk_level: 'critical',
              reason_code: 'blocked_by_policy',
              step_id: 'step-1',
              tool_call_id: 'call-2',
              function_name: 'shell_execute',
            },
          ],
          source_event_ids: ['tool-1'],
          decision_counts: { allow: 1, block: 1 },
          risk_counts: { low: 1, critical: 1 },
          blocked_count: 1,
          rewrite_count: 0,
          confirmation_count: 0,
          summary: '1 个高风险动作，1 个阻断',
          runtime_metadata: {
            visibility: 'hidden',
            projection_key: 'safety_audit:v1:user:session:run:audit',
            schema_version: 'safety_audit_event.v1',
          },
        },
      },
    },
  ]

  assert.deepEqual(collectSafetyAuditSummaryItems(events), [
    {
      auditId: 'audit-block',
      decision: 'block',
      riskLevel: 'critical',
      reasonCode: 'blocked_by_policy',
      functionName: 'shell_execute',
      stepId: 'step-1',
      toolCallId: 'call-2',
    },
  ])
})
