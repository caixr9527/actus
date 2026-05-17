import assert from 'node:assert/strict'
import test from 'node:test'

import { buildRunTimeline, resolveStepDetail } from '../src/lib/run-timeline'
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
    visibility: 'timeline',
    ...overrides,
  }
}

test('resolveStepDetail should prefer next_hint for failed step', () => {
  assert.equal(
    resolveStepDetail({
      done: false,
      summary: '失败 summary 不应展示',
      produced_artifacts: [],
      blockers: ['失败 blocker'],
      facts_learned: [],
      open_questions: [],
      next_hint: '请缩小范围后重试',
    }, 'failed'),
    '请缩小范围后重试',
  )
})

test('resolveStepDetail should fallback to blocker for failed step without next_hint', () => {
  assert.equal(
    resolveStepDetail({
      done: false,
      summary: '失败 summary 不应展示',
      produced_artifacts: [],
      blockers: ['失败 blocker'],
      facts_learned: [],
      open_questions: [],
      next_hint: '',
    }, 'failed'),
    '失败 blocker',
  )
})

test('resolveStepDetail should keep summary first for completed step', () => {
  assert.equal(
    resolveStepDetail({
      done: true,
      summary: '完成 summary',
      produced_artifacts: [],
      blockers: ['不应优先展示'],
      facts_learned: [],
      open_questions: [],
      next_hint: '不应优先展示',
    }, 'completed'),
    '完成 summary',
  )
})

test('buildRunTimeline should not infer safety audit status from tool failure text', () => {
  const events: SSEEventData[] = [
    {
      type: 'tool',
      data: {
        runtime: runtime(),
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
          data: { reason_code: 'blocked_by_policy' },
        },
        status: 'called',
      },
    },
    {
      type: 'safety_audit',
      data: {
        runtime: runtime({
          source_event_id: 'audit-evt-1',
          cursor_event_id: 'audit-evt-1',
          visibility: 'hidden',
        }),
        event_id: 'audit-evt-1',
        payload: {
          audit_refs: [{
            audit_id: 'audit-1',
            decision: 'block',
            risk_level: 'critical',
            reason_code: 'blocked_by_policy',
            step_id: 'step-1',
            tool_call_id: 'call-1',
            function_name: 'shell_execute',
          }],
          source_event_ids: ['tool-1'],
          decision_counts: { block: 1 },
          risk_counts: { critical: 1 },
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

  const timeline = buildRunTimeline(events, 'zh-CN')

  assert.equal(timeline.length, 1)
  assert.equal(timeline[0].kind, 'tool')
  assert.equal(timeline[0].summary.includes('安全策略阻断'), false)
  assert.equal(timeline.some((item) => item.kind === 'error'), false)
})
