import assert from 'node:assert/strict'
import test from 'node:test'

import { buildRunTimeline, buildStepViewState, findLatestWaitEventContext } from '../src/lib/run-timeline'
import type { SSEEventData } from '../src/lib/api/types'

function eventOf(type: SSEEventData['type'], data: unknown): SSEEventData {
  return { type, data } as SSEEventData
}

test('buildRunTimeline should expose all runtime event kinds explicitly', () => {
  const events: SSEEventData[] = [
    eventOf('message', { role: 'user', message: '请帮我总结计划' }),
    eventOf('plan', { steps: [{ id: 's1', description: 'step 1', status: 'pending' }], title: '任务计划' }),
    eventOf('step', { id: 's1', description: 'step 1', status: 'running' }),
    eventOf('tool', { name: 'search', function: 'search_web', args: { query: 'Actus' }, status: 'calling' }),
    eventOf('wait', { reason: '需要用户确认是否继续' }),
    eventOf('error', { error: 'tool timeout' }),
    eventOf('done', {}),
  ]

  const timeline = buildRunTimeline(events, 'zh-CN')
  assert.deepEqual(
    timeline.map((item) => item.kind),
    ['message', 'plan', 'step', 'tool', 'wait', 'error', 'done'],
  )
  assert.equal(timeline[0]?.summary.startsWith('用户:'), true)
  assert.equal(timeline[3]?.semanticType, 'tool.calling')
})

test('buildRunTimeline should consume compat semantic type when available', () => {
  const events: SSEEventData[] = [
    eventOf('step', {
      id: 's1',
      description: 'step 1',
      status: 'running',
      extensions: {
        compat: {
          schema_version: 'be-lg-08.v2',
          semantic_type: 'step.started',
        },
      },
    }),
  ]

  const timeline = buildRunTimeline(events, 'en-US')
  assert.equal(timeline[0]?.semanticType, 'step.started')
  assert.equal(timeline[0]?.compatSchemaVersion, 'be-lg-08.v2')
})

test('buildRunTimeline should carry runtime multimodal context for downstream rendering', () => {
  const events: SSEEventData[] = [
    eventOf('step', {
      id: 's1',
      description: 'step 1',
      status: 'running',
      extensions: {
        runtime: {
          input_part_summary: { total: 3, by_type: { text: 1, file_ref: 1, image: 1 } },
          downgrade_reason: 'model_multimodal_disabled',
          unsupported_parts: [{ type: 'image', reason: 'model_multimodal_disabled' }],
        },
      },
    }),
  ]

  const timeline = buildRunTimeline(events, 'zh-CN')
  const runtimeContext = timeline[0]?.runtimeContext as Record<string, unknown> | null

  assert.ok(runtimeContext)
  assert.deepEqual(runtimeContext?.input_part_summary, {
    total: 3,
    by_type: { text: 1, file_ref: 1, image: 1 },
  })
  assert.equal(runtimeContext?.downgrade_reason, 'model_multimodal_disabled')
  assert.deepEqual(runtimeContext?.unsupported_parts, [
    { type: 'image', reason: 'model_multimodal_disabled' },
  ])
})

test('buildRunTimeline should prioritize wait question in summary', () => {
  const events: SSEEventData[] = [
    eventOf('wait', {
      question: '请确认是否继续执行后续步骤？',
      reason: '需要用户确认',
      message: 'fallback message',
    }),
  ]

  const timeline = buildRunTimeline(events, 'zh-CN')
  assert.equal(timeline[0]?.kind, 'wait')
  assert.equal(timeline[0]?.summary, '请确认是否继续执行后续步骤？')
})

test('findLatestWaitEventContext should parse latest wait payload for resume', () => {
  const events: SSEEventData[] = [
    eventOf('wait', {
      reason: '需要确认',
      resume_token: 'old-token',
      timeout_at: '1711111111',
    }),
    eventOf('message', { role: 'assistant', message: '补充说明' }),
    eventOf('wait', {
      question: '请补充目标网站地址',
      resume_token: 'new-token',
      suggest_user_takeover: true,
      timeout_at: 1712222222,
    }),
  ]

  const context = findLatestWaitEventContext(events)
  assert.ok(context)
  assert.equal(context?.displayText, '请补充目标网站地址')
  assert.equal(context?.resumeToken, 'new-token')
  assert.equal(context?.suggestUserTakeover, true)
  assert.equal(context?.timeoutAt, 1712222222000)
})

test('buildStepViewState should merge plan snapshot and step updates', () => {
  const events: SSEEventData[] = [
    eventOf('plan', {
      steps: [
        { id: 's1', description: 'collect facts', status: 'pending' },
        { id: 's2', description: 'write summary', status: 'pending' },
      ],
    }),
    eventOf('step', { id: 's1', description: 'collect facts', status: 'completed' }),
    eventOf('step', { id: 's2', description: 'write summary', status: 'running' }),
    eventOf('step', { id: 's3', description: 'review output', status: 'waiting' }),
  ]

  const stepView = buildStepViewState(events)
  assert.equal(stepView.totalCount, 3)
  assert.equal(stepView.completedCount, 1)
  assert.deepEqual(
    stepView.steps.map((step) => ({ id: step.id, status: step.status })),
    [
      { id: 's1', status: 'completed' },
      { id: 's2', status: 'running' },
      { id: 's3', status: 'waiting' },
    ],
  )
})
