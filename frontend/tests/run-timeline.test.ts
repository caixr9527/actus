import assert from 'node:assert/strict'
import test from 'node:test'

import { buildRunTimeline, buildStepViewState } from '../src/lib/run-timeline'
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
