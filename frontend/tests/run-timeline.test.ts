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
    eventOf('wait', { payload: { kind: 'confirm', prompt: '需要用户确认是否继续' } }),
    eventOf('error', { error: 'tool timeout' }),
    eventOf('done', {}),
  ]

  const timeline = buildRunTimeline(events, 'zh-CN')
  assert.deepEqual(
    timeline.map((item) => item.kind),
    ['message', 'plan', 'step', 'tool', 'wait', 'error', 'done'],
  )
  assert.equal(timeline[0]?.summary.startsWith('用户:'), true)
})

test('buildRunTimeline should prioritize wait question in summary', () => {
  const events: SSEEventData[] = [
    eventOf('wait', {
      interrupt_id: 'interrupt-1',
      payload: {
        kind: 'confirm',
        prompt: '请确认是否继续执行后续步骤？',
        details: '需要用户确认',
      },
    }),
  ]

  const timeline = buildRunTimeline(events, 'zh-CN')
  assert.equal(timeline[0]?.kind, 'wait')
  assert.equal(timeline[0]?.summary, '请确认是否继续执行后续步骤？')
})

test('findLatestWaitEventContext should parse latest interrupt wait payload', () => {
  const events: SSEEventData[] = [
    eventOf('wait', {
      interrupt_id: 'interrupt-1',
      payload: {
        kind: 'input_text',
        prompt: '需要确认',
        response_key: 'message',
      },
    }),
    eventOf('message', { role: 'assistant', message: '补充说明' }),
    eventOf('wait', {
      interrupt_id: 'interrupt-2',
      payload: {
        kind: 'input_text',
        prompt: '请补充目标网站地址',
        response_key: 'message',
        suggest_user_takeover: 'browser',
      },
    }),
  ]

  const context = findLatestWaitEventContext(events)
  assert.ok(context)
  assert.equal(context?.interruptId, 'interrupt-2')
  assert.equal(context?.payload?.kind, 'input_text')
  assert.equal(context?.prompt, '请补充目标网站地址')
  assert.equal(context?.suggestUserTakeover, true)
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

test('buildRunTimeline and buildStepViewState should surface step outcome detail', () => {
  const events: SSEEventData[] = [
    eventOf('step', {
      id: 's1',
      description: 'search course page',
      status: 'failed',
      outcome: {
        done: false,
        summary: '步骤执行超时：搜索课程页面',
        produced_artifacts: [],
        blockers: ['当前步骤超过 180 秒未完成'],
        facts_learned: [],
        open_questions: [],
        next_hint: '请缩小当前步骤范围后重试',
      },
    }),
  ]

  const timeline = buildRunTimeline(events, 'zh-CN')
  const stepView = buildStepViewState(events)

  assert.equal(timeline[0]?.summary, '步骤执行超时：搜索课程页面')
  assert.equal(stepView.steps[0]?.detail, '步骤执行超时：搜索课程页面')
})
