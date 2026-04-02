import assert from 'node:assert/strict'
import test from 'node:test'

import {
  appendTimelineEvent,
  createTimelineBuildContext,
  eventsToTimeline,
} from '../src/lib/session-events'
import type { SSEEventData } from '../src/lib/api/types'

function eventOf(type: SSEEventData['type'], data: unknown): SSEEventData {
  return { type, data } as SSEEventData
}

test('eventsToTimeline should merge tool updates and reset step context across user turns', () => {
  const events: SSEEventData[] = [
    eventOf('message', {
      role: 'user',
      message: 'first question',
      attachments: [{ file_id: 'f1', filename: 'a.txt' }],
    }),
    eventOf('step', { id: 'step-1', status: 'running', description: 'phase 1' }),
    eventOf('tool', {
      tool_call_id: 'tool-1',
      name: 'browser',
      function: 'browser_open',
      args: { url: 'https://example.com' },
      status: 'calling',
    }),
    eventOf('tool', {
      tool_call_id: 'tool-1',
      name: 'browser',
      function: 'browser_open',
      args: { url: 'https://example.com' },
      status: 'called',
      content: 'ok',
    }),
    eventOf('step', { id: 'step-1', status: 'completed', description: 'phase 1' }),
    eventOf('message', { role: 'user', message: 'second question' }),
    eventOf('step', {
      id: 'step-1',
      status: 'running',
      description: 'phase 2',
      outcome: {
        done: false,
        summary: '上一步搜索任务超时，准备重试',
        produced_artifacts: [],
        blockers: ['当前步骤超过 180 秒未完成'],
        facts_learned: [],
        open_questions: [],
        next_hint: '请缩小当前步骤范围后重试',
      },
    }),
    eventOf('error', {
      error: 'broken',
      error_key: 'error.session.not_found',
    }),
  ]

  const timeline = eventsToTimeline(events, 'en-US')

  const stepItems = timeline.filter((item) => item.kind === 'step')
  assert.equal(stepItems.length, 2)

  const firstStep = stepItems[0]
  assert.equal(firstStep.kind, 'step')
  if (firstStep.kind === 'step') {
    // 同一 tool_call_id 的 calling/called 应合并为一条最新状态记录，避免重复展示。
    assert.equal(firstStep.tools.length, 1)
    assert.equal((firstStep.tools[0] as { status?: string }).status, 'called')
    assert.equal((firstStep.tools[0] as { content?: string }).content, 'ok')
  }

  const secondStep = stepItems[1]
  assert.equal(secondStep.kind, 'step')
  if (secondStep.kind === 'step') {
    assert.equal(secondStep.tools.length, 0)
    assert.equal(secondStep.data.description, 'phase 2')
    assert.equal(secondStep.data.outcome?.summary, '上一步搜索任务超时，准备重试')
  }

  const attachmentItems = timeline.filter((item) => item.kind === 'attachments')
  assert.equal(attachmentItems.length, 1)

  const errorItems = timeline.filter((item) => item.kind === 'error')
  assert.equal(errorItems.length, 1)
  if (errorItems[0]?.kind === 'error') {
    assert.equal(errorItems[0].error, 'This task session does not exist or has been deleted')
  }
})

test('appendTimelineEvent should match full timeline build in append-only stream', () => {
  const events: SSEEventData[] = [
    eventOf('message', { role: 'user', message: 'q1' }),
    eventOf('step', { id: 's-1', status: 'running', description: 'step 1 running' }),
    eventOf('tool', {
      tool_call_id: 'tool-1',
      name: 'search',
      function: 'search_web',
      args: { q: 'hello' },
      status: 'calling',
    }),
    eventOf('tool', {
      tool_call_id: 'tool-1',
      name: 'search',
      function: 'search_web',
      args: { q: 'hello' },
      status: 'called',
      content: { results: [] },
    }),
    eventOf('step', { id: 's-1', status: 'completed', description: 'step 1 done' }),
    eventOf('message', { role: 'assistant', message: 'answer' }),
    eventOf('message', { role: 'user', message: 'q2' }),
    eventOf('step', { id: 's-1', status: 'running', description: 'step 2 running' }),
    eventOf('error', { error: 'boom', error_key: 'error.session.sandbox_unavailable' }),
  ]

  const context = createTimelineBuildContext()
  for (const event of events) {
    appendTimelineEvent(context, event)
  }

  const full = eventsToTimeline(events)
  assert.deepEqual(context.list, full)
})
