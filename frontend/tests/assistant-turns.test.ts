import assert from 'node:assert/strict'
import test from 'node:test'

import { timelineToConversationItems } from '../src/lib/assistant-turns'
import { eventsToTimeline } from '../src/lib/session-events'
import type { RuntimeEventMeta, SSEEventData } from '../src/lib/api/types'

let runtimeEventSequence = 0

function runtime(overrides: Partial<RuntimeEventMeta> = {}): RuntimeEventMeta {
  runtimeEventSequence += 1
  const eventId = `turn-evt-${runtimeEventSequence}`
  return {
    session_id: 'session-1',
    run_id: 'run-1',
    status_after_event: null,
    current_step_id: null,
    source_event_id: eventId,
    cursor_event_id: eventId,
    durability: 'persistent',
    visibility: 'timeline',
    ...overrides,
  }
}

function eventOf(
  type: SSEEventData['type'],
  data: Record<string, unknown>,
  runtimeOverrides: Partial<RuntimeEventMeta> = {},
): SSEEventData {
  return {
    type,
    data: {
      runtime: runtime(runtimeOverrides),
      ...data,
    },
  } as SSEEventData
}

test('timelineToConversationItems should bind process items and final answer into one assistant turn', () => {
  const timeline = eventsToTimeline([
    eventOf('message', { role: 'user', message: '分析日志' }),
    eventOf('step', { id: 's-1', status: 'running', description: '读取日志' }),
    eventOf('tool', {
      tool_call_id: 'tool-1',
      name: 'file',
      function: 'read_file',
      args: { filepath: '.agentdocs/log/1.log' },
      status: 'called',
      content: 'ok',
    }, { current_step_id: 's-1' }),
    eventOf('message', { role: 'assistant', stage: 'intermediate', message: '我正在定位失败点。' }),
    eventOf('step', { id: 's-1', status: 'completed', description: '读取日志' }),
    eventOf('message', { role: 'assistant', stage: 'final', message: '结论如下。' }),
  ])

  const items = timelineToConversationItems(timeline)

  assert.equal(items.length, 2)
  assert.equal(items[0]?.kind, 'user')
  assert.equal(items[1]?.kind, 'assistant_turn')
  if (items[1]?.kind === 'assistant_turn') {
    assert.equal(items[1].status, 'completed')
    assert.equal(items[1].stepCount, 1)
    assert.equal(items[1].toolCount, 1)
    assert.equal(items[1].finalMessage?.data.message, '结论如下。')
    assert.equal(items[1].processItems.some((item) => item.kind === 'assistant'), true)
  }
})

test('timelineToConversationItems should keep virtual tool calls invisible through existing timeline filtering', () => {
  const timeline = eventsToTimeline([
    eventOf('message', { role: 'user', message: '继续使用之前的搜索结果' }),
    eventOf('step', { id: 's-1', status: 'running', description: '复用已有结果' }),
    eventOf('tool', {
      tool_call_id: 'tool-virtual',
      name: 'search',
      function: 'search_web',
      args: { query: 'same query' },
      status: 'called',
      is_virtual: true,
      virtual_kind: 'evidence_reuse',
      content: { message: '内部复用结果' },
    }, { current_step_id: 's-1' }),
    eventOf('step', { id: 's-1', status: 'completed', description: '复用已有结果' }),
    eventOf('message', { role: 'assistant', stage: 'final', message: '已基于已有结果回答。' }),
  ])

  const items = timelineToConversationItems(timeline)

  assert.equal(items.length, 2)
  assert.equal(items[1]?.kind, 'assistant_turn')
  if (items[1]?.kind === 'assistant_turn') {
    assert.equal(items[1].toolCount, 0)
    assert.equal(items[1].processItems.some((item) => item.kind === 'tool'), false)
  }
})
