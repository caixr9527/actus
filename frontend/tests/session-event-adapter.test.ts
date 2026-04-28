import assert from 'node:assert/strict'
import test from 'node:test'

import {
  assertSessionDetailRuntime,
  normalizeSessionDetailRuntimeStatus,
  normalizeEvent,
  normalizeEvents,
  unwrapNestedEvent,
  visitSessionEvent,
} from '../src/lib/session-event-adapter'
import type { SSEEventData } from '../src/lib/api/types'

const runtime = {
  session_id: 'session-1',
  run_id: 'run-1',
  status_after_event: null,
  current_step_id: null,
  source_event_id: 'evt-1',
  cursor_event_id: 'evt-1',
  durability: 'persistent',
  visibility: 'timeline',
} as const

function eventOf(type: SSEEventData['type'], data: unknown): SSEEventData {
  return { type, data } as SSEEventData
}

test('normalizeEvent should support both event/data and type/data', () => {
  const fromEvent = normalizeEvent({
    event: 'message',
    data: { runtime, role: 'assistant', message: 'hello' },
  })
  assert.deepEqual(fromEvent, {
    type: 'message',
    data: { runtime, role: 'assistant', message: 'hello' },
  })

  const fromType = normalizeEvent({
    type: 'done',
    data: { runtime },
  })
  assert.deepEqual(fromType, {
    type: 'done',
    data: { runtime },
  })

  assert.equal(
    normalizeEvent({
      event: 'unknown_event',
      data: {},
    }),
    null,
  )
})

test('normalizeEvent should reject known events without runtime metadata', () => {
  assert.equal(
    normalizeEvent({
      event: 'message',
      data: { role: 'assistant', message: 'hello' },
    }),
    null,
  )
})

test('assertSessionDetailRuntime should reject detail snapshots without runtime', () => {
  assert.throws(
    () => assertSessionDetailRuntime({
      session_id: 'session-1',
      title: 'missing runtime',
      latest_message: '',
      latest_message_at: '',
      status: 'running',
      unread_message_count: 0,
      events: [],
    } as never),
    /SESSION_DETAIL_RUNTIME_CONTRACT_MISSING/,
  )
})

test('normalizeSessionDetailRuntimeStatus should mirror top-level status from runtime status', () => {
  const detail = normalizeSessionDetailRuntimeStatus({
    session_id: 'session-1',
    title: 'runtime wins',
    latest_message: '',
    latest_message_at: '',
    status: 'running',
    unread_message_count: 0,
    current_model_id: 'auto',
    runtime: {
      session_id: 'session-1',
      run_id: 'run-1',
      status: 'waiting',
      current_step_id: null,
      cursor: {
        latest_event_id: 'evt-1',
        has_more: false,
      },
      capabilities: {
        can_send_message: false,
        can_resume: true,
        can_cancel: false,
        can_continue_cancelled: false,
        disabled_reasons: {},
      },
      interaction: {
        kind: 'wait',
        interrupt_id: 'interrupt-1',
        payload: {},
      },
    },
    events: [],
  })

  assert.equal(detail.status, 'waiting')
  assert.equal(detail.runtime.status, 'waiting')
})

test('normalizeEvents should filter invalid items', () => {
  const events = normalizeEvents([
    { event: 'message', data: { runtime, role: 'assistant', message: 'ok' } },
    { event: 'xxx', data: {} },
    { type: 'wait', data: { runtime } },
    { type: 'step' },
    { type: 'done', data: {} },
  ])

  assert.equal(events.length, 2)
  assert.equal(events[0]?.type, 'message')
  assert.equal(events[1]?.type, 'wait')
})

test('unwrapNestedEvent should normalize nested payload event', () => {
  const wrapped = eventOf('message', {
    event: 'step',
    data: { runtime, id: 's-1', status: 'running', description: 'run' },
  })
  const unwrapped = unwrapNestedEvent(wrapped)
  assert.equal(unwrapped.type, 'step')
  assert.deepEqual(unwrapped.data, { runtime, id: 's-1', status: 'running', description: 'run' })
})

test('visitSessionEvent should dispatch by type and fallback to default', () => {
  const called: string[] = []

  visitSessionEvent(eventOf('done', { runtime }), {
    done: () => called.push('done'),
    default: () => called.push('default'),
  })
  visitSessionEvent(eventOf('title', { runtime, title: 'abc' }), {
    done: () => called.push('done'),
    default: () => called.push('default'),
  })

  assert.deepEqual(called, ['done', 'default'])
})
