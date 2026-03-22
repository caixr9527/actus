import assert from 'node:assert/strict'
import test from 'node:test'

import {
  adaptSessionEvent,
  normalizeEvent,
  normalizeEvents,
  unwrapNestedEvent,
  visitSessionEvent,
} from '../src/lib/session-event-adapter'
import type { SSEEventData } from '../src/lib/api/types'

function eventOf(type: SSEEventData['type'], data: unknown): SSEEventData {
  return { type, data } as SSEEventData
}

test('normalizeEvent should support both event/data and type/data', () => {
  const fromEvent = normalizeEvent({
    event: 'message',
    data: { role: 'assistant', message: 'hello' },
  })
  assert.deepEqual(fromEvent, {
    type: 'message',
    data: { role: 'assistant', message: 'hello' },
  })

  const fromType = normalizeEvent({
    type: 'done',
    data: {},
  })
  assert.deepEqual(fromType, {
    type: 'done',
    data: {},
  })

  assert.equal(
    normalizeEvent({
      event: 'unknown_event',
      data: {},
    }),
    null,
  )
})

test('normalizeEvents should filter invalid items', () => {
  const events = normalizeEvents([
    { event: 'message', data: { role: 'assistant', message: 'ok' } },
    { event: 'xxx', data: {} },
    { type: 'wait', data: {} },
    { type: 'step' },
  ])

  assert.equal(events.length, 2)
  assert.equal(events[0]?.type, 'message')
  assert.equal(events[1]?.type, 'wait')
})

test('unwrapNestedEvent should normalize nested payload event', () => {
  const wrapped = eventOf('message', {
    event: 'step',
    data: { id: 's-1', status: 'running', description: 'run' },
  })
  const unwrapped = unwrapNestedEvent(wrapped)
  assert.equal(unwrapped.type, 'step')
  assert.deepEqual(unwrapped.data, { id: 's-1', status: 'running', description: 'run' })
})

test('adaptSessionEvent should prefer compat metadata and parse runtime context', () => {
  const event = eventOf('step', {
    id: 's-1',
    status: 'running',
    description: 'run',
    event_status: 'started',
    extensions: {
      compat: {
        schema_version: 'be-lg-08.v2',
        semantic_type: 'step.started',
      },
      runtime: {
        session_id: 'sid',
        run_id: 'rid',
        channel: 'chat_stream',
      },
    },
  })

  const adapted = adaptSessionEvent(event)
  assert.equal(adapted.semanticType, 'step.started')
  assert.equal(adapted.compatSchemaVersion, 'be-lg-08.v2')
  assert.equal(adapted.runtimeContext?.session_id, 'sid')
  assert.equal(adapted.runtimeContext?.run_id, 'rid')
  assert.equal(adapted.runtimeContext?.channel, 'chat_stream')
})

test('adaptSessionEvent should fallback to legacy semantic type', () => {
  const event = eventOf('tool', {
    name: 'browser',
    function: 'browser_open',
    args: {},
    status: 'calling',
  })
  const adapted = adaptSessionEvent(event)
  assert.equal(adapted.semanticType, 'tool.calling')
  assert.equal(adapted.compatSchemaVersion, null)
  assert.equal(adapted.runtimeContext, null)
})

test('visitSessionEvent should dispatch by type and fallback to default', () => {
  const called: string[] = []

  visitSessionEvent(eventOf('done', {}), {
    done: () => called.push('done'),
    default: () => called.push('default'),
  })
  visitSessionEvent(eventOf('title', { title: 'abc' }), {
    done: () => called.push('done'),
    default: () => called.push('default'),
  })

  assert.deepEqual(called, ['done', 'default'])
})
