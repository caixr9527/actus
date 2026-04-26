import assert from 'node:assert/strict'
import test from 'node:test'

import {
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
