import assert from 'node:assert/strict'
import test from 'node:test'

import {
  classifyMessageStreamCloseReason,
  collectSessionEventIds,
  getSessionEventId,
  reduceSessionRuntimeStateOnEvent,
  shouldReloadSnapshotAfterMessageStreamClose,
  type SessionRuntimeState,
} from '../src/lib/session-detail-runtime'
import type { SSEEventData } from '../src/lib/api/types'

function eventOf(type: SSEEventData['type'], data: unknown): SSEEventData {
  return { type, data } as SSEEventData
}

test('reduceSessionRuntimeStateOnEvent should process a complex event sequence', () => {
  const sequence: SSEEventData[] = [
    eventOf('step', { status: 'running' }),
    eventOf('tool', { function: 'message_ask_user', status: 'calling' }),
    eventOf('step', { status: 'running' }),
    eventOf('wait', {}),
    eventOf('done', {}),
  ]

  let state: SessionRuntimeState = { status: 'pending', streaming: true }
  for (const ev of sequence) {
    state = reduceSessionRuntimeStateOnEvent(state, ev)
  }

  assert.deepEqual(state, { status: 'completed', streaming: false })
})

test('reduceSessionRuntimeStateOnEvent should not switch to waiting on ask-user tool calling event alone', () => {
  const event = eventOf('tool', {
    function: 'message_ask_user',
    status: 'calling',
  })

  const next = reduceSessionRuntimeStateOnEvent(
    { status: 'running', streaming: true },
    event,
  )
  assert.deepEqual(next, { status: 'running', streaming: true })
})

test('reduceSessionRuntimeStateOnEvent should switch to cancelled on cancelled step event', () => {
  const next = reduceSessionRuntimeStateOnEvent(
    { status: 'running', streaming: true },
    eventOf('step', { status: 'cancelled' }),
  )

  assert.deepEqual(next, { status: 'cancelled', streaming: false })
})

test('reduceSessionRuntimeStateOnEvent should switch to cancelled on cancelled plan event', () => {
  const next = reduceSessionRuntimeStateOnEvent(
    { status: 'running', streaming: true },
    eventOf('plan', { plan_status: 'cancelled', steps: [] }),
  )

  assert.deepEqual(next, { status: 'cancelled', streaming: false })
})

test('reduceSessionRuntimeStateOnEvent should keep cancelled as terminal status', () => {
  const next = reduceSessionRuntimeStateOnEvent(
    { status: 'cancelled', streaming: false },
    eventOf('step', { status: 'running' }),
  )

  assert.deepEqual(next, { status: 'cancelled', streaming: false })
})

test('getSessionEventId and collectSessionEventIds should normalize event ids for dedupe', () => {
  const events = [
    eventOf('message', { event_id: ' evt-1 ' }),
    eventOf('step', { event_id: 'evt-2', status: 'running' }),
    eventOf('done', {}),
    eventOf('message', { event_id: 'evt-1' }),
  ]

  assert.equal(getSessionEventId(events[0]), 'evt-1')
  assert.equal(getSessionEventId(events[2]), null)
  assert.deepEqual(Array.from(collectSessionEventIds(events)).sort(), ['evt-1', 'evt-2'])
})

test('classifyMessageStreamCloseReason should map abort, stream end and generic errors', () => {
  const abortError = new Error('aborted')
  abortError.name = 'AbortError'
  assert.equal(classifyMessageStreamCloseReason(abortError), 'abort')

  assert.equal(classifyMessageStreamCloseReason(new Error('SSE_STREAM_END')), 'stream_end')
  assert.equal(classifyMessageStreamCloseReason(new Error('socket reset')), 'error')
})

test('shouldReloadSnapshotAfterMessageStreamClose should only reload for stream_end/error', () => {
  assert.equal(shouldReloadSnapshotAfterMessageStreamClose('done'), false)
  assert.equal(shouldReloadSnapshotAfterMessageStreamClose('abort'), false)
  assert.equal(shouldReloadSnapshotAfterMessageStreamClose('stream_end'), true)
  assert.equal(shouldReloadSnapshotAfterMessageStreamClose('error'), true)
})
