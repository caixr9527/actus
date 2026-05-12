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
import type { RuntimeEventMeta, SSEEventData, SessionStatus } from '../src/lib/api/types'

function runtime(overrides: Partial<RuntimeEventMeta> = {}): RuntimeEventMeta {
  return {
    session_id: 'session-1',
    run_id: 'run-1',
    status_after_event: null,
    current_step_id: null,
    source_event_id: 'evt-1',
    cursor_event_id: 'evt-1',
    durability: 'persistent',
    visibility: 'timeline',
    ...overrides,
  }
}

function eventOf(type: SSEEventData['type'], data: Record<string, unknown>): SSEEventData {
  return { type, data } as SSEEventData
}

test('reduceSessionRuntimeStateOnEvent should not infer state from event type', () => {
  const sequence: SSEEventData[] = [
    eventOf('step', { runtime: runtime(), status: 'running' }),
    eventOf('tool', { runtime: runtime(), function: 'message_ask_user', status: 'calling' }),
    eventOf('wait', { runtime: runtime() }),
    eventOf('done', { runtime: runtime() }),
    eventOf('error', { runtime: runtime(), error: 'boom' }),
    eventOf('plan', { runtime: runtime(), steps: [] }),
  ]

  let state: SessionRuntimeState = { status: 'running', streaming: true }
  for (const ev of sequence) {
    state = reduceSessionRuntimeStateOnEvent(state, ev)
  }

  assert.deepEqual(state, { status: 'running', streaming: true })
})

test('reduceSessionRuntimeStateOnEvent should consume runtime.status_after_event', () => {
  const cases: Array<[SSEEventData['type'], SessionStatus, boolean]> = [
    ['wait', 'waiting', false],
    ['done', 'completed', false],
    ['error', 'failed', false],
    ['step', 'cancelled', false],
    ['message', 'running', true],
  ]

  for (const [type, status, streaming] of cases) {
    const next = reduceSessionRuntimeStateOnEvent(
      { status: 'running', streaming: true },
      eventOf(type, {
        runtime: runtime({ status_after_event: status }),
        ...(type === 'message' ? { role: 'assistant', message: 'ok' } : {}),
      }),
    )

    assert.deepEqual(next, { status, streaming })
  }
})

test('getSessionEventId and collectSessionEventIds should use persistent runtime cursor only', () => {
  const events = [
    eventOf('message', {
      runtime: runtime({ cursor_event_id: ' evt-1 ' }),
      event_id: 'legacy-ignored',
      role: 'assistant',
      message: 'ok',
    }),
    eventOf('step', {
      runtime: runtime({ cursor_event_id: 'evt-2' }),
      event_id: 'legacy-step-ignored',
      status: 'running',
      id: 'step-1',
      description: 'run',
    }),
    eventOf('done', { runtime: runtime({ cursor_event_id: null }) }),
    eventOf('text_stream_delta', {
      runtime: runtime({
        source_event_id: null,
        cursor_event_id: null,
        durability: 'live_only',
        visibility: 'draft',
      }),
      event_id: 'live-only-ignored',
      stream_id: 'stream-1',
      channel: 'final_message',
      text: 'draft',
      sequence: 1,
    }),
    eventOf('message', {
      runtime: runtime({ cursor_event_id: 'evt-1' }),
      role: 'assistant',
      message: 'duplicate',
    }),
  ]

  assert.equal(getSessionEventId(events[0]), 'evt-1')
  assert.equal(getSessionEventId(events[2]), null)
  assert.equal(getSessionEventId(events[3]), null)
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
