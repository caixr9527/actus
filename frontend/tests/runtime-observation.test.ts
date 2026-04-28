import assert from 'node:assert/strict'
import test from 'node:test'

import {
  collectPersistentCursorEventIds,
  createRuntimeObservationFromSnapshot,
  getPersistentCursorEventId,
  reduceRuntimeObservationOnEvent,
  reduceSessionRuntimeStateOnEvent,
} from '../src/lib/runtime-observation'
import type {
  RuntimeEventMeta,
  RuntimeObservation,
  SSEEventData,
  SessionDetail,
  SessionStatus,
} from '../src/lib/api/types'

function runtimeMeta(overrides: Partial<RuntimeEventMeta> = {}): RuntimeEventMeta {
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

function runtimeObservation(overrides: Partial<RuntimeObservation> = {}): RuntimeObservation {
  return {
    session_id: 'session-1',
    run_id: 'run-1',
    status: 'running',
    current_step_id: 'step-1',
    cursor: {
      latest_event_id: 'evt-1',
      has_more: false,
    },
    capabilities: {
      can_send_message: false,
      can_resume: false,
      can_cancel: true,
      can_continue_cancelled: false,
      disabled_reasons: {},
    },
    interaction: {
      kind: 'none',
      interrupt_id: null,
      payload: {},
    },
    ...overrides,
  }
}

function sessionDetail(runtime: RuntimeObservation): SessionDetail {
  return {
    session_id: runtime.session_id,
    title: 'runtime snapshot',
    latest_message: '',
    latest_message_at: '',
    status: 'completed',
    unread_message_count: 0,
    current_model_id: 'auto',
    runtime,
    events: [],
  }
}

function eventOf(type: SSEEventData['type'], data: Record<string, unknown>): SSEEventData {
  return { type, data } as SSEEventData
}

test('createRuntimeObservationFromSnapshot should use detail.runtime as the source of truth', () => {
  const observation = runtimeObservation({
    status: 'waiting',
    current_step_id: null,
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
      payload: { prompt: 'confirm' },
    },
  })

  const state = createRuntimeObservationFromSnapshot(sessionDetail(observation))

  assert.equal(state.status, 'waiting')
  assert.equal(state.sessionId, 'session-1')
  assert.equal(state.capabilities.can_resume, true)
  assert.equal(state.interaction.kind, 'wait')
})

test('reduceRuntimeObservationOnEvent should update status only from runtime.status_after_event', () => {
  const prev = createRuntimeObservationFromSnapshot(sessionDetail(runtimeObservation()))
  const doneWithoutRuntimeStatus = eventOf('done', {
    runtime: runtimeMeta({ status_after_event: null, cursor_event_id: 'evt-2' }),
  })
  const afterDone = reduceRuntimeObservationOnEvent(prev, doneWithoutRuntimeStatus)
  assert.equal(afterDone.status, 'running')
  assert.equal(afterDone.cursor.latest_event_id, 'evt-2')

  const waitEvent = eventOf('wait', {
    runtime: runtimeMeta({
      status_after_event: 'waiting',
      current_step_id: null,
      cursor_event_id: 'evt-3',
    }),
  })
  const afterWait = reduceRuntimeObservationOnEvent(afterDone, waitEvent)
  assert.equal(afterWait.status, 'waiting')
  assert.equal(afterWait.currentStepId, null)
  assert.equal(afterWait.cursor.latest_event_id, 'evt-3')
  assert.equal(afterWait.capabilities.can_resume, true)
  assert.equal(afterWait.capabilities.can_cancel, true)
  assert.equal(afterWait.interaction.kind, 'wait')
})

test('reduceRuntimeObservationOnEvent should project capabilities from explicit runtime status', () => {
  const prev = createRuntimeObservationFromSnapshot(sessionDetail(runtimeObservation({
    status: 'waiting',
    capabilities: {
      can_send_message: false,
      can_resume: true,
      can_cancel: true,
      can_continue_cancelled: false,
      disabled_reasons: {},
    },
    interaction: {
      kind: 'wait',
      interrupt_id: 'interrupt-1',
      payload: { prompt: 'confirm' },
    },
  })))

  const running = reduceRuntimeObservationOnEvent(prev, eventOf('message', {
    runtime: runtimeMeta({ status_after_event: 'running' }),
    role: 'assistant',
    message: 'resumed',
  }))

  assert.equal(running.status, 'running')
  assert.equal(running.capabilities.can_cancel, true)
  assert.equal(running.capabilities.can_resume, false)
  assert.equal(running.interaction.kind, 'none')

  const completed = reduceRuntimeObservationOnEvent(running, eventOf('done', {
    runtime: runtimeMeta({ status_after_event: 'completed' }),
  }))

  assert.equal(completed.status, 'completed')
  assert.equal(completed.capabilities.can_send_message, true)
  assert.equal(completed.capabilities.can_cancel, false)
})

test('reduceRuntimeObservationOnEvent should preserve coordinator-owned cancelled continue capability', () => {
  const cancellable = createRuntimeObservationFromSnapshot(sessionDetail(runtimeObservation({
    status: 'running',
    capabilities: {
      can_send_message: false,
      can_resume: false,
      can_cancel: true,
      can_continue_cancelled: true,
      disabled_reasons: {},
    },
  })))
  const notContinuable = createRuntimeObservationFromSnapshot(sessionDetail(runtimeObservation({
    status: 'running',
    capabilities: {
      can_send_message: false,
      can_resume: false,
      can_cancel: true,
      can_continue_cancelled: false,
      disabled_reasons: {},
    },
  })))

  const cancelledEvent = eventOf('done', {
    runtime: runtimeMeta({ status_after_event: 'cancelled' }),
  })

  assert.equal(
    reduceRuntimeObservationOnEvent(cancellable, cancelledEvent).capabilities.can_continue_cancelled,
    true,
  )
  assert.equal(
    reduceRuntimeObservationOnEvent(notContinuable, cancelledEvent).capabilities.can_continue_cancelled,
    false,
  )
})

test('reduceSessionRuntimeStateOnEvent should not infer status from event type', () => {
  const events: SSEEventData[] = [
    eventOf('step', { runtime: runtimeMeta(), status: 'running' }),
    eventOf('wait', { runtime: runtimeMeta() }),
    eventOf('done', { runtime: runtimeMeta() }),
    eventOf('error', { runtime: runtimeMeta(), error: 'boom' }),
    eventOf('plan', { runtime: runtimeMeta(), steps: [] }),
  ]

  for (const event of events) {
    const state = reduceSessionRuntimeStateOnEvent(
      { status: 'running', streaming: true },
      event,
    )
    assert.deepEqual(state, { status: 'running', streaming: true })
  }
})

test('reduceSessionRuntimeStateOnEvent should apply explicit status_after_event', () => {
  const statuses: Array<[SessionStatus, boolean]> = [
    ['running', true],
    ['waiting', false],
    ['completed', false],
    ['failed', false],
    ['cancelled', false],
  ]

  for (const [status, streaming] of statuses) {
    const state = reduceSessionRuntimeStateOnEvent(
      { status: 'running', streaming: true },
      eventOf('done', {
        runtime: runtimeMeta({ status_after_event: status }),
      }),
    )
    assert.deepEqual(state, { status, streaming })
  }
})

test('persistent cursor helpers should only read runtime.cursor_event_id from persistent events', () => {
  const persistent = eventOf('message', {
    runtime: runtimeMeta({ cursor_event_id: ' evt-1 ' }),
    role: 'assistant',
    message: 'ok',
    event_id: 'legacy-ignored',
  })
  const liveOnly = eventOf('text_stream_delta', {
    runtime: runtimeMeta({
      source_event_id: null,
      cursor_event_id: null,
      durability: 'live_only',
      visibility: 'draft',
    }),
    stream_id: 'stream-1',
    channel: 'final_message',
    text: 'draft',
    sequence: 1,
  })

  assert.equal(getPersistentCursorEventId(persistent), 'evt-1')
  assert.equal(getPersistentCursorEventId(liveOnly), null)
  assert.deepEqual(Array.from(collectPersistentCursorEventIds([persistent, liveOnly])), ['evt-1'])
})
