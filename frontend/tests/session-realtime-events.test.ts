import assert from 'node:assert/strict'
import test from 'node:test'

import {
  appendSessionRealtimeEvent,
  buildDuplicatePersistentEventCommit,
  buildSessionRealtimeStateFromSnapshot,
  shouldCommitDuplicatePersistentEvent,
} from '../src/lib/session-realtime-events'
import { eventsToTimeline } from '../src/lib/session-events'
import {
  classifyMessageStreamCloseReason,
  shouldReloadSnapshotAfterMessageStreamClose,
} from '../src/lib/session-detail-runtime'
import { sessionApi } from '../src/lib/api/session'
import { ApiError } from '../src/lib/api/fetch'
import {
  createRuntimeObservationFromSnapshot,
  reduceRuntimeObservationOnEvent,
} from '../src/lib/runtime-observation'
import type {
  RuntimeCapabilities,
  RuntimeEventMeta,
  RuntimeObservation,
  SSEEventData,
  SessionDetail,
} from '../src/lib/api/types'

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

function capabilities(overrides: Partial<RuntimeCapabilities> = {}): RuntimeCapabilities {
  return {
    can_send_message: false,
    can_resume: false,
    can_cancel: false,
    can_continue_cancelled: false,
    disabled_reasons: {},
    ...overrides,
  }
}

function observation(overrides: Partial<RuntimeObservation> = {}): RuntimeObservation {
  return {
    session_id: 'session-1',
    run_id: 'run-1',
    status: 'running',
    current_step_id: null,
    cursor: {
      latest_event_id: 'evt-1',
      has_more: false,
    },
    capabilities: capabilities({ can_cancel: true }),
    interaction: {
      kind: 'none',
      interrupt_id: null,
      payload: {},
    },
    sandbox_profile: null,
    ...overrides,
  }
}

function sessionDetail(runtimeObservation: RuntimeObservation, events: SSEEventData[] = []): SessionDetail {
  return {
    session_id: runtimeObservation.session_id,
    title: 'session',
    latest_message: '',
    latest_message_at: '',
    status: runtimeObservation.status,
    unread_message_count: 0,
    current_model_id: 'auto',
    runtime: runtimeObservation,
    events,
  }
}

function eventOf(type: SSEEventData['type'], data: Record<string, unknown>): SSEEventData {
  return { type, data } as SSEEventData
}

function messageEvent(eventId: string, message: string): SSEEventData {
  return eventOf('message', {
    runtime: runtime({
      source_event_id: eventId,
      cursor_event_id: eventId,
    }),
    event_id: eventId,
    role: 'assistant',
    message,
  })
}

function doneEvent(eventId: string): SSEEventData {
  return eventOf('done', {
    runtime: runtime({
      source_event_id: eventId,
      cursor_event_id: eventId,
      status_after_event: 'completed',
    }),
    event_id: eventId,
  })
}

function textStreamDelta(): SSEEventData {
  return eventOf('text_stream_delta', {
    runtime: runtime({
      source_event_id: null,
      cursor_event_id: null,
      durability: 'live_only',
      visibility: 'draft',
    }),
    event_id: null,
    stream_id: 'stream-1',
    channel: 'final_message',
    text: 'draft',
    sequence: 1,
  })
}

function textStreamStart(): SSEEventData {
  return eventOf('text_stream_start', {
    runtime: runtime({
      source_event_id: null,
      cursor_event_id: null,
      durability: 'live_only',
      visibility: 'draft',
    }),
    event_id: null,
    stream_id: 'stream-1',
    channel: 'final_message',
    stage: 'final',
  })
}

test('snapshot and SSE duplicate persistent event should appear only once', () => {
  const initial = buildSessionRealtimeStateFromSnapshot({
    rawEvents: [messageEvent('evt-1', 'from snapshot')],
    snapshotLatestEventId: 'evt-1',
  })

  const duplicate = appendSessionRealtimeEvent(initial, messageEvent('evt-1', 'from sse'))

  assert.equal(duplicate.duplicatePersistentEvent, true)
  assert.equal(duplicate.appendedToTimeline, false)
  assert.equal(duplicate.state.events.length, 1)
  assert.equal(duplicate.state.lastEventId, 'evt-1')
})

test('live-only text stream should not enter timeline or update persistent cursor', () => {
  const initial = buildSessionRealtimeStateFromSnapshot({
    rawEvents: [messageEvent('evt-1', 'persisted')],
    snapshotLatestEventId: 'evt-1',
  })

  const started = appendSessionRealtimeEvent(initial, textStreamStart())
  const next = appendSessionRealtimeEvent(started.state, textStreamDelta())

  assert.equal(next.duplicatePersistentEvent, false)
  assert.equal(next.appendedToTimeline, false)
  assert.equal(next.shouldUpdateDisplayedTextStreams, false)
  assert.equal(next.state.events.length, 1)
  assert.equal(next.state.lastEventId, 'evt-1')
  assert.deepEqual(Array.from(next.state.seenPersistentCursorIds), ['evt-1'])
  assert.equal(next.state.textStreams['stream-1']?.text, 'draft')
})

test('hidden feedback_input should not enter normal timeline but should keep persistent cursor', () => {
  const feedbackInputEvent = eventOf('feedback_input', {
    runtime: runtime({
      source_event_id: 'evt-feedback-input',
      cursor_event_id: 'evt-feedback-input',
      visibility: 'hidden',
    }),
    event_id: 'evt-feedback-input',
    payload: {
      source_action: 'final_satisfaction',
      intent_kind: 'dissatisfaction',
      target_ref: {
        target_type: 'message_event',
        target_id: 'evt-final',
        target_run_id: 'run-1',
      },
      reason_code: 'user_reported_dissatisfaction',
      sanitized_summary: '用户反馈上一轮答案不满意',
      input_hash: 'feedback_input:hash',
      runtime_metadata: {
        session_id: 'session-1',
        workspace_id: 'workspace-1',
        run_id: 'run-1',
      },
    },
  })

  const state = buildSessionRealtimeStateFromSnapshot({
    rawEvents: [messageEvent('evt-1', 'persisted'), feedbackInputEvent],
    snapshotLatestEventId: 'evt-feedback-input',
  })

  assert.equal(state.events.length, 2)
  assert.equal(state.lastEventId, 'evt-feedback-input')
  assert.deepEqual(Array.from(state.seenPersistentCursorIds), ['evt-1', 'evt-feedback-input'])
  assert.equal(eventsToTimeline(state.events).length, 1)
})

test('hidden feedback projection should dedupe by event id and replace live payload', () => {
  const feedbackEvent = (refs: string[]): SSEEventData => eventOf('feedback', {
    runtime: runtime({
      source_event_id: 'feedback:run-1:evt-source',
      cursor_event_id: 'feedback:run-1:evt-source',
      visibility: 'hidden',
    }),
    event_id: 'feedback:run-1:evt-source',
    payload: {
      feedback_refs: refs,
      counts: { feedback_count: refs.length },
      severity_counts: { error: refs.length },
      status_counts: { open: refs.length },
      kind_counts: { runtime_feedback: refs.length },
      summary: 'Feedback Ledger 投影',
      source_event_ids: ['evt-source'],
      runtime_metadata: {
        schema_version: 'feedback_event.v1',
        source_run_id: 'run-1',
        aggregation_key: 'evt-source',
        aggregation_kind: 'source_event',
      },
    },
  })
  const initial = buildSessionRealtimeStateFromSnapshot({
    rawEvents: [messageEvent('evt-1', 'persisted'), feedbackEvent(['fb-1'])],
    snapshotLatestEventId: 'feedback:run-1:evt-source',
  })

  const updated = appendSessionRealtimeEvent(initial, feedbackEvent(['fb-1', 'fb-2']))
  const projected = updated.state.events.find((event) => event.type === 'feedback')

  assert.equal(updated.duplicatePersistentEvent, true)
  assert.equal(updated.appendedToTimeline, false)
  assert.equal(updated.state.events.length, 2)
  assert.equal(eventsToTimeline(updated.state.events).length, 1)
  assert.deepEqual(
    projected?.type === 'feedback' ? projected.data.payload.feedback_refs : [],
    ['fb-1', 'fb-2'],
  )
  assert.equal(shouldCommitDuplicatePersistentEvent(updated.event), true)
  assert.equal(shouldCommitDuplicatePersistentEvent(messageEvent('evt-1', 'duplicate')), false)

  let committedEvents: SSEEventData[] = []
  const commit = buildDuplicatePersistentEventCommit(updated)
  if (commit) committedEvents = commit.events
  assert.deepEqual(
    committedEvents.find((event) => event.type === 'feedback')?.type === 'feedback'
      ? committedEvents.find((event) => event.type === 'feedback')?.data.payload.feedback_refs
      : [],
    ['fb-1', 'fb-2'],
  )
  assert.equal(eventsToTimeline(committedEvents).length, 1)
  assert.equal(buildDuplicatePersistentEventCommit(appendSessionRealtimeEvent(initial, messageEvent('evt-1', 'duplicate'))), null)
})

test('snapshot state should use last persistent event over runtime cursor and ignore text streams', () => {
  const state = buildSessionRealtimeStateFromSnapshot({
    rawEvents: [
      messageEvent('evt-1', 'persisted'),
      textStreamDelta(),
    ],
    snapshotLatestEventId: 'evt-runtime-cursor',
  })

  assert.equal(state.events.length, 1)
  assert.equal(state.lastEventId, 'evt-1')
  assert.deepEqual(Array.from(state.seenPersistentCursorIds), ['evt-1'])
})

test('message stream end should reload snapshot and use snapshot runtime status as truth', () => {
  const reason = classifyMessageStreamCloseReason(new Error('SSE_STREAM_END'))
  assert.equal(shouldReloadSnapshotAfterMessageStreamClose(reason), true)

  const snapshot = sessionDetail(observation({
    status: 'completed',
    capabilities: capabilities({ can_send_message: true }),
    cursor: {
      latest_event_id: 'evt-done',
      has_more: false,
    },
  }), [messageEvent('evt-final', 'answer'), doneEvent('evt-done')])

  const runtimeState = createRuntimeObservationFromSnapshot(snapshot)
  const realtimeState = buildSessionRealtimeStateFromSnapshot({
    rawEvents: snapshot.events,
    snapshotLatestEventId: snapshot.runtime.cursor.latest_event_id,
  })

  assert.equal(runtimeState.status, 'completed')
  assert.equal(runtimeState.capabilities.can_send_message, true)
  assert.equal(realtimeState.lastEventId, 'evt-done')
})

test('document input preflight rejection should not create timeline event or persistent cursor', async () => {
  const initialRealtimeState = buildSessionRealtimeStateFromSnapshot({
    rawEvents: [messageEvent('evt-1', 'persisted')],
    snapshotLatestEventId: 'evt-1',
  })
  let eventCount = 0
  let capturedError: unknown = null

  const originalFetch = globalThis.fetch
  globalThis.fetch = (async () => new Response(JSON.stringify({
    code: 400,
    msg: '不支持该类型作为任务输入',
    data: null,
    error_key: 'error.document_input.unsupported_media_image',
    error_params: {
      reason_code: 'unsupported_media_image',
    },
  }), {
    status: 400,
    headers: {
      'Content-Type': 'application/json',
    },
  })) as typeof fetch

  try {
    const handle = sessionApi.openChatStream(
      'session-1',
      { message: '分析图片', attachments: ['file-image'] },
      () => {
        eventCount += 1
      },
      (error) => {
        capturedError = error
      },
    )

    await assert.rejects(handle.ready, (error: unknown) => {
      assert.ok(error instanceof ApiError)
      assert.equal(error.errorKey, 'error.document_input.unsupported_media_image')
      assert.deepEqual(error.errorParams, { reason_code: 'unsupported_media_image' })
      return true
    })

    assert.equal(eventCount, 0)
    assert.ok(capturedError instanceof ApiError)
    assert.equal(initialRealtimeState.events.length, 1)
    assert.equal(initialRealtimeState.lastEventId, 'evt-1')
    assert.deepEqual(Array.from(initialRealtimeState.seenPersistentCursorIds), ['evt-1'])
    handle.cleanup()
  } finally {
    globalThis.fetch = originalFetch
  }
})

test('waiting resume running final done chain should follow runtime capabilities and cursor', () => {
  let runtimeState = createRuntimeObservationFromSnapshot(sessionDetail(observation({
    status: 'waiting',
    capabilities: capabilities({ can_resume: true, can_cancel: true }),
    interaction: {
      kind: 'wait',
      interrupt_id: 'interrupt-1',
      payload: { kind: 'confirm', prompt: '继续？' },
    },
    cursor: {
      latest_event_id: 'evt-wait',
      has_more: false,
    },
  })))
  let realtimeState = buildSessionRealtimeStateFromSnapshot({
    rawEvents: [],
    snapshotLatestEventId: 'evt-wait',
  })

  assert.equal(runtimeState.status, 'waiting')
  assert.equal(runtimeState.capabilities.can_resume, true)
  assert.equal(runtimeState.capabilities.can_cancel, true)
  assert.equal(realtimeState.lastEventId, 'evt-wait')

  const resumed = messageEvent('evt-resume', 'resumed')
  resumed.data.runtime.status_after_event = 'running'
  runtimeState = reduceRuntimeObservationOnEvent(runtimeState, resumed)
  realtimeState = appendSessionRealtimeEvent(realtimeState, resumed).state
  assert.equal(runtimeState.status, 'running')
  assert.equal(runtimeState.capabilities.can_cancel, true)
  assert.equal(runtimeState.capabilities.can_resume, false)
  assert.equal(realtimeState.lastEventId, 'evt-resume')

  const final = messageEvent('evt-final', 'final answer')
  runtimeState = reduceRuntimeObservationOnEvent(runtimeState, final)
  realtimeState = appendSessionRealtimeEvent(realtimeState, final).state
  assert.equal(runtimeState.status, 'running')
  assert.equal(realtimeState.lastEventId, 'evt-final')

  const done = doneEvent('evt-done')
  runtimeState = reduceRuntimeObservationOnEvent(runtimeState, done)
  realtimeState = appendSessionRealtimeEvent(realtimeState, done).state
  assert.equal(runtimeState.status, 'completed')
  assert.equal(runtimeState.capabilities.can_send_message, true)
  assert.equal(runtimeState.capabilities.can_cancel, false)
  assert.equal(realtimeState.lastEventId, 'evt-done')
})
