import assert from 'node:assert/strict'
import test from 'node:test'

import { sessionApi } from '../src/lib/api/session'
import { translateRuntime } from '../src/lib/i18n/runtime'

test('chat request should omit feedback_intent when not provided', async () => {
  const originalFetch = globalThis.fetch
  const requests: Array<{ url: string; body: unknown }> = []

  globalThis.fetch = (async (input, init) => {
    requests.push({
      url: String(input),
      body: init?.body ? JSON.parse(String(init.body)) : null,
    })
    return new Response(new ReadableStream<Uint8Array>({
      start(controller) {
        controller.close()
      },
    }), {
      status: 200,
      headers: {
        'Content-Type': 'text/event-stream',
      },
    })
  }) as typeof fetch

  try {
    const handle = sessionApi.openChatStream('session-1', { message: 'hello' }, () => undefined)
    await handle.ready
    handle.cleanup()
  } finally {
    globalThis.fetch = originalFetch
  }

  assert.equal(requests.length, 1)
  assert.equal(requests[0].url.endsWith('/sessions/session-1/chat'), true)
  assert.deepEqual(requests[0].body, { message: 'hello' })
  assert.equal(Object.prototype.hasOwnProperty.call(requests[0].body as object, 'feedback_intent'), false)
})

test('submitFeedback should post structured final message target', async () => {
  const originalFetch = globalThis.fetch
  let capturedBody: unknown = null

  globalThis.fetch = (async (_input, init) => {
    capturedBody = init?.body ? JSON.parse(String(init.body)) : null
    return new Response(JSON.stringify({
      code: 0,
      msg: 'ok',
      data: { success: true },
    }), {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
      },
    })
  }) as typeof fetch

  try {
    await sessionApi.submitFeedback('session-1', {
      source_action: 'final_satisfaction',
      intent_kind: 'dissatisfaction',
      target_ref: {
        target_type: 'message_event',
        target_id: 'evt-final-1',
        target_run_id: 'run-1',
      },
      reason_code: 'user_reported_dissatisfaction',
    })
  } finally {
    globalThis.fetch = originalFetch
  }

  assert.deepEqual(capturedBody, {
    source_action: 'final_satisfaction',
    intent_kind: 'dissatisfaction',
    target_ref: {
      target_type: 'message_event',
      target_id: 'evt-final-1',
      target_run_id: 'run-1',
    },
    reason_code: 'user_reported_dissatisfaction',
  })
})

test('submitFeedback should post structured artifact revision target', async () => {
  const originalFetch = globalThis.fetch
  let capturedBody: unknown = null

  globalThis.fetch = (async (_input, init) => {
    capturedBody = init?.body ? JSON.parse(String(init.body)) : null
    return new Response(JSON.stringify({
      code: 0,
      msg: 'ok',
      data: { success: true },
    }), {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
      },
    })
  }) as typeof fetch

  try {
    await sessionApi.submitFeedback('session-1', {
      source_action: 'artifact_satisfaction',
      intent_kind: 'correction',
      target_ref: {
        target_type: 'artifact_revision',
        target_id: 'artifact-1',
        target_revision_id: 'rev-1',
        target_content_hash: 'sha256:abc',
        target_run_id: 'run-1',
      },
      reason_code: 'user_corrected_requirement',
    })
  } finally {
    globalThis.fetch = originalFetch
  }

  assert.deepEqual(capturedBody, {
    source_action: 'artifact_satisfaction',
    intent_kind: 'correction',
    target_ref: {
      target_type: 'artifact_revision',
      target_id: 'artifact-1',
      target_revision_id: 'rev-1',
      target_content_hash: 'sha256:abc',
      target_run_id: 'run-1',
    },
    reason_code: 'user_corrected_requirement',
  })
})

test('feedback labels should come from i18n dictionaries', () => {
  assert.equal(translateRuntime('feedback.finalDissatisfied', undefined, 'zh-CN'), '不满意')
  assert.equal(translateRuntime('feedback.artifactIncorrect', undefined, 'en-US'), 'Content is wrong')
})
