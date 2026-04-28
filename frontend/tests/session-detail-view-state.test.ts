import assert from 'node:assert/strict'
import test from 'node:test'

import {
  createSessionScopedDetailViewState,
  createSessionScopedRuntimeState,
  resolveSessionActionAvailability,
  resolveWaitResumeContext,
  resolveStepExpandedState,
  shouldAutoCloseTaskPreview,
  shouldAutoExpandStep,
  shouldAutoScrollToLatest,
  shouldHideWaitResumeCard,
  shouldResetWaitResumePending,
  shouldShowSessionThinking,
} from '../src/lib/session-detail-view-state'
import type { RuntimeCapabilities, RuntimeEventMeta, RuntimeInteraction, SSEEventData } from '../src/lib/api/types'

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

function waitEvent(payloadPrompt: string): SSEEventData {
  return {
    type: 'wait',
    data: {
      runtime: runtime({ status_after_event: 'waiting' }),
      interrupt_id: 'interrupt-event',
      payload: {
        kind: 'input_text',
        prompt: payloadPrompt,
        response_key: 'message',
      },
    },
  } as SSEEventData
}

function waitInteraction(prompt: string): RuntimeInteraction {
  return {
    kind: 'wait',
    interrupt_id: 'interrupt-runtime',
    payload: {
      kind: 'input_text',
      prompt,
      response_key: 'message',
    },
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

test('shouldAutoExpandStep should only expand running steps by default', () => {
  assert.equal(shouldAutoExpandStep('running'), true)
  assert.equal(shouldAutoExpandStep('completed'), false)
  assert.equal(shouldAutoExpandStep('failed'), false)
  assert.equal(shouldAutoExpandStep('pending'), false)
})

test('resolveStepExpandedState should auto expand running step and collapse when it finishes', () => {
  assert.equal(resolveStepExpandedState({
    currentExpanded: false,
    previousStatus: null,
    nextStatus: 'running',
  }), true)

  assert.equal(resolveStepExpandedState({
    currentExpanded: true,
    previousStatus: 'running',
    nextStatus: 'completed',
  }), false)

  assert.equal(resolveStepExpandedState({
    currentExpanded: true,
    previousStatus: 'running',
    nextStatus: 'failed',
  }), false)

  assert.equal(resolveStepExpandedState({
    currentExpanded: true,
    previousStatus: 'completed',
    nextStatus: 'completed',
  }), true)
})

test('shouldAutoCloseTaskPreview should only close preview when running task completes', () => {
  assert.equal(shouldAutoCloseTaskPreview('running', 'completed'), true)
  assert.equal(shouldAutoCloseTaskPreview('running', 'failed'), true)
  assert.equal(shouldAutoCloseTaskPreview('running', 'cancelled'), true)
  assert.equal(shouldAutoCloseTaskPreview('running', 'waiting'), false)
  assert.equal(shouldAutoCloseTaskPreview('completed', 'completed'), false)
  assert.equal(shouldAutoCloseTaskPreview(null, 'completed'), false)
})

test('shouldAutoScrollToLatest should auto scroll once per session when content exists', () => {
  assert.equal(shouldAutoScrollToLatest({
    hasAutoScrolled: false,
    timelineLength: 3,
    shouldShowThinking: false,
  }), true)

  assert.equal(shouldAutoScrollToLatest({
    hasAutoScrolled: true,
    timelineLength: 3,
    shouldShowThinking: true,
  }), false)

  assert.equal(shouldAutoScrollToLatest({
    hasAutoScrolled: false,
    timelineLength: 0,
    shouldShowThinking: true,
  }), true)

  assert.equal(shouldAutoScrollToLatest({
    hasAutoScrolled: false,
    timelineLength: 0,
    shouldShowThinking: false,
  }), false)
})

test('session scoped detail state factories should reset view state and runtime refs explicitly', () => {
  assert.deepEqual(createSessionScopedDetailViewState(), {
    fileListOpen: false,
    previewFile: null,
    previewTool: null,
    timelineExpanded: false,
    vncOpen: false,
  })

  assert.deepEqual(createSessionScopedRuntimeState(), {
    initialMessageSent: false,
    previousToolCount: 0,
    hasAutoScrolled: false,
    previousSessionStatus: null,
  })
})

test('resolveSessionActionAvailability should map UI actions only from runtime capabilities', () => {
  assert.deepEqual(resolveSessionActionAvailability(null), {
    canSendMessage: false,
    canResume: false,
    canCancel: false,
    canContinueCancelled: false,
  })

  assert.deepEqual(resolveSessionActionAvailability(capabilities({
    can_send_message: true,
  })), {
    canSendMessage: true,
    canResume: false,
    canCancel: false,
    canContinueCancelled: false,
  })

  assert.deepEqual(resolveSessionActionAvailability(capabilities({
    can_resume: true,
    can_cancel: true,
  })), {
    canSendMessage: false,
    canResume: true,
    canCancel: true,
    canContinueCancelled: false,
  })

  assert.deepEqual(resolveSessionActionAvailability(capabilities({
    can_continue_cancelled: true,
  })), {
    canSendMessage: false,
    canResume: false,
    canCancel: false,
    canContinueCancelled: true,
  })
})

test('resolveSessionActionAvailability should not infer actions from runtime status', () => {
  const statusOnly = capabilities()

  assert.deepEqual(resolveSessionActionAvailability(statusOnly), {
    canSendMessage: false,
    canResume: false,
    canCancel: false,
    canContinueCancelled: false,
  })
})

test('shouldShowSessionThinking should hide thinking while any step is running', () => {
  assert.equal(shouldShowSessionThinking({
    streaming: true,
    sessionStatus: 'running',
    hasInitialMessage: false,
    timelineLength: 5,
    hasError: false,
    hasRunningStep: true,
  }), false)

  assert.equal(shouldShowSessionThinking({
    streaming: false,
    sessionStatus: 'running',
    hasInitialMessage: false,
    timelineLength: 5,
    hasError: false,
    hasRunningStep: false,
  }), true)

  assert.equal(shouldShowSessionThinking({
    streaming: false,
    sessionStatus: 'completed',
    hasInitialMessage: true,
    timelineLength: 0,
    hasError: false,
    hasRunningStep: false,
  }), true)
})

test('shouldHideWaitResumeCard should hide wait card only during optimistic resume phase', () => {
  assert.equal(shouldHideWaitResumeCard({
    sessionStatus: 'waiting',
    waitContextAvailable: true,
    waitResumePending: true,
  }), true)

  assert.equal(shouldHideWaitResumeCard({
    sessionStatus: 'running',
    waitContextAvailable: true,
    waitResumePending: true,
  }), false)

  assert.equal(shouldHideWaitResumeCard({
    sessionStatus: 'waiting',
    waitContextAvailable: false,
    waitResumePending: true,
  }), false)
})

test('shouldResetWaitResumePending should reset after leaving waiting or when waiting stream settles', () => {
  assert.equal(shouldResetWaitResumePending({
    waitResumePending: true,
    sessionStatus: 'running',
    streaming: true,
  }), true)

  assert.equal(shouldResetWaitResumePending({
    waitResumePending: true,
    sessionStatus: 'waiting',
    streaming: false,
  }), true)

  assert.equal(shouldResetWaitResumePending({
    waitResumePending: true,
    sessionStatus: 'waiting',
    streaming: true,
  }), false)

  assert.equal(shouldResetWaitResumePending({
    waitResumePending: false,
    sessionStatus: 'waiting',
    streaming: false,
  }), false)
})

test('resolveWaitResumeContext should use runtime interaction when waiting snapshot has no events', () => {
  const context = resolveWaitResumeContext({
    canResume: true,
    runtimeInteraction: waitInteraction('请补充预算'),
    events: [],
  })

  assert.ok(context)
  assert.equal(context?.interruptId, 'interrupt-runtime')
  assert.equal(context?.prompt, '请补充预算')
})

test('resolveWaitResumeContext should prefer runtime interaction over historical wait events', () => {
  const context = resolveWaitResumeContext({
    canResume: true,
    runtimeInteraction: waitInteraction('runtime 中的问题'),
    events: [waitEvent('历史 wait 问题')],
  })

  assert.equal(context?.interruptId, 'interrupt-runtime')
  assert.equal(context?.prompt, 'runtime 中的问题')
})

test('resolveWaitResumeContext should not expose historical wait action when can_resume is false', () => {
  const context = resolveWaitResumeContext({
    canResume: false,
    runtimeInteraction: {
      kind: 'none',
      interrupt_id: null,
      payload: {},
    },
    events: [waitEvent('旧 wait 不应可提交')],
  })

  assert.equal(context, null)
})
