import assert from 'node:assert/strict'
import test from 'node:test'

import {
  closeTaskPreviewStateOnSend,
  createSessionScopedDetailViewState,
  createSessionScopedRuntimeState,
  resolveStepExpandedState,
  shouldAutoCloseTaskPreview,
  shouldAutoExpandStep,
  shouldAutoScrollToLatest,
  shouldHideWaitResumeCard,
  shouldResetWaitResumePending,
  shouldShowSessionThinking,
} from '../src/lib/session-detail-view-state'

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

test('closeTaskPreviewStateOnSend should close preview panel and file list', () => {
  const next = closeTaskPreviewStateOnSend({
    fileListOpen: true,
    previewFile: { id: 'file-1' },
    previewTool: { id: 'tool-1' },
    timelineExpanded: true,
    vncOpen: false,
  })

  assert.deepEqual(next, {
    fileListOpen: false,
    previewFile: null,
    previewTool: null,
    timelineExpanded: true,
    vncOpen: false,
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
