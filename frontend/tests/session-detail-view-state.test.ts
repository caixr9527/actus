import assert from 'node:assert/strict'
import test from 'node:test'

import {
  createSessionScopedDetailViewState,
  createSessionScopedRuntimeState,
  resolveStepExpandedState,
  shouldAutoCloseTaskPreview,
  shouldAutoExpandStep,
  shouldAutoScrollToLatest,
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
