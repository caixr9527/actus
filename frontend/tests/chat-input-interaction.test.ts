import assert from 'node:assert/strict'
import test from 'node:test'

import { resolveChatInputInteractionState } from '../src/lib/chat-input-interaction'

test('resolveChatInputInteractionState should allow typing but block send and model switch while running', () => {
  const state = resolveChatInputInteractionState({
    disabled: true,
    isRunning: true,
    sending: false,
    inputValue: '继续记录草稿',
    modelsLoading: false,
    modelUpdating: false,
  })

  assert.equal(state.canEditInput, true)
  assert.equal(state.canSend, false)
  assert.equal(state.canSwitchModel, false)
})

test('resolveChatInputInteractionState should block typing when disabled in non-running state', () => {
  const state = resolveChatInputInteractionState({
    disabled: true,
    isRunning: false,
    sending: false,
    inputValue: 'draft',
    modelsLoading: false,
    modelUpdating: false,
  })

  assert.equal(state.canEditInput, false)
  assert.equal(state.canSend, false)
  assert.equal(state.canSwitchModel, false)
})

test('resolveChatInputInteractionState should allow send and model switch when idle and input is valid', () => {
  const state = resolveChatInputInteractionState({
    disabled: false,
    isRunning: false,
    sending: false,
    inputValue: '可发送内容',
    modelsLoading: false,
    modelUpdating: false,
  })

  assert.equal(state.canEditInput, true)
  assert.equal(state.canSend, true)
  assert.equal(state.canSwitchModel, true)
})

