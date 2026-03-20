import assert from 'node:assert/strict'
import test from 'node:test'

import {
  performModelSelection,
  resolveModelSelectorState,
} from '../src/lib/chat-model-selector'
import type { ListModelItem } from '../src/lib/api/types'

const MODELS: ListModelItem[] = [
  {
    id: 'gpt-5.4',
    display_name: 'GPT-5.4',
    provider: 'openai',
    enabled: true,
    sort_order: 1,
    config: {
      badge: 'Reasoning',
      description: '复杂任务',
    },
  },
  {
    id: 'deepseek',
    display_name: 'DeepSeek',
    provider: 'deepseek',
    enabled: true,
    sort_order: 2,
    config: {
      badge: 'Fast',
      description: '快速响应',
    },
  },
]

test('resolveModelSelectorState should map null current model to auto and keep default model', () => {
  const state = resolveModelSelectorState(MODELS, null, 'gpt-5.4')

  assert.equal(state.selectedModelId, 'auto')
  assert.equal(state.selectedModel, null)
  assert.equal(state.defaultModel?.id, 'gpt-5.4')
})

test('resolveModelSelectorState should resolve explicit selected model', () => {
  const state = resolveModelSelectorState(MODELS, 'deepseek', 'gpt-5.4')

  assert.equal(state.selectedModelId, 'deepseek')
  assert.equal(state.selectedModel?.display_name, 'DeepSeek')
  assert.equal(state.defaultModel?.display_name, 'GPT-5.4')
})

test('performModelSelection should close menu and skip update when selecting current model', async () => {
  const calls: string[] = []
  let closeCount = 0

  const changed = await performModelSelection({
    nextModelId: 'deepseek',
    selectedModelId: 'deepseek',
    onModelChange: async (modelId) => {
      calls.push(modelId)
    },
    closeMenu: () => {
      closeCount += 1
    },
  })

  assert.equal(changed, false)
  assert.deepEqual(calls, [])
  assert.equal(closeCount, 1)
})

test('performModelSelection should call update and close menu for a new model', async () => {
  const calls: string[] = []
  let closeCount = 0

  const changed = await performModelSelection({
    nextModelId: 'gpt-5.4',
    selectedModelId: 'auto',
    onModelChange: async (modelId) => {
      calls.push(modelId)
    },
    closeMenu: () => {
      closeCount += 1
    },
  })

  assert.equal(changed, true)
  assert.deepEqual(calls, ['gpt-5.4'])
  assert.equal(closeCount, 1)
})

test('performModelSelection should keep menu open when update fails', async () => {
  let closeCount = 0

  await assert.rejects(
    performModelSelection({
      nextModelId: 'gpt-5.4',
      selectedModelId: 'auto',
      onModelChange: async () => {
        throw new Error('update failed')
      },
      closeMenu: () => {
        closeCount += 1
      },
    }),
    /update failed/,
  )

  assert.equal(closeCount, 0)
})
