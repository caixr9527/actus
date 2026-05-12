import assert from 'node:assert/strict'
import test from 'node:test'

import { resolveAssistantTurnSummary } from '../src/lib/assistant-turn-summary'
import type { AssistantTurnStatus } from '../src/lib/assistant-turns'
import type { Translate } from '../src/lib/i18n'

const messages: Record<string, string> = {
  'sessionDetail.turnSummary.working': '工作中',
  'sessionDetail.turnSummary.done': '工作完成',
}

const t: Translate = (key) => messages[key] ?? key

test('resolveAssistantTurnSummary should use working for active statuses', () => {
  const activeStatuses: AssistantTurnStatus[] = ['running', 'waiting', 'idle']

  for (const status of activeStatuses) {
    assert.equal(resolveAssistantTurnSummary(status, t), '工作中')
  }
})

test('resolveAssistantTurnSummary should use done for every terminal status', () => {
  const terminalStatuses: AssistantTurnStatus[] = ['completed', 'failed', 'cancelled']

  for (const status of terminalStatuses) {
    assert.equal(resolveAssistantTurnSummary(status, t), '工作完成')
  }
})
