import assert from 'node:assert/strict'
import test from 'node:test'

import { eventsToTimeline } from '../src/lib/session-events'
import type { SSEEventData } from '../src/lib/api/types'

function eventOf(type: SSEEventData['type'], data: unknown): SSEEventData {
  return { type, data } as SSEEventData
}

test('eventsToTimeline should merge tool updates and reset step context across user turns', () => {
  const events: SSEEventData[] = [
    eventOf('message', {
      role: 'user',
      message: 'first question',
      attachments: [{ file_id: 'f1', filename: 'a.txt' }],
    }),
    eventOf('step', { id: 'step-1', status: 'running', description: 'phase 1' }),
    eventOf('tool', {
      tool_call_id: 'tool-1',
      name: 'browser',
      function: 'browser_open',
      args: { url: 'https://example.com' },
      status: 'calling',
    }),
    eventOf('tool', {
      tool_call_id: 'tool-1',
      name: 'browser',
      function: 'browser_open',
      args: { url: 'https://example.com' },
      status: 'called',
      content: 'ok',
    }),
    eventOf('step', { id: 'step-1', status: 'completed', description: 'phase 1' }),
    eventOf('message', { role: 'user', message: 'second question' }),
    eventOf('step', { id: 'step-1', status: 'running', description: 'phase 2' }),
    eventOf('error', { error: 'broken' }),
  ]

  const timeline = eventsToTimeline(events)

  const stepItems = timeline.filter((item) => item.kind === 'step')
  assert.equal(stepItems.length, 2)

  const firstStep = stepItems[0]
  assert.equal(firstStep.kind, 'step')
  if (firstStep.kind === 'step') {
    assert.equal(firstStep.tools.length, 1)
    assert.equal((firstStep.tools[0] as { status?: string }).status, 'called')
    assert.equal((firstStep.tools[0] as { content?: string }).content, 'ok')
  }

  const secondStep = stepItems[1]
  assert.equal(secondStep.kind, 'step')
  if (secondStep.kind === 'step') {
    assert.equal(secondStep.tools.length, 0)
    assert.equal(secondStep.data.description, 'phase 2')
  }

  const attachmentItems = timeline.filter((item) => item.kind === 'attachments')
  assert.equal(attachmentItems.length, 1)

  const errorItems = timeline.filter((item) => item.kind === 'error')
  assert.equal(errorItems.length, 1)
  if (errorItems[0]?.kind === 'error') {
    assert.equal(errorItems[0].error, 'broken')
  }
})
