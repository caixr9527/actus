import assert from 'node:assert/strict'
import test from 'node:test'

import type { SSEEventData } from '../src/lib/api/types'
import {
  formatRuntimeInputPartSummary,
  hasRuntimeInputPolicySignal,
  parseRuntimeInputPolicyFromContext,
  pickLatestRuntimeInputPolicySnapshot,
} from '../src/lib/runtime-input-policy'

function eventOf(type: SSEEventData['type'], data: unknown): SSEEventData {
  return { type, data } as SSEEventData
}

test('parseRuntimeInputPolicyFromContext should parse summary and unsupported details', () => {
  const snapshot = parseRuntimeInputPolicyFromContext({
    input_part_summary: { total: 3, by_type: { text: 1, file_ref: 1, image: 1 } },
    downgrade_reason: 'model_multimodal_disabled',
    unsupported_parts: [
      { type: 'image', filepath: '/home/ubuntu/upload/a.png', reason: 'model_multimodal_disabled' },
    ],
  })

  assert.deepEqual(snapshot.inputPartSummary, {
    total: 3,
    byType: { text: 1, file_ref: 1, image: 1 },
  })
  assert.equal(snapshot.downgradeReason, 'model_multimodal_disabled')
  assert.deepEqual(snapshot.unsupportedParts, [
    {
      type: 'image',
      filepath: '/home/ubuntu/upload/a.png',
      reason: 'model_multimodal_disabled',
    },
  ])
  assert.equal(hasRuntimeInputPolicySignal(snapshot), true)
})

test('pickLatestRuntimeInputPolicySnapshot should pick latest event with policy signal', () => {
  const events: SSEEventData[] = [
    eventOf('step', {
      id: 's1',
      extensions: {
        runtime: {
          input_part_summary: { total: 1, by_type: { text: 1 } },
        },
      },
    }),
    eventOf('message', { role: 'assistant', message: 'no runtime context' }),
    eventOf('tool', {
      status: 'called',
      extensions: {
        runtime: {
          downgrade_reason: 'model_supported_not_contains_type',
          unsupported_parts: [{ type: 'audio', reason: 'model_supported_not_contains_type' }],
        },
      },
    }),
  ]

  const snapshot = pickLatestRuntimeInputPolicySnapshot(events)
  assert.equal(snapshot.downgradeReason, 'model_supported_not_contains_type')
  assert.equal(snapshot.unsupportedParts.length, 1)
  assert.equal(snapshot.unsupportedParts[0]?.type, 'audio')
})

test('formatRuntimeInputPartSummary should keep type order deterministic', () => {
  const summaryText = formatRuntimeInputPartSummary({
    total: 4,
    byType: { image: 1, text: 2, audio: 1 },
  })
  assert.equal(summaryText, '4 (text×2, image×1, audio×1)')
})
