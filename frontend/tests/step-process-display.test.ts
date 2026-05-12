import assert from 'node:assert/strict'
import test from 'node:test'

import { resolveStepProcessSectionOrder } from '../src/lib/step-process-display'

test('resolveStepProcessSectionOrder should render tools before step summary', () => {
  assert.deepEqual(
    resolveStepProcessSectionOrder({ hasTools: true, hasSummary: true }),
    ['tools', 'summary'],
  )
})

test('resolveStepProcessSectionOrder should omit absent sections', () => {
  assert.deepEqual(
    resolveStepProcessSectionOrder({ hasTools: true, hasSummary: false }),
    ['tools'],
  )
  assert.deepEqual(
    resolveStepProcessSectionOrder({ hasTools: false, hasSummary: true }),
    ['summary'],
  )
  assert.deepEqual(
    resolveStepProcessSectionOrder({ hasTools: false, hasSummary: false }),
    [],
  )
})
