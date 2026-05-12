import assert from 'node:assert/strict'
import test from 'node:test'

import { resolveStepDetail } from '../src/lib/run-timeline'

test('resolveStepDetail should prefer next_hint for failed step', () => {
  assert.equal(
    resolveStepDetail({
      done: false,
      summary: '失败 summary 不应展示',
      produced_artifacts: [],
      blockers: ['失败 blocker'],
      facts_learned: [],
      open_questions: [],
      next_hint: '请缩小范围后重试',
    }, 'failed'),
    '请缩小范围后重试',
  )
})

test('resolveStepDetail should fallback to blocker for failed step without next_hint', () => {
  assert.equal(
    resolveStepDetail({
      done: false,
      summary: '失败 summary 不应展示',
      produced_artifacts: [],
      blockers: ['失败 blocker'],
      facts_learned: [],
      open_questions: [],
      next_hint: '',
    }, 'failed'),
    '失败 blocker',
  )
})

test('resolveStepDetail should keep summary first for completed step', () => {
  assert.equal(
    resolveStepDetail({
      done: true,
      summary: '完成 summary',
      produced_artifacts: [],
      blockers: ['不应优先展示'],
      facts_learned: [],
      open_questions: [],
      next_hint: '不应优先展示',
    }, 'completed'),
    '完成 summary',
  )
})
