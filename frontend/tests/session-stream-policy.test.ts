import assert from 'node:assert/strict'
import test from 'node:test'

import {
  canRetry,
  computeRetryDelayMs,
  shouldStartEmptySessionStream,
  type RetryPolicy,
} from '../src/lib/session-stream-policy'

const retryPolicy: RetryPolicy = {
  maxRetries: 3,
  baseDelayMs: 1000,
  maxDelayMs: 5000,
}

test('computeRetryDelayMs should use exponential backoff and honor max cap', () => {
  assert.equal(computeRetryDelayMs(0, retryPolicy), 1000)
  assert.equal(computeRetryDelayMs(1, retryPolicy), 2000)
  assert.equal(computeRetryDelayMs(2, retryPolicy), 4000)
  assert.equal(computeRetryDelayMs(3, retryPolicy), 5000)
  assert.equal(computeRetryDelayMs(99, retryPolicy), 5000)
})

test('canRetry should stop at maxRetries boundary', () => {
  assert.equal(canRetry(0, retryPolicy), true)
  assert.equal(canRetry(2, retryPolicy), true)
  assert.equal(canRetry(3, retryPolicy), false)
  assert.equal(canRetry(4, retryPolicy), false)
})

test('shouldStartEmptySessionStream should only subscribe in valid runtime states', () => {
  assert.equal(shouldStartEmptySessionStream('running', false, false), true)
  assert.equal(shouldStartEmptySessionStream('waiting', false, false), true)
  assert.equal(shouldStartEmptySessionStream('completed', false, false), false)
  assert.equal(shouldStartEmptySessionStream('cancelled', false, false), false)
  assert.equal(shouldStartEmptySessionStream('running', true, false), false)
  assert.equal(shouldStartEmptySessionStream('running', false, true), false)
  assert.equal(shouldStartEmptySessionStream(null, false, false), false)
  assert.equal(shouldStartEmptySessionStream(undefined, false, false), false)
})
