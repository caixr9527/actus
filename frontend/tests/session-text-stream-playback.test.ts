import assert from 'node:assert/strict'
import test from 'node:test'

import {
  advanceDisplayedTextStreams,
  syncDisplayedTextStreams,
} from '../src/lib/session-text-stream-playback'
import type { TextStreamState } from '../src/lib/session-text-stream-state'

test('syncDisplayedTextStreams should preserve existing playback progress and remove stale streams', () => {
  const previous: TextStreamState = {
    'stream-1': {
      streamId: 'stream-1',
      channel: 'final_message',
      stage: 'final',
      text: '已经播到这里',
      completed: false,
    },
    'stale-stream': {
      streamId: 'stale-stream',
      channel: 'planner_message',
      stage: 'planner',
      text: '旧内容',
      completed: false,
    },
  }
  const source: TextStreamState = {
    'stream-1': {
      streamId: 'stream-1',
      channel: 'final_message',
      stage: 'final',
      text: '完整最终内容',
      completed: false,
    },
  }

  const synced = syncDisplayedTextStreams(previous, source)

  assert.deepEqual(Object.keys(synced), ['stream-1'])
  assert.equal(synced['stream-1']?.text, '已经播到这里')
})

test('advanceDisplayedTextStreams should reveal text gradually and mark completed only after catch-up', () => {
  const source: TextStreamState = {
    'stream-1': {
      streamId: 'stream-1',
      channel: 'final_message',
      stage: 'final',
      text: 'abcdef',
      completed: true,
    },
  }

  const first = advanceDisplayedTextStreams({}, source, 2)
  assert.equal(first['stream-1']?.text, 'ab')
  assert.equal(first['stream-1']?.completed, false)

  const second = advanceDisplayedTextStreams(first, source, 2)
  assert.equal(second['stream-1']?.text, 'abcd')
  assert.equal(second['stream-1']?.completed, false)

  const third = advanceDisplayedTextStreams(second, source, 2)
  assert.equal(third['stream-1']?.text, 'abcdef')
  assert.equal(third['stream-1']?.completed, true)
})
