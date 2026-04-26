import assert from 'node:assert/strict'
import test from 'node:test'

import {
  getLatestActiveStreamByChannel,
  isTextStreamEvent,
  reduceTextStreamState,
  type TextStreamState,
} from '../src/lib/session-text-stream-state'
import type { SSEEventData } from '../src/lib/api/types'

function eventOf(type: SSEEventData['type'], data: unknown): SSEEventData {
  return { type, data } as SSEEventData
}

test('isTextStreamEvent should classify live-only draft events', () => {
  assert.equal(isTextStreamEvent(eventOf('text_stream_start', {
    stream_id: 'stream-1',
    channel: 'planner_message',
    stage: 'planner',
  })), true)
  assert.equal(isTextStreamEvent(eventOf('text_stream_delta', {
    stream_id: 'stream-1',
    channel: 'planner_message',
    text: 'draft',
    sequence: 1,
  })), true)
  assert.equal(isTextStreamEvent(eventOf('text_stream_end', {
    stream_id: 'stream-1',
    channel: 'planner_message',
    full_text_length: 5,
    reason: 'completed',
  })), true)
  assert.equal(isTextStreamEvent(eventOf('message', {
    stage: 'final',
    role: 'assistant',
    message: '正式消息',
  })), false)
})

test('reduceTextStreamState should accumulate planner draft and mark completed on end', () => {
  let state: TextStreamState = {}

  state = reduceTextStreamState(state, eventOf('text_stream_start', {
    stream_id: 'stream-1',
    channel: 'planner_message',
    stage: 'planner',
  }))
  state = reduceTextStreamState(state, eventOf('text_stream_delta', {
    stream_id: 'stream-1',
    channel: 'planner_message',
    text: '第一段',
    sequence: 1,
  }))
  state = reduceTextStreamState(state, eventOf('text_stream_delta', {
    stream_id: 'stream-1',
    channel: 'planner_message',
    text: '第二段',
    sequence: 2,
  }))
  state = reduceTextStreamState(state, eventOf('text_stream_end', {
    stream_id: 'stream-1',
    channel: 'planner_message',
    full_text_length: 6,
    reason: 'completed',
  }))

  assert.equal(state['stream-1']?.text, '第一段第二段')
  assert.equal(state['stream-1']?.completed, true)
})

test('reduceTextStreamState should clear matching channel after final message settles', () => {
  let state: TextStreamState = {}

  state = reduceTextStreamState(state, eventOf('text_stream_start', {
    stream_id: 'planner-1',
    channel: 'planner_message',
    stage: 'planner',
  }))
  state = reduceTextStreamState(state, eventOf('text_stream_start', {
    stream_id: 'final-1',
    channel: 'final_message',
    stage: 'final',
  }))

  state = reduceTextStreamState(state, eventOf('message', {
    stage: 'final',
    role: 'assistant',
    message: '最终回复',
  }))

  assert.equal(state['final-1'], undefined)
  assert.ok(state['planner-1'])
})

test('reduceTextStreamState should clear planner draft after intermediate message settles', () => {
  let state: TextStreamState = {}

  state = reduceTextStreamState(state, eventOf('text_stream_start', {
    stream_id: 'planner-1',
    channel: 'planner_message',
    stage: 'planner',
  }))
  state = reduceTextStreamState(state, eventOf('text_stream_start', {
    stream_id: 'final-1',
    channel: 'final_message',
    stage: 'final',
  }))

  state = reduceTextStreamState(state, eventOf('message', {
    stage: 'intermediate',
    role: 'assistant',
    message: '计划说明',
  }))

  assert.equal(state['planner-1'], undefined)
  assert.ok(state['final-1'])
})

test('reduceTextStreamState should clear all draft streams on done or error', () => {
  let state: TextStreamState = {}

  state = reduceTextStreamState(state, eventOf('text_stream_start', {
    stream_id: 'stream-1',
    channel: 'planner_message',
    stage: 'planner',
  }))
  assert.equal(Object.keys(state).length, 1)

  state = reduceTextStreamState(state, eventOf('done', {}))
  assert.deepEqual(state, {})

  state = reduceTextStreamState(state, eventOf('text_stream_start', {
    stream_id: 'stream-2',
    channel: 'final_message',
    stage: 'final',
  }))
  state = reduceTextStreamState(state, eventOf('error', {
    error: 'boom',
  }))
  assert.deepEqual(state, {})
})

test('getLatestActiveStreamByChannel should return the most recently started stream for the channel', () => {
  let state: TextStreamState = {}

  state = reduceTextStreamState(state, eventOf('text_stream_start', {
    stream_id: 'z-stream',
    channel: 'planner_message',
    stage: 'planner',
  }))
  state = reduceTextStreamState(state, eventOf('text_stream_start', {
    stream_id: 'a-stream',
    channel: 'planner_message',
    stage: 'planner',
  }))

  const latest = getLatestActiveStreamByChannel(state, 'planner_message')
  assert.equal(latest?.streamId, 'a-stream')
})
