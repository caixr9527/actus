import type { SSEEventData, SessionStatus } from './api/types'
import { visitSessionEvent } from './session-event-adapter'

export type SessionRuntimeState = {
  status: SessionStatus | null
  streaming: boolean
}

export type MessageStreamCloseReason = 'done' | 'abort' | 'stream_end' | 'error'

export function reduceSessionRuntimeStateOnEvent(
  prev: SessionRuntimeState,
  event: SSEEventData,
): SessionRuntimeState {
  let nextStatus = prev.status
  let nextStreaming = prev.streaming

  visitSessionEvent(event, {
    step: (stepEvent) => {
      const stepData = stepEvent.data as { status?: string }
      if (stepData.status === 'running') {
        nextStatus = 'running'
      }
      if (stepData.status === 'waiting') {
        nextStatus = 'waiting'
        nextStreaming = false
      }
    },
    tool: (toolEvent) => {
      const toolData = toolEvent.data as { function?: string; status?: string }
      if (toolData.function === 'message_ask_user' && toolData.status === 'calling') {
        nextStatus = 'waiting'
        nextStreaming = false
      }
    },
    wait: () => {
      nextStatus = 'waiting'
      nextStreaming = false
    },
    done: () => {
      nextStatus = 'completed'
      nextStreaming = false
    },
    error: () => {
      nextStatus = 'completed'
      nextStreaming = false
    },
  })

  return {
    status: nextStatus,
    streaming: nextStreaming,
  }
}

export function classifyMessageStreamCloseReason(err: Error): MessageStreamCloseReason {
  if (err.name === 'AbortError') return 'abort'
  if (err.message === 'SSE_STREAM_END') return 'stream_end'
  return 'error'
}

export function shouldReloadSnapshotAfterMessageStreamClose(reason: MessageStreamCloseReason): boolean {
  return reason === 'stream_end' || reason === 'error'
}
