import type { SSEEventData, SessionStatus } from './api/types'
import { adaptSessionEvent, visitSessionEvent } from './session-event-adapter'

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
  const adapted = adaptSessionEvent(event)

  visitSessionEvent(event, {
    step: (stepEvent) => {
      const stepData = stepEvent.data as { status?: string }
      if (stepData.status === 'running' || adapted.semanticType === 'step.started') {
        nextStatus = 'running'
      }
      if (stepData.status === 'waiting' || adapted.semanticType === 'step.waiting') {
        nextStatus = 'waiting'
        nextStreaming = false
      }
    },
    tool: (toolEvent) => {
      const toolData = toolEvent.data as { function?: string; status?: string }
      const isCalling = toolData.status === 'calling' || adapted.semanticType === 'tool.calling'
      if (toolData.function === 'message_ask_user' && isCalling) {
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
