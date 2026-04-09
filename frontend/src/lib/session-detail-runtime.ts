import type { SSEEventData, SessionStatus } from './api/types'
import { visitSessionEvent } from './session-event-adapter'

export type SessionRuntimeState = {
  status: SessionStatus | null
  streaming: boolean
}

export type MessageStreamCloseReason = 'done' | 'abort' | 'stream_end' | 'error'

export function getSessionEventId(event: SSEEventData): string | null {
  const eventId = (event.data as { event_id?: string | null })?.event_id
  if (typeof eventId !== 'string') return null
  const normalized = eventId.trim()
  return normalized.length > 0 ? normalized : null
}

export function collectSessionEventIds(events: SSEEventData[]): Set<string> {
  const eventIds = new Set<string>()
  for (const event of events) {
    const eventId = getSessionEventId(event)
    if (eventId) {
      eventIds.add(eventId)
    }
  }
  return eventIds
}

export function reduceSessionRuntimeStateOnEvent(
  prev: SessionRuntimeState,
  event: SSEEventData,
): SessionRuntimeState {
  // cancelled 是当前会话运行态的终态。
  // 一旦 stop 已经收敛，旧流里迟到的 running/waiting 事件不允许再把状态改回去。
  if (prev.status === 'cancelled') {
    return {
      status: 'cancelled',
      streaming: false,
    }
  }

  let nextStatus: SessionStatus | null = prev.status
  let nextStreaming = prev.streaming

  visitSessionEvent(event, {
    plan: (planEvent) => {
      const planData = planEvent.data as { plan_status?: string }
      if (planData.plan_status === 'cancelled') {
        nextStatus = 'cancelled'
        nextStreaming = false
      }
    },
    step: (stepEvent) => {
      const stepData = stepEvent.data as { status?: string }
      if (stepData.status === 'running') {
        nextStatus = 'running'
      }
      if (stepData.status === 'cancelled') {
        nextStatus = 'cancelled'
        nextStreaming = false
      }
      if (stepData.status === 'waiting') {
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
