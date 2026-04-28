import type {
  RuntimeCapabilities,
  RuntimeCursor,
  RuntimeInteraction,
  RuntimeObservation,
  SSEEventData,
  SessionDetail,
  SessionStatus,
} from './api/types'

export type RuntimeObservationViewState = {
  sessionId: string
  runId: string | null
  status: SessionStatus
  currentStepId: string | null
  cursor: RuntimeCursor
  capabilities: RuntimeCapabilities
  interaction: RuntimeInteraction
}

export type SessionRuntimeState = {
  status: SessionStatus | null
  streaming: boolean
}

export function createRuntimeObservationFromSnapshot(detail: SessionDetail): RuntimeObservationViewState {
  return runtimeObservationToViewState(detail.runtime)
}

export function reduceRuntimeObservationOnEvent(
  prev: RuntimeObservationViewState,
  event: SSEEventData,
): RuntimeObservationViewState {
  const runtime = event.data.runtime
  const nextStatus = runtime.status_after_event ?? prev.status
  const nextCursorEventId = getPersistentCursorEventId(event)
  const nextInteraction = reduceRuntimeInteractionOnEvent(prev.interaction, event, nextStatus)
  const nextCapabilities = runtime.status_after_event
    ? projectRuntimeCapabilitiesForStatus(nextStatus, prev.capabilities, nextInteraction)
    : prev.capabilities
  return {
    ...prev,
    runId: runtime.run_id ?? prev.runId,
    status: nextStatus,
    currentStepId: runtime.current_step_id,
    capabilities: nextCapabilities,
    interaction: nextInteraction,
    cursor: nextCursorEventId
      ? {
          ...prev.cursor,
          latest_event_id: nextCursorEventId,
        }
      : prev.cursor,
  }
}

export function reduceSessionRuntimeStateOnEvent(
  prev: SessionRuntimeState,
  event: SSEEventData,
): SessionRuntimeState {
  const statusAfterEvent = event.data.runtime.status_after_event
  if (!statusAfterEvent) {
    return prev
  }
  return {
    status: statusAfterEvent,
    streaming: isStreamingStatus(statusAfterEvent),
  }
}

export function getPersistentCursorEventId(event: SSEEventData): string | null {
  const runtime = event.data.runtime
  if (runtime.durability !== 'persistent') return null
  const cursorEventId = runtime.cursor_event_id
  if (typeof cursorEventId !== 'string') return null
  const normalized = cursorEventId.trim()
  return normalized.length > 0 ? normalized : null
}

export function collectPersistentCursorEventIds(events: SSEEventData[]): Set<string> {
  const eventIds = new Set<string>()
  for (const event of events) {
    const eventId = getPersistentCursorEventId(event)
    if (eventId) {
      eventIds.add(eventId)
    }
  }
  return eventIds
}

export function isStreamingStatus(status: SessionStatus | null | undefined): boolean {
  return status === 'running'
}

export function projectRuntimeCapabilitiesForStatus(
  status: SessionStatus,
  previous: RuntimeCapabilities,
  interaction: RuntimeInteraction,
): RuntimeCapabilities {
  if (status === 'running') {
    return {
      can_send_message: false,
      can_resume: false,
      can_cancel: true,
      can_continue_cancelled: false,
      disabled_reasons: previous.disabled_reasons,
    }
  }
  if (status === 'waiting') {
    return {
      can_send_message: false,
      can_resume: interaction.kind === 'wait',
      can_cancel: true,
      can_continue_cancelled: false,
      disabled_reasons: previous.disabled_reasons,
    }
  }
  if (status === 'completed' || status === 'failed') {
    return {
      can_send_message: true,
      can_resume: false,
      can_cancel: false,
      can_continue_cancelled: false,
      disabled_reasons: previous.disabled_reasons,
    }
  }
  if (status === 'cancelled') {
    return {
      can_send_message: true,
      can_resume: false,
      can_cancel: false,
      can_continue_cancelled: previous.can_continue_cancelled,
      disabled_reasons: previous.disabled_reasons,
    }
  }
  return {
    can_send_message: false,
    can_resume: false,
    can_cancel: false,
    can_continue_cancelled: false,
    disabled_reasons: previous.disabled_reasons,
  }
}

function runtimeObservationToViewState(runtime: RuntimeObservation): RuntimeObservationViewState {
  return {
    sessionId: runtime.session_id,
    runId: runtime.run_id,
    status: runtime.status,
    currentStepId: runtime.current_step_id,
    cursor: runtime.cursor,
    capabilities: runtime.capabilities,
    interaction: runtime.interaction,
  }
}

function reduceRuntimeInteractionOnEvent(
  previous: RuntimeInteraction,
  event: SSEEventData,
  nextStatus: SessionStatus,
): RuntimeInteraction {
  if (event.type === 'wait' && nextStatus === 'waiting') {
    return {
      kind: 'wait',
      interrupt_id: event.data.interrupt_id ?? null,
      payload: event.data.payload && typeof event.data.payload === 'object'
        ? { ...event.data.payload }
        : {},
    }
  }
  if (nextStatus !== 'waiting') {
    return {
      kind: 'none',
      interrupt_id: null,
      payload: {},
    }
  }
  return previous
}
