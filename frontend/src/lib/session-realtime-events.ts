import { normalizeEvents, unwrapNestedEvent } from './session-event-adapter'
import { collectSessionEventIds, getSessionEventId } from './session-detail-runtime'
import {
  isTextStreamEvent,
  reduceTextStreamState,
  type TextStreamState,
} from './session-text-stream-state'
import type { SSEEventData } from './api/types'

export type SessionRealtimeState = {
  events: SSEEventData[]
  seenPersistentCursorIds: Set<string>
  lastEventId: string | null
  textStreams: TextStreamState
  displayedTextStreams: TextStreamState
}

export type SessionRealtimeAppendResult = {
  state: SessionRealtimeState
  event: SSEEventData
  appendedToTimeline: boolean
  duplicatePersistentEvent: boolean
  shouldUpdateDisplayedTextStreams: boolean
}

export function buildSessionRealtimeStateFromSnapshot(params: {
  rawEvents: unknown
  snapshotLatestEventId: string | null
}): SessionRealtimeState {
  const normalizedEvents = normalizeEvents(params.rawEvents)
  const persistentEvents = normalizedEvents.filter((event) => !isTextStreamEvent(event))
  const textStreams = persistentEvents.reduce<TextStreamState>(
    (state, event) => reduceTextStreamState(state, event),
    {},
  )
  const lastPersistentEventId = normalizedEvents
    .map((event) => getSessionEventId(event))
    .filter((eventId): eventId is string => Boolean(eventId))
    .at(-1)

  return {
    events: persistentEvents,
    seenPersistentCursorIds: collectSessionEventIds(persistentEvents),
    lastEventId: lastPersistentEventId ?? params.snapshotLatestEventId,
    textStreams,
    displayedTextStreams: textStreams,
  }
}

export function appendSessionRealtimeEvent(
  state: SessionRealtimeState,
  event: SSEEventData,
): SessionRealtimeAppendResult {
  const eventToAppend = unwrapNestedEvent(event)
  const eventId = getSessionEventId(eventToAppend)

  if (eventId && state.seenPersistentCursorIds.has(eventId)) {
    return {
      state,
      event: eventToAppend,
      appendedToTimeline: false,
      duplicatePersistentEvent: true,
      shouldUpdateDisplayedTextStreams: false,
    }
  }

  const nextSeenPersistentCursorIds = new Set(state.seenPersistentCursorIds)
  const nextLastEventId = eventId ?? state.lastEventId
  if (eventId) {
    nextSeenPersistentCursorIds.add(eventId)
  }

  const isTemporaryTextStreamEvent = isTextStreamEvent(eventToAppend)
  const nextTextStreams = reduceTextStreamState(state.textStreams, eventToAppend)
  const shouldSyncDisplayedTextStreams =
    eventToAppend.type === 'message' ||
    eventToAppend.type === 'done' ||
    eventToAppend.type === 'error'

  return {
    state: {
      events: isTemporaryTextStreamEvent
        ? state.events
        : [...state.events, eventToAppend],
      seenPersistentCursorIds: nextSeenPersistentCursorIds,
      lastEventId: nextLastEventId,
      textStreams: nextTextStreams,
      displayedTextStreams: shouldSyncDisplayedTextStreams
        ? reduceTextStreamState(state.displayedTextStreams, eventToAppend)
        : state.displayedTextStreams,
    },
    event: eventToAppend,
    appendedToTimeline: !isTemporaryTextStreamEvent,
    duplicatePersistentEvent: false,
    shouldUpdateDisplayedTextStreams: shouldSyncDisplayedTextStreams,
  }
}
