import type {
  SSEEventData,
  TextStreamChannel,
  TextStreamDeltaEventData,
  TextStreamEndEventData,
  TextStreamStartEventData,
} from './api/types'

export type ActiveTextStream = {
  streamId: string
  channel: TextStreamChannel
  stage: 'planner' | 'summary' | 'final'
  text: string
  completed: boolean
}

export type TextStreamState = Record<string, ActiveTextStream>

function toStartData(event: SSEEventData): TextStreamStartEventData | null {
  return event.type === 'text_stream_start' ? event.data : null
}

function toDeltaData(event: SSEEventData): TextStreamDeltaEventData | null {
  return event.type === 'text_stream_delta' ? event.data : null
}

function toEndData(event: SSEEventData): TextStreamEndEventData | null {
  return event.type === 'text_stream_end' ? event.data : null
}

export function isTextStreamEvent(event: SSEEventData): boolean {
  return (
    event.type === 'text_stream_start' ||
    event.type === 'text_stream_delta' ||
    event.type === 'text_stream_end'
  )
}

export function reduceTextStreamState(prev: TextStreamState, event: SSEEventData): TextStreamState {
  const startData = toStartData(event)
  if (startData) {
    return {
      ...prev,
      [startData.stream_id]: {
        streamId: startData.stream_id,
        channel: startData.channel,
        stage: startData.stage,
        text: '',
        completed: false,
      },
    }
  }

  const deltaData = toDeltaData(event)
  if (deltaData) {
    const existing = prev[deltaData.stream_id]
    if (!existing) return prev
    return {
      ...prev,
      [deltaData.stream_id]: {
        ...existing,
        text: `${existing.text}${deltaData.text || ''}`,
      },
    }
  }

  const endData = toEndData(event)
  if (endData) {
    const existing = prev[endData.stream_id]
    if (!existing) return prev
    return {
      ...prev,
      [endData.stream_id]: {
        ...existing,
        completed: true,
      },
    }
  }

  if (event.type === 'message') {
    const nextState: TextStreamState = {}
    const stage = String((event.data as { stage?: string }).stage || '')
    const channelToClear: TextStreamChannel | null = stage === 'final' ? 'final_message' : 'planner_message'
    for (const [streamId, stream] of Object.entries(prev)) {
      if (stream.channel !== channelToClear) {
        nextState[streamId] = stream
      }
    }
    return nextState
  }

  if (event.type === 'done' || event.type === 'error') {
    return {}
  }

  return prev
}

export function getLatestActiveStreamByChannel(
  state: TextStreamState,
  channel: TextStreamChannel,
): ActiveTextStream | null {
  let latest: ActiveTextStream | null = null
  for (const stream of Object.values(state)) {
    if (stream.channel !== channel) continue
    latest = stream
  }
  return latest
}
