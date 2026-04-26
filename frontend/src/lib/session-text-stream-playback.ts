import type { ActiveTextStream, TextStreamState } from './session-text-stream-state'

const DEFAULT_PLAYBACK_CHARS_PER_TICK = 12

function syncSingleStream(
  previous: ActiveTextStream | undefined,
  next: ActiveTextStream,
): ActiveTextStream {
  if (!previous) {
    return {
      ...next,
      text: '',
      completed: false,
    }
  }
  return {
    ...next,
    text: previous.text,
    completed: previous.completed,
  }
}

export function syncDisplayedTextStreams(
  previous: TextStreamState,
  source: TextStreamState,
): TextStreamState {
  const nextState: TextStreamState = {}
  for (const [streamId, stream] of Object.entries(source)) {
    nextState[streamId] = syncSingleStream(previous[streamId], stream)
  }
  return nextState
}

export function advanceDisplayedTextStreams(
  displayed: TextStreamState,
  source: TextStreamState,
  charsPerTick: number = DEFAULT_PLAYBACK_CHARS_PER_TICK,
): TextStreamState {
  const safeCharsPerTick = Math.max(1, charsPerTick)
  let changed = false
  const nextState: TextStreamState = {}

  for (const [streamId, sourceStream] of Object.entries(source)) {
    const displayedStream = displayed[streamId] ?? {
      ...sourceStream,
      text: '',
      completed: false,
    }
    const displayedLength = displayedStream.text.length
    const targetText = sourceStream.text
    const nextText = targetText.slice(0, Math.min(displayedLength + safeCharsPerTick, targetText.length))
    const isCompleted = Boolean(sourceStream.completed) && nextText.length >= targetText.length

    if (nextText !== displayedStream.text || isCompleted !== displayedStream.completed) {
      changed = true
    }

    nextState[streamId] = {
      ...sourceStream,
      text: nextText,
      completed: isCompleted,
    }
  }

  return changed ? nextState : displayed
}
