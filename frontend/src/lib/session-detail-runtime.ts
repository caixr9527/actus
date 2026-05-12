export {
  collectPersistentCursorEventIds as collectSessionEventIds,
  getPersistentCursorEventId as getSessionEventId,
  reduceSessionRuntimeStateOnEvent,
  type SessionRuntimeState,
} from './runtime-observation'

export type MessageStreamCloseReason = 'done' | 'abort' | 'stream_end' | 'error'

export function classifyMessageStreamCloseReason(err: Error): MessageStreamCloseReason {
  if (err.name === 'AbortError') return 'abort'
  if (err.message === 'SSE_STREAM_END') return 'stream_end'
  return 'error'
}

export function shouldReloadSnapshotAfterMessageStreamClose(reason: MessageStreamCloseReason): boolean {
  return reason === 'stream_end' || reason === 'error'
}
