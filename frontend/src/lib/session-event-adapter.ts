import type { SSEEventData, SSEEventType } from './api/types'

type RawSessionEvent = {
  event?: string
  type?: string
  data?: unknown
}

type SessionEventHandlers = Partial<{
  [K in SSEEventType]: (event: Extract<SSEEventData, { type: K }>) => void
}> & {
  default?: (event: SSEEventData) => void
}

const KNOWN_SSE_EVENT_TYPES: ReadonlySet<SSEEventType> = new Set<SSEEventType>([
  'message',
  'title',
  'plan',
  'step',
  'tool',
  'wait',
  'done',
  'error',
  'text_stream_start',
  'text_stream_delta',
  'text_stream_end',
])

function isKnownSSEEventType(type: unknown): type is SSEEventType {
  return typeof type === 'string' && KNOWN_SSE_EVENT_TYPES.has(type as SSEEventType)
}

function isNestedRawEvent(value: unknown): value is RawSessionEvent {
  if (typeof value !== 'object' || value === null) return false
  const rawEvent = value as RawSessionEvent
  const hasRawType = typeof rawEvent.event === 'string' || typeof rawEvent.type === 'string'
  return hasRawType && 'data' in rawEvent
}

/**
 * 将后端单条事件转为前端统一 SSEEventData（仅接受已知事件类型）。
 */
export function normalizeEvent(raw: RawSessionEvent): SSEEventData | null {
  const type = raw.type ?? raw.event
  if (!isKnownSSEEventType(type)) return null
  if (!('data' in raw)) return null
  return {
    type,
    data: raw.data,
  } as SSEEventData
}

/**
 * 将后端事件列表标准化为前端事件数组。
 */
export function normalizeEvents(rawList: unknown): SSEEventData[] {
  if (!Array.isArray(rawList)) return []
  const normalized: SSEEventData[] = []
  for (const raw of rawList) {
    const event = normalizeEvent((raw ?? {}) as RawSessionEvent)
    if (event) normalized.push(event)
  }
  return normalized
}

/**
 * 兼容“事件嵌套事件”结构，返回可消费的统一事件对象。
 */
export function unwrapNestedEvent(event: SSEEventData): SSEEventData {
  if (!isNestedRawEvent(event.data)) return event
  return normalizeEvent(event.data) ?? event
}

/**
 * 统一事件分发入口，避免消费侧重复 `if/switch` 分支堆叠。
 */
export function visitSessionEvent(event: SSEEventData, handlers: SessionEventHandlers): void {
  const handler = handlers[event.type] as ((event: SSEEventData) => void) | undefined
  if (handler) {
    handler(event)
    return
  }
  handlers.default?.(event)
}
