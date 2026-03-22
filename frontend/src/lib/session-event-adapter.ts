import type { PlanEvent, SSEEventData, SSEEventType, StepEvent, ToolEvent } from './api/types';

type RawSessionEvent = {
  event?: string;
  type?: string;
  data?: unknown;
};

type EventCompatExtension = {
  schema_version?: string;
  semantic_type?: string;
  [key: string]: unknown;
};

export type EventRuntimeContext = {
  session_id?: string;
  run_id?: string;
  channel?: string;
  [key: string]: unknown;
};

type EventExtensions = {
  compat?: EventCompatExtension;
  runtime?: EventRuntimeContext;
  [key: string]: unknown;
};

export type AdaptedSessionEvent = {
  event: SSEEventData;
  semanticType: string;
  compatSchemaVersion: string | null;
  runtimeContext: EventRuntimeContext | null;
};

type SessionEventHandlers = Partial<{
  [K in SSEEventType]: (event: Extract<SSEEventData, { type: K }>) => void;
}> & {
  default?: (event: SSEEventData) => void;
};

const KNOWN_SSE_EVENT_TYPES: ReadonlySet<SSEEventType> = new Set<SSEEventType>([
  'message',
  'title',
  'plan',
  'step',
  'tool',
  'wait',
  'done',
  'error',
]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function isKnownSSEEventType(type: unknown): type is SSEEventType {
  return typeof type === 'string' && KNOWN_SSE_EVENT_TYPES.has(type as SSEEventType);
}

function getExtensions(data: unknown): EventExtensions | null {
  if (!isRecord(data)) return null;
  const rawExtensions = data.extensions;
  if (!isRecord(rawExtensions)) return null;
  return rawExtensions as EventExtensions;
}

function deriveLegacySemanticType(event: SSEEventData): string {
  if (event.type === 'plan') {
    const status = (event.data as PlanEvent & { status?: string }).status;
    return typeof status === 'string' && status.length > 0 ? `plan.${status}` : 'plan';
  }
  if (event.type === 'step') {
    const stepData = event.data as StepEvent & { event_status?: string };
    if (typeof stepData.event_status === 'string' && stepData.event_status.length > 0) {
      return `step.${stepData.event_status}`;
    }
    return typeof stepData.status === 'string' && stepData.status.length > 0 ? `step.${stepData.status}` : 'step';
  }
  if (event.type === 'tool') {
    const status = (event.data as ToolEvent).status;
    return typeof status === 'string' && status.length > 0 ? `tool.${status}` : 'tool';
  }
  return event.type;
}

function isNestedRawEvent(value: unknown): value is RawSessionEvent {
  if (!isRecord(value)) return false;
  const hasRawType = typeof value.event === 'string' || typeof value.type === 'string';
  return hasRawType && 'data' in value;
}

/**
 * 将后端单条事件转为前端统一 SSEEventData（仅接受已知事件类型）。
 */
export function normalizeEvent(raw: RawSessionEvent): SSEEventData | null {
  const type = raw.type ?? raw.event;
  if (!isKnownSSEEventType(type)) return null;
  if (!('data' in raw)) return null;
  return {
    type,
    data: raw.data,
  } as SSEEventData;
}

/**
 * 将后端事件列表标准化为前端事件数组。
 */
export function normalizeEvents(rawList: unknown): SSEEventData[] {
  if (!Array.isArray(rawList)) return [];
  const normalized: SSEEventData[] = [];
  for (const raw of rawList) {
    const event = normalizeEvent((raw ?? {}) as RawSessionEvent);
    if (event) normalized.push(event);
  }
  return normalized;
}

/**
 * 兼容“事件嵌套事件”结构，返回可消费的统一事件对象。
 */
export function unwrapNestedEvent(event: SSEEventData): SSEEventData {
  if (!isNestedRawEvent(event.data)) return event;
  return normalizeEvent(event.data) ?? event;
}

/**
 * 读取事件扩展元信息，并提供旧协议语义兜底。
 */
export function adaptSessionEvent(event: SSEEventData): AdaptedSessionEvent {
  const extensions = getExtensions(event.data);
  const compat = extensions?.compat;
  const semanticFromCompat =
    compat && typeof compat.semantic_type === 'string' && compat.semantic_type.length > 0
      ? compat.semantic_type
      : null;

  return {
    event,
    semanticType: semanticFromCompat ?? deriveLegacySemanticType(event),
    compatSchemaVersion:
      compat && typeof compat.schema_version === 'string' && compat.schema_version.length > 0
        ? compat.schema_version
        : null,
    runtimeContext:
      extensions && isRecord(extensions.runtime)
        ? (extensions.runtime as EventRuntimeContext)
        : null,
  };
}

/**
 * 统一事件分发入口，避免消费侧重复 `if/switch` 分支堆叠。
 */
export function visitSessionEvent(event: SSEEventData, handlers: SessionEventHandlers): void {
  const handler = handlers[event.type] as ((event: SSEEventData) => void) | undefined;
  if (handler) {
    handler(event);
    return;
  }
  handlers.default?.(event);
}
