/**
 * 将 SSE 事件列表转换为时间线展示项与计划步骤
 * 与 chat 流式 / 任务详情接口的响应格式一致
 *
 * 后端事件格式为 { event: "message"|"title"|..., data: {...} }，
 * 前端统一使用 { type, data }，需先归一化。
 */

import type {
  SSEEventData,
  SSEEventType,
  ChatMessage,
  PlanStep,
  PlanEvent,
  StepEvent,
  ToolEvent,
  SessionFile,
} from "./api/types";

/** 后端返回的原始事件（可能用 event 或 type 表示类型） */
type RawEvent = { event?: string; type?: string; data?: unknown };

/**
 * 将后端单条事件转为前端 SSEEventData（统一 type + data）
 */
export function normalizeEvent(raw: RawEvent): SSEEventData | null {
  const type = (raw.type ?? raw.event) as SSEEventType | undefined;
  const data = raw.data;
  if (!type || data === undefined) return null;
  return { type, data } as SSEEventData;
}

/**
 * 将后端事件列表转为前端 SSEEventData[]
 */
export function normalizeEvents(rawList: unknown): SSEEventData[] {
  if (!Array.isArray(rawList)) return [];
  const out: SSEEventData[] = [];
  for (const raw of rawList) {
    const normalized = normalizeEvent(raw as RawEvent);
    if (normalized) out.push(normalized);
  }
  return out;
}

/** 时间线单项：用于渲染对话区的一条记录 */
export type TimelineItem =
  | { kind: "user"; id: string; data: ChatMessage }
  | { kind: "attachments"; id: string; role: "user" | "assistant"; files: AttachmentFile[] }
  | { kind: "assistant"; id: string; data: ChatMessage }
  | { kind: "tool"; id: string; data: ToolEvent; timeLabel?: string }
  | { kind: "step"; id: string; data: StepEvent; tools: ToolEvent[] }
  | { kind: "error"; id: string; error: string; timestamp?: number };

/** 附件展示用（文件名、类型、大小等） */
export type AttachmentFile = {
  id: string;
  filename: string;
  extension: string;
  size: number;
  sizeLabel?: string;
};

/** 从 SessionFile 转为 AttachmentFile */
export function sessionFileToAttachment(f: SessionFile): AttachmentFile {
  return {
    id: f.id,
    filename: f.filename,
    extension: f.extension,
    size: f.size,
  };
}

/** 从 ChatMessage.attachments 项转为 AttachmentFile（无 size 时用 0） */
export function chatAttachmentToDisplay(
  a: { file_id?: string; id?: string; filename: string; size?: number; [key: string]: unknown }
): AttachmentFile {
  const ext = (a.filename || "").split(".").pop() || "";
  return {
    id: a.file_id || a.id || "",
    filename: a.filename || "",
    extension: ext,
    size: typeof a.size === "number" ? a.size : 0,
  };
}

function stableId(prefix: string, index: number, suffix: string): string {
  return `${prefix}-${index}-${suffix}`;
}

export type TimelineBuildContext = {
  list: TimelineItem[];
  lastStepId: string | null;
  messageIndex: number;
  toolIndex: number;
  stepIndex: number;
  errorIndex: number;
  stepIndexById: Map<string, number>;
  stepToolIndexByStepId: Map<string, Map<string, number>>;
  standaloneToolIndexByCallId: Map<string, number>;
};

export function createTimelineBuildContext(): TimelineBuildContext {
  return {
    list: [],
    lastStepId: null,
    messageIndex: 0,
    toolIndex: 0,
    stepIndex: 0,
    errorIndex: 0,
    stepIndexById: new Map<string, number>(),
    stepToolIndexByStepId: new Map<string, Map<string, number>>(),
    standaloneToolIndexByCallId: new Map<string, number>(),
  };
}

/** 将时间戳格式化为相对时间，如 2天前、刚刚 */
function formatTimeLabel(ts: number | string | undefined): string | undefined {
  if (ts === undefined || ts === null) return undefined;
  let t = typeof ts === "string" ? parseInt(ts, 10) : ts;
  if (Number.isNaN(t)) return undefined;
  
  // 后端返回的是秒级时间戳（10位数），需要转为毫秒级（13位数）
  if (t < 10000000000) {
    t = t * 1000;
  }
  
  const now = Date.now();
  const diff = now - t;
  if (diff < 0) return "刚刚";
  if (diff < 60 * 1000) return "刚刚";
  if (diff < 60 * 60 * 1000) return `${Math.floor(diff / (60 * 1000))}分钟前`;
  if (diff < 24 * 60 * 60 * 1000) return `${Math.floor(diff / (60 * 60 * 1000))}小时前`;
  if (diff < 2 * 24 * 60 * 60 * 1000) return "昨天";
  if (diff < 7 * 24 * 60 * 60 * 1000) return `${Math.floor(diff / (24 * 60 * 60 * 1000))}天前`;
  if (diff < 30 * 24 * 60 * 60 * 1000) return `${Math.floor(diff / (7 * 24 * 60 * 60 * 1000))}周前`;
  return undefined;
}

export function getToolTimeLabel(tool: ToolEvent): string | undefined {
  const ts = (tool as { timestamp?: number; created_at?: number; ts?: number }).timestamp
    ?? (tool as { created_at?: number }).created_at
    ?? (tool as { ts?: number }).ts;
  return formatTimeLabel(ts);
}

function appendMessageEvent(context: TimelineBuildContext, msg: ChatMessage): void {
  if (msg.role === "user") {
    // 用户消息标志着新的对话轮次，清除 step 上下文
    context.lastStepId = null;

    context.list.push({
      kind: "user",
      id: stableId("user", context.messageIndex++, String(context.list.length)),
      data: msg,
    });
    if (msg.attachments && msg.attachments.length > 0) {
      context.list.push({
        kind: "attachments",
        id: stableId("att", context.messageIndex, "user"),
        role: "user",
        files: msg.attachments.map(chatAttachmentToDisplay),
      });
    }
    return;
  }

  if (msg.role === "assistant") {
    context.list.push({
      kind: "assistant",
      id: stableId("assistant", context.messageIndex++, String(context.list.length)),
      data: msg,
    });
    if (msg.attachments && msg.attachments.length > 0) {
      context.list.push({
        kind: "attachments",
        id: stableId("att", context.messageIndex, "assistant"),
        role: "assistant",
        files: msg.attachments.map(chatAttachmentToDisplay),
      });
    }
  }
}

function appendStepEvent(context: TimelineBuildContext, step: StepEvent): void {
  const shouldUpdateCurrentStep = context.lastStepId !== null && context.lastStepId === step.id;
  const existingIdx = context.stepIndexById.get(step.id);

  if (shouldUpdateCurrentStep && existingIdx !== undefined) {
    const existing = context.list[existingIdx];
    if (existing?.kind === "step") {
      context.list[existingIdx] = {
        kind: "step",
        id: existing.id,
        data: step,
        tools: existing.tools, // 保留已有的 tools
      };
    } else {
      const newIdx = context.list.length;
      context.list.push({
        kind: "step",
        id: stableId("step", context.stepIndex++, step.id + "_" + String(context.list.length)),
        data: step,
        tools: [],
      });
      context.stepIndexById.set(step.id, newIdx);
      context.stepToolIndexByStepId.set(step.id, new Map<string, number>());
    }
  } else {
    const newIdx = context.list.length;
    context.list.push({
      kind: "step",
      id: stableId("step", context.stepIndex++, step.id + "_" + String(context.list.length)),
      data: step,
      tools: [],
    });
    context.stepIndexById.set(step.id, newIdx);
    // 相同 step.id 在新轮次中可能复用，需要清理旧索引
    context.stepToolIndexByStepId.set(step.id, new Map<string, number>());
  }

  // 只要 step 不是 completed/failed 状态，就保持跟踪
  if (step.status === "completed" || step.status === "failed") {
    context.lastStepId = null;
  } else {
    context.lastStepId = step.id;
  }
}

function appendToolEvent(context: TimelineBuildContext, tool: ToolEvent): void {
  const toolCallId = (tool as { tool_call_id?: string }).tool_call_id;
  const activeStepId = context.lastStepId;

  if (activeStepId !== null) {
    const stepIdx = context.stepIndexById.get(activeStepId);
    if (stepIdx !== undefined) {
      const stepItem = context.list[stepIdx];
      if (stepItem?.kind === "step") {
        if (toolCallId != null) {
          const stepToolIndexes = context.stepToolIndexByStepId.get(activeStepId) ?? new Map<string, number>();
          context.stepToolIndexByStepId.set(activeStepId, stepToolIndexes);
          const existingToolIdx = stepToolIndexes.get(toolCallId);
          if (
            existingToolIdx !== undefined &&
            existingToolIdx >= 0 &&
            existingToolIdx < stepItem.tools.length
          ) {
            const newTools = [...stepItem.tools];
            newTools[existingToolIdx] = tool;
            context.list[stepIdx] = { ...stepItem, tools: newTools };
            return;
          }
          const nextToolIdx = stepItem.tools.length;
          context.list[stepIdx] = { ...stepItem, tools: [...stepItem.tools, tool] };
          stepToolIndexes.set(toolCallId, nextToolIdx);
          return;
        }

        context.list[stepIdx] = { ...stepItem, tools: [...stepItem.tools, tool] };
        return;
      }
    }
  }

  if (toolCallId != null) {
    const existingStandaloneIdx = context.standaloneToolIndexByCallId.get(toolCallId);
    if (existingStandaloneIdx !== undefined) {
      const existingItem = context.list[existingStandaloneIdx];
      if (existingItem?.kind === "tool") {
        context.list[existingStandaloneIdx] = { ...existingItem, data: tool };
        return;
      }
    }
  }

  const nextIdx = context.list.length;
  context.list.push({
    kind: "tool",
    id: stableId("tool", context.toolIndex++, (tool.name || "") + (tool.function || "")),
    data: tool,
    timeLabel: getToolTimeLabel(tool),
  });
  if (toolCallId != null) {
    context.standaloneToolIndexByCallId.set(toolCallId, nextIdx);
  }
}

function appendErrorEvent(context: TimelineBuildContext, data: unknown): void {
  const errorData = data as { error?: string; created_at?: number; event_id?: string; [key: string]: unknown };
  if (!errorData.error) return;
  context.list.push({
    kind: "error",
    id: stableId("error", context.errorIndex++, String(context.list.length)),
    error: errorData.error,
    timestamp: errorData.created_at,
  });
}

/**
 * 将单条 SSE 事件增量归并到时间线 context。
 * 适用于 append-only 的事件流场景，可避免每次全量重算。
 */
export function appendTimelineEvent(context: TimelineBuildContext, ev: SSEEventData): void {
  switch (ev.type) {
    case "message":
      appendMessageEvent(context, ev.data as ChatMessage);
      break;
    case "step":
      appendStepEvent(context, ev.data as StepEvent);
      break;
    case "tool":
      appendToolEvent(context, ev.data as ToolEvent);
      break;
    case "error":
      appendErrorEvent(context, ev.data);
      break;
    case "title":
    case "plan":
    case "wait":
    case "done":
      break;
    default:
      break;
  }
}

function cloneTimelineBuildContext(source: TimelineBuildContext): TimelineBuildContext {
  return {
    list: [...source.list],
    lastStepId: source.lastStepId,
    messageIndex: source.messageIndex,
    toolIndex: source.toolIndex,
    stepIndex: source.stepIndex,
    errorIndex: source.errorIndex,
    stepIndexById: new Map(source.stepIndexById),
    stepToolIndexByStepId: new Map(
      Array.from(source.stepToolIndexByStepId.entries()).map(([stepId, indexMap]) => [stepId, new Map(indexMap)])
    ),
    standaloneToolIndexByCallId: new Map(source.standaloneToolIndexByCallId),
  };
}

let timelineCacheEvents: SSEEventData[] | null = null;
let timelineCacheContext: TimelineBuildContext | null = null;

/**
 * 将 SSE 事件列表归并为时间线展示项（顺序与设计一致）
 */
export function eventsToTimeline(events: SSEEventData[]): TimelineItem[] {
  const cachedEvents = timelineCacheEvents;
  const cachedContext = timelineCacheContext;

  const isAppendOnly =
    cachedEvents !== null &&
    cachedContext !== null &&
    events.length > cachedEvents.length &&
    (cachedEvents.length === 0 ||
      (
        events[0] === cachedEvents[0] &&
        events[cachedEvents.length - 1] === cachedEvents[cachedEvents.length - 1]
      ));

  const context = isAppendOnly && cachedContext
    ? cloneTimelineBuildContext(cachedContext)
    : createTimelineBuildContext();

  const startIndex = isAppendOnly && cachedEvents ? cachedEvents.length : 0;
  for (let i = startIndex; i < events.length; i++) {
    const ev = events[i];
    appendTimelineEvent(context, ev);
  }

  timelineCacheEvents = events;
  timelineCacheContext = context;
  return context.list;
}

/**
 * 从事件列表中取最新的 plan 步骤（用于底部任务进度面板）
 */
export function getLatestPlanFromEvents(events: SSEEventData[]): PlanStep[] {
  let steps: PlanStep[] = [];
  for (let i = events.length - 1; i >= 0; i--) {
    const ev = events[i];
    if (ev.type === "plan") {
      const plan = ev.data as PlanEvent;
      if (plan.steps && Array.isArray(plan.steps)) {
        steps = plan.steps;
      }
      break;
    }
  }
  return steps;
}
