import type { SSEEventData, PlanEvent, StepEvent, StepOutcome, ToolEvent, WaitEventData } from './api/types'
import type { AppLocale } from './i18n'
import { getFriendlyToolLabel } from '../components/tool-use/utils'
import { visitSessionEvent } from './session-event-adapter'
import { parseWaitEventContext, type WaitEventContext } from './wait-event'

export type RunTimelineKind = 'message' | 'plan' | 'step' | 'tool' | 'wait' | 'error' | 'done'

export type RunTimelineItem = {
  id: string
  kind: RunTimelineKind
  summary: string
  timestamp: number | null
}

export type StepViewStatus = 'pending' | 'running' | 'waiting' | 'completed' | 'failed'

export type StepViewItem = {
  id: string
  description: string
  status: StepViewStatus
  stepIndex: number
  detail: string | null
}

export type StepViewState = {
  steps: StepViewItem[]
  completedCount: number
  totalCount: number
}

function parseTimestamp(raw: unknown): number | null {
  if (typeof raw !== 'number' || Number.isNaN(raw) || raw <= 0) return null
  // 兼容秒级时间戳
  return raw < 10000000000 ? raw * 1000 : raw
}

function pickTimestamp(event: SSEEventData): number | null {
  const data = event.data as { created_at?: unknown; timestamp?: unknown; ts?: unknown }
  return (
    parseTimestamp(data.created_at)
    ?? parseTimestamp(data.timestamp)
    ?? parseTimestamp(data.ts)
  )
}

function truncate(text: string, limit: number): string {
  if (text.length <= limit) return text
  return `${text.slice(0, limit)}...`
}

function toNonEmptyText(raw: unknown): string | null {
  if (typeof raw !== 'string') return null
  const trimmed = raw.trim()
  return trimmed.length > 0 ? trimmed : null
}

export function resolveWaitDisplayText(data: WaitEventData): string | null {
  const context = parseWaitEventContext(data)
  return context?.prompt ?? null
}

export function findLatestWaitEventContext(events: SSEEventData[]): WaitEventContext | null {
  for (let i = events.length - 1; i >= 0; i--) {
    const event = events[i]
    if (event.type !== 'wait') continue
    return parseWaitEventContext(event.data as WaitEventData)
  }
  return null
}

function messageSummary(data: { role?: string; message?: string }, locale: AppLocale): string {
  const role = data.role === 'assistant'
    ? (locale === 'en-US' ? 'Assistant' : '助手')
    : (locale === 'en-US' ? 'User' : '用户')
  const message = typeof data.message === 'string' && data.message.trim().length > 0
    ? data.message.trim()
    : (locale === 'en-US' ? '(empty message)' : '（空消息）')
  return `${role}: ${truncate(message, 80)}`
}

function planSummary(data: PlanEvent, locale: AppLocale): string {
  const title = typeof data.title === 'string' ? data.title.trim() : ''
  const goal = typeof data.goal === 'string' ? data.goal.trim() : ''
  const message = typeof data.message === 'string' ? data.message.trim() : ''
  const steps = Array.isArray(data.steps) ? data.steps.length : 0
  if (title) return truncate(title, 80)
  if (goal) return truncate(goal, 80)
  if (message) return truncate(message, 80)
  return locale === 'en-US'
    ? `Plan updated · ${steps} step(s)`
    : `计划已更新 · ${steps} 个步骤`
}

function stepSummary(data: StepEvent, locale: AppLocale): string {
  const detail = resolveStepDetail(data.outcome)
  if (detail) return truncate(detail, 80)
  const description = typeof data.description === 'string' ? data.description.trim() : ''
  if (description) return truncate(description, 80)
  return locale === 'en-US' ? `Step ${data.id}` : `步骤 ${data.id}`
}

export function resolveStepDetail(outcome: StepOutcome | null | undefined): string | null {
  if (!outcome) return null
  const summary = toNonEmptyText(outcome.summary)
  if (summary) return summary
  if (Array.isArray(outcome.blockers)) {
    const blocker = outcome.blockers.find((item) => typeof item === 'string' && item.trim().length > 0)
    if (blocker) return blocker.trim()
  }
  return toNonEmptyText(outcome.next_hint)
}

function waitSummary(data: WaitEventData, locale: AppLocale): string {
  const question = resolveWaitDisplayText(data)
  if (question) return truncate(question, 80)
  return locale === 'en-US' ? 'Waiting for your input' : '等待你的输入'
}

function errorSummary(data: Record<string, unknown>, locale: AppLocale): string {
  const error = typeof data.error === 'string' ? data.error : ''
  const errorKey = typeof data.error_key === 'string' ? data.error_key : ''
  if (error.trim().length > 0) return truncate(error.trim(), 80)
  if (errorKey.trim().length > 0) return truncate(errorKey.trim(), 80)
  return locale === 'en-US' ? 'Runtime error' : '运行时错误'
}

function doneSummary(data: Record<string, unknown>, locale: AppLocale): string {
  const message = typeof data.message === 'string' ? data.message.trim() : ''
  const result = typeof data.result === 'string' ? data.result.trim() : ''
  if (message) return truncate(message, 80)
  if (result) return truncate(result, 80)
  return locale === 'en-US' ? 'Run completed' : '运行完成'
}

export function buildRunTimeline(events: SSEEventData[], locale: AppLocale = 'zh-CN'): RunTimelineItem[] {
  const items: RunTimelineItem[] = []
  for (let i = 0; i < events.length; i++) {
    const event = events[i]
    const base = {
      id: String((event.data as { event_id?: string }).event_id ?? `${event.type}-${i}`),
      timestamp: pickTimestamp(event),
    }

    visitSessionEvent(event, {
      message: (messageEvent) => {
        items.push({
          ...base,
          kind: 'message',
          summary: messageSummary(messageEvent.data as { role?: string; message?: string }, locale),
        })
      },
      plan: (planEvent) => {
        items.push({
          ...base,
          kind: 'plan',
          summary: planSummary(planEvent.data as PlanEvent, locale),
        })
      },
      step: (stepEvent) => {
        items.push({
          ...base,
          kind: 'step',
          summary: stepSummary(stepEvent.data as StepEvent, locale),
        })
      },
      tool: (toolEvent) => {
        items.push({
          ...base,
          kind: 'tool',
          summary: getFriendlyToolLabel(toolEvent.data as ToolEvent, locale),
        })
      },
      wait: (waitEvent) => {
        items.push({
          ...base,
          kind: 'wait',
          summary: waitSummary(waitEvent.data as WaitEventData, locale),
        })
      },
      error: (errorEvent) => {
        items.push({
          ...base,
          kind: 'error',
          summary: errorSummary(errorEvent.data as Record<string, unknown>, locale),
        })
      },
      done: (doneEvent) => {
        items.push({
          ...base,
          kind: 'done',
          summary: doneSummary(doneEvent.data as Record<string, unknown>, locale),
        })
      },
    })
  }
  return items
}

function normalizeStepStatus(raw: unknown): StepViewStatus {
  if (raw === 'completed') return 'completed'
  if (raw === 'failed') return 'failed'
  if (raw === 'waiting') return 'waiting'
  if (raw === 'running') return 'running'
  return 'pending'
}

export function buildStepViewState(events: SSEEventData[]): StepViewState {
  const stepMap = new Map<string, StepViewItem>()
  const orderedStepIds: string[] = []

  for (const event of events) {
    visitSessionEvent(event, {
      plan: (planEvent) => {
        const plan = planEvent.data as PlanEvent
        if (!Array.isArray(plan.steps)) return

        stepMap.clear()
        orderedStepIds.length = 0
        for (let i = 0; i < plan.steps.length; i++) {
          const step = plan.steps[i]
          if (!step?.id) continue
          stepMap.set(step.id, {
            id: step.id,
            description: step.description ?? step.id,
            status: normalizeStepStatus(step.status),
            stepIndex: i,
            detail: resolveStepDetail(step.outcome),
          })
          orderedStepIds.push(step.id)
        }
      },
      step: (stepEvent) => {
        const step = stepEvent.data as StepEvent
        if (!step.id) return
        const existing = stepMap.get(step.id)
        if (existing) {
          stepMap.set(step.id, {
            ...existing,
            description: step.description || existing.description,
            status: normalizeStepStatus(step.status),
            detail: resolveStepDetail(step.outcome) ?? existing.detail,
          })
          return
        }
        stepMap.set(step.id, {
          id: step.id,
          description: step.description || step.id,
          status: normalizeStepStatus(step.status),
          stepIndex: orderedStepIds.length,
          detail: resolveStepDetail(step.outcome),
        })
        orderedStepIds.push(step.id)
      },
    })
  }

  const steps = orderedStepIds
    .map((stepId) => stepMap.get(stepId))
    .filter((step): step is StepViewItem => Boolean(step))
    .sort((a, b) => a.stepIndex - b.stepIndex)

  const completedCount = steps.filter((step) => step.status === 'completed').length
  return {
    steps,
    completedCount,
    totalCount: steps.length,
  }
}
