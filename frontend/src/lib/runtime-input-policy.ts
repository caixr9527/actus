import type { SSEEventData } from './api/types'
import type { EventRuntimeContext } from './session-event-adapter'
import { adaptSessionEvent } from './session-event-adapter'

const INPUT_PART_TYPES_ORDER = ['text', 'file_ref', 'image', 'audio', 'pdf']

export type RuntimeInputPartSummary = {
  total: number
  byType: Record<string, number>
}

export type RuntimeUnsupportedPart = {
  type: string
  filepath: string | null
  reason: string | null
}

export type RuntimeInputPolicySnapshot = {
  inputPartSummary: RuntimeInputPartSummary | null
  downgradeReason: string | null
  unsupportedParts: RuntimeUnsupportedPart[]
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function toNonEmptyString(value: unknown): string | null {
  if (typeof value !== 'string') return null
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : null
}

function parseInputPartSummary(value: unknown): RuntimeInputPartSummary | null {
  if (!isRecord(value)) return null
  const total = Number(value.total)
  const byTypeRaw = value.by_type
  if (!Number.isFinite(total) || !isRecord(byTypeRaw)) return null

  const byType: Record<string, number> = {}
  for (const [key, rawCount] of Object.entries(byTypeRaw)) {
    const count = Number(rawCount)
    if (!Number.isFinite(count) || count <= 0) continue
    byType[key] = Math.floor(count)
  }

  return {
    total: Math.max(0, Math.floor(total)),
    byType,
  }
}

function parseUnsupportedParts(value: unknown): RuntimeUnsupportedPart[] {
  if (!Array.isArray(value)) return []
  const parts: RuntimeUnsupportedPart[] = []
  for (const item of value) {
    if (!isRecord(item)) continue
    parts.push({
      type: toNonEmptyString(item.type) ?? 'unknown',
      filepath: toNonEmptyString(item.filepath),
      reason: toNonEmptyString(item.reason),
    })
  }
  return parts
}

export function parseRuntimeInputPolicyFromContext(
  runtimeContext: EventRuntimeContext | null | undefined,
): RuntimeInputPolicySnapshot {
  if (!runtimeContext || !isRecord(runtimeContext)) {
    return {
      inputPartSummary: null,
      downgradeReason: null,
      unsupportedParts: [],
    }
  }

  return {
    inputPartSummary: parseInputPartSummary(runtimeContext.input_part_summary),
    downgradeReason: toNonEmptyString(runtimeContext.downgrade_reason),
    unsupportedParts: parseUnsupportedParts(runtimeContext.unsupported_parts),
  }
}

export function hasRuntimeInputPolicySignal(snapshot: RuntimeInputPolicySnapshot): boolean {
  return Boolean(snapshot.inputPartSummary || snapshot.downgradeReason || snapshot.unsupportedParts.length > 0)
}

export function pickLatestRuntimeInputPolicySnapshot(events: SSEEventData[]): RuntimeInputPolicySnapshot {
  for (let i = events.length - 1; i >= 0; i--) {
    const adapted = adaptSessionEvent(events[i])
    const snapshot = parseRuntimeInputPolicyFromContext(adapted.runtimeContext)
    if (hasRuntimeInputPolicySignal(snapshot)) return snapshot
  }
  return {
    inputPartSummary: null,
    downgradeReason: null,
    unsupportedParts: [],
  }
}

export function formatRuntimeInputPartSummary(summary: RuntimeInputPartSummary): string {
  const entries = Object.entries(summary.byType).sort(([a], [b]) => {
    const indexA = INPUT_PART_TYPES_ORDER.indexOf(a)
    const indexB = INPUT_PART_TYPES_ORDER.indexOf(b)
    if (indexA !== -1 && indexB !== -1) return indexA - indexB
    if (indexA !== -1) return -1
    if (indexB !== -1) return 1
    return a.localeCompare(b)
  })
  const detail = entries.map(([type, count]) => `${type}×${count}`).join(', ')
  return detail.length > 0 ? `${summary.total} (${detail})` : String(summary.total)
}
