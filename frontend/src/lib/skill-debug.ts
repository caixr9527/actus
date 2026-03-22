import type { SSEEventData } from './api/types'
import { adaptSessionEvent } from './session-event-adapter'

type SkillDebugSource = Record<string, unknown>

export type SkillDebugItem = {
  id: string
  eventType: SSEEventData['type']
  semanticType: string
  timestamp: number | null
  skillId: string | null
  skillVersion: string | null
  subgraph: string | null
  node: string | null
  failedNode: string | null
  status: string | null
  stepId: string | null
  input: unknown | null
  output: unknown | null
  error: string | null
}

const SKILL_KEYWORD_HINTS = ['skill', 'subgraph']

function asRecord(value: unknown): SkillDebugSource | null {
  if (typeof value !== 'object' || value === null) return null
  return value as SkillDebugSource
}

function toNonEmptyString(value: unknown): string | null {
  if (typeof value !== 'string') return null
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : null
}

function parseTimestamp(raw: unknown): number | null {
  if (typeof raw !== 'number' || Number.isNaN(raw) || raw <= 0) return null
  return raw < 10000000000 ? raw * 1000 : raw
}

function parseOptionalTimestamp(raw: unknown): number | null {
  if (typeof raw === 'number') return parseTimestamp(raw)
  if (typeof raw === 'string' && raw.trim().length > 0) {
    const parsed = Number(raw)
    if (!Number.isNaN(parsed)) return parseTimestamp(parsed)
  }
  return null
}

function pickTimestamp(data: SkillDebugSource): number | null {
  return (
    parseOptionalTimestamp(data.created_at)
    ?? parseOptionalTimestamp(data.timestamp)
    ?? parseOptionalTimestamp(data.ts)
  )
}

function hasSkillHints(source: SkillDebugSource): boolean {
  return Object.keys(source).some((key) => {
    const normalized = key.toLowerCase()
    return SKILL_KEYWORD_HINTS.some((hint) => normalized.includes(hint))
  })
}

function normalizePayload(value: unknown): unknown | null {
  if (value === null || value === undefined) return null
  if (typeof value === 'string') return toNonEmptyString(value)
  if (Array.isArray(value)) return value.length > 0 ? value : null
  if (asRecord(value)) return Object.keys(value as SkillDebugSource).length > 0 ? value : null
  return value
}

function pickFirstString(sources: SkillDebugSource[], keys: string[]): string | null {
  for (const source of sources) {
    for (const key of keys) {
      const value = toNonEmptyString(source[key])
      if (value) return value
    }
  }
  return null
}

function pickFirstPayload(sources: SkillDebugSource[], keys: string[]): unknown | null {
  for (const source of sources) {
    for (const key of keys) {
      const value = normalizePayload(source[key])
      if (value !== null) return value
    }
  }
  return null
}

function collectSources(data: SkillDebugSource): SkillDebugSource[] {
  const sources: SkillDebugSource[] = []
  const seen = new Set<SkillDebugSource>()

  const addSource = (value: unknown): void => {
    const source = asRecord(value)
    if (!source || seen.has(source)) return
    seen.add(source)
    sources.push(source)
  }

  addSource(data)
  addSource(data.runtime)
  addSource(data.skill)
  addSource(data.skill_runtime)
  addSource(data.skill_debug)
  addSource(data.subgraph)

  const extensions = asRecord(data.extensions)
  addSource(extensions)
  const runtime = asRecord(extensions?.runtime)
  addSource(runtime)
  addSource(runtime?.skill)
  addSource(runtime?.skill_runtime)
  addSource(runtime?.subgraph)

  const skill = asRecord(data.skill)
  addSource(skill?.runtime)
  addSource(skill?.subgraph)

  return sources
}

function collectSkillSpecificSources(data: SkillDebugSource): SkillDebugSource[] {
  const sources: SkillDebugSource[] = []
  const seen = new Set<SkillDebugSource>()

  const addSource = (value: unknown): void => {
    const source = asRecord(value)
    if (!source || seen.has(source)) return
    seen.add(source)
    sources.push(source)
  }

  addSource(data.skill)
  addSource(data.skill_runtime)
  addSource(data.skill_debug)
  addSource(data.subgraph)

  const extensions = asRecord(data.extensions)
  const runtime = asRecord(extensions?.runtime)
  addSource(runtime?.skill)
  addSource(runtime?.skill_runtime)
  addSource(runtime?.subgraph)

  return sources
}

function buildSkillDebugItem(event: SSEEventData, index: number): SkillDebugItem | null {
  const data = asRecord(event.data)
  if (!data) return null

  const sources = collectSources(data)
  const skillSources = collectSkillSpecificSources(data)
  const adapted = adaptSessionEvent(event)

  const skillId = (
    pickFirstString(skillSources, ['skill_id', 'skillId', 'id'])
    ?? pickFirstString(sources, ['skill_id', 'skillId'])
  )
  const skillVersion = (
    pickFirstString(skillSources, ['skill_version', 'skillVersion', 'version'])
    ?? pickFirstString(sources, ['skill_version', 'skillVersion'])
  )
  const subgraph = (
    pickFirstString(skillSources, ['subgraph', 'subgraph_id', 'subgraphId', 'graph', 'graph_id'])
    ?? pickFirstString(sources, ['subgraph', 'subgraph_id', 'subgraphId'])
  )
  const node = pickFirstString(sources, ['node', 'node_id', 'nodeId', 'node_name', 'nodeName'])
  const failedNode = pickFirstString(sources, ['failed_node', 'failedNode', 'error_node', 'errorNode'])
  const status = pickFirstString(sources, ['status', 'event_status', 'phase'])
  const stepId = pickFirstString(sources, ['step_id', 'stepId', 'id'])
  const hasSkillHintsInSource = sources.some((source) => hasSkillHints(source))
  const hintedInput = hasSkillHintsInSource ? pickFirstPayload(sources, ['input', 'payload', 'request']) : null
  const hintedOutput = hasSkillHintsInSource ? pickFirstPayload(sources, ['output', 'result', 'response']) : null
  const input = (
    pickFirstPayload(skillSources, ['skill_input', 'input', 'payload', 'request'])
    ?? pickFirstPayload(sources, ['skill_input'])
    ?? hintedInput
  )
  const output = (
    pickFirstPayload(skillSources, ['skill_output', 'output', 'result', 'response'])
    ?? pickFirstPayload(sources, ['skill_output'])
    ?? hintedOutput
  )
  const skillError = (
    pickFirstString(skillSources, ['error', 'error_message', 'errorMessage', 'reason'])
    ?? pickFirstString(sources, ['skill_error', 'skill_error_message'])
  )
  const error = skillError ?? (hasSkillHintsInSource ? pickFirstString(sources, ['error', 'error_message', 'errorMessage']) : null)
  const hasSkillSignal = (
    Boolean(skillId)
    || Boolean(skillVersion)
    || Boolean(subgraph)
    || Boolean(node)
    || Boolean(failedNode)
    || input !== null
    || output !== null
    || hasSkillHintsInSource
    || adapted.semanticType.startsWith('skill.')
  )
  if (!hasSkillSignal) return null

  const eventId = toNonEmptyString(data.event_id)
  return {
    id: eventId ?? `skill-debug-${index}`,
    eventType: event.type,
    semanticType: adapted.semanticType,
    timestamp: pickTimestamp(data),
    skillId,
    skillVersion,
    subgraph,
    node,
    failedNode,
    status,
    stepId,
    input,
    output,
    error,
  }
}

/**
 * 从会话事件中提取 Skill/Subgraph 调试条目。
 * 字段缺失时返回空数组，避免影响既有会话详情渲染。
 */
export function buildSkillDebugItems(events: SSEEventData[]): SkillDebugItem[] {
  const items: SkillDebugItem[] = []
  for (let index = 0; index < events.length; index++) {
    const item = buildSkillDebugItem(events[index], index)
    if (item) items.push(item)
  }
  return items
}
