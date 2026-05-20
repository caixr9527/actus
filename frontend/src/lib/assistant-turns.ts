import type { ChatMessage, ExecutionStatus, SessionStatus } from './api/types'
import type { TimelineItem } from './session-events'

export type AssistantTurnStatus = 'running' | 'waiting' | 'completed' | 'failed' | 'cancelled' | 'idle'

export type AssistantTurnRuntimeOptions = {
  currentRunId?: string | null
  runtimeStatus?: SessionStatus | null
}

export type AssistantTurnItem = {
  kind: 'assistant_turn'
  id: string
  processItems: Array<Extract<TimelineItem, { kind: 'step' | 'tool' | 'error' | 'assistant' }>>
  finalMessage: Extract<TimelineItem, { kind: 'assistant' }> | null
  attachments: Array<Extract<TimelineItem, { kind: 'attachments' }>>
  status: AssistantTurnStatus
  stepCount: number
  completedStepCount: number
  toolCount: number
}

export type ConversationItem =
  | TimelineItem
  | Extract<TimelineItem, { kind: 'assistant' }>
  | AssistantTurnItem

function isAssistantIntermediateMessage(item: Extract<TimelineItem, { kind: 'assistant' }>): boolean {
  const data = item.data as ChatMessage
  return data.stage === 'intermediate'
}

function isFinalAssistantMessage(item: Extract<TimelineItem, { kind: 'assistant' }>): boolean {
  return !isAssistantIntermediateMessage(item)
}

function rankStepStatus(status: ExecutionStatus | undefined): number {
  if (status === 'failed') return 5
  if (status === 'cancelled') return 4
  if (status === 'running') return 2
  if (status === 'completed') return 1
  return 0
}

function resolveTurnStatus(
  processItems: AssistantTurnItem['processItems'],
  finalMessage: AssistantTurnItem['finalMessage'],
  runtimeOptions: AssistantTurnRuntimeOptions = {},
): AssistantTurnStatus {
  const runId = resolveTurnRunId(processItems, finalMessage)
  const runtimeStatus = resolveRuntimeStatusForTurn(runId, runtimeOptions)
  if (runtimeStatus) return runtimeStatus

  let strongest: ExecutionStatus | undefined
  for (const item of processItems) {
    if (item.kind === 'error') return 'failed'
    if (item.kind !== 'step') continue
    if (rankStepStatus(item.data.status) > rankStepStatus(strongest)) {
      strongest = item.data.status
    }
  }

  if (strongest === 'failed') return 'failed'
  if (strongest === 'cancelled') return 'cancelled'
  if (strongest === 'running') return 'running'
  if (finalMessage || strongest === 'completed') return 'completed'
  return 'idle'
}

function normalizeRuntimeStatus(status: SessionStatus | null | undefined): AssistantTurnStatus | null {
  if (status === 'pending') return 'idle'
  return status ?? null
}

function resolveRuntimeStatusForTurn(
  runId: string | null,
  runtimeOptions: AssistantTurnRuntimeOptions,
): AssistantTurnStatus | null {
  const currentRunId = runtimeOptions.currentRunId ?? null
  if (!currentRunId || !runId || currentRunId !== runId) return null
  return normalizeRuntimeStatus(runtimeOptions.runtimeStatus)
}

function extractRunId(item: AssistantTurnItem['processItems'][number] | AssistantTurnItem['finalMessage']): string | null {
  if (!item) return null
  if (item.kind === 'error') return null
  return item.data.runtime.run_id ?? null
}

function resolveTurnRunId(
  processItems: AssistantTurnItem['processItems'],
  finalMessage: AssistantTurnItem['finalMessage'],
): string | null {
  if (finalMessage) return extractRunId(finalMessage)
  for (const item of processItems) {
    const runId = extractRunId(item)
    if (runId) return runId
  }
  return null
}

function buildAssistantTurn(
  index: number,
  processItems: AssistantTurnItem['processItems'],
  finalMessage: AssistantTurnItem['finalMessage'],
  attachments: AssistantTurnItem['attachments'] = [],
  runtimeOptions: AssistantTurnRuntimeOptions = {},
): AssistantTurnItem {
  const stepItems = processItems.filter((item): item is Extract<TimelineItem, { kind: 'step' }> => item.kind === 'step')
  const standaloneToolCount = processItems.filter((item) => item.kind === 'tool').length
  const nestedToolCount = stepItems.reduce((count, item) => count + item.tools.length, 0)
  const idSource = finalMessage?.id ?? processItems[0]?.id ?? attachments[0]?.id ?? String(index)

  return {
    kind: 'assistant_turn',
    id: `assistant-turn-${index}-${idSource}`,
    processItems,
    finalMessage,
    attachments,
    status: resolveTurnStatus(processItems, finalMessage, runtimeOptions),
    stepCount: stepItems.length,
    completedStepCount: stepItems.filter((item) => item.data.status === 'completed').length,
    toolCount: standaloneToolCount + nestedToolCount,
  }
}

export function timelineToConversationItems(
  timeline: TimelineItem[],
  runtimeOptions: AssistantTurnRuntimeOptions = {},
): ConversationItem[] {
  const result: ConversationItem[] = []
  let processBuffer: AssistantTurnItem['processItems'] = []
  let turnIndex = 0

  const flushProcessBuffer = () => {
    if (processBuffer.length === 0) return
    result.push(buildAssistantTurn(turnIndex++, processBuffer, null, [], runtimeOptions))
    processBuffer = []
  }

  for (let i = 0; i < timeline.length; i++) {
    const item = timeline[i]

    if (item.kind === 'user') {
      flushProcessBuffer()
      result.push(item)
      continue
    }

    if (item.kind === 'step' || item.kind === 'tool' || item.kind === 'error') {
      processBuffer.push(item)
      continue
    }

    if (item.kind === 'assistant') {
      if (isAssistantIntermediateMessage(item)) {
        processBuffer.push(item)
        continue
      }

      if (processBuffer.length > 0 && isFinalAssistantMessage(item)) {
        const attachments: AssistantTurnItem['attachments'] = []
        const next = timeline[i + 1]
        if (next?.kind === 'attachments' && next.role === 'assistant') {
          attachments.push(next)
          i += 1
        }
        result.push(buildAssistantTurn(turnIndex++, processBuffer, item, attachments, runtimeOptions))
        processBuffer = []
        continue
      }

      result.push(item)
      continue
    }

    flushProcessBuffer()
    result.push(item)
  }

  flushProcessBuffer()
  return result
}
