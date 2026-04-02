import type { SessionStatus, StepEvent } from './api/types'

export function shouldAutoExpandStep(status: StepEvent['status']): boolean {
  return status === 'running'
}

export function resolveStepExpandedState(params: {
  currentExpanded: boolean
  previousStatus: StepEvent['status'] | null
  nextStatus: StepEvent['status']
}): boolean {
  const { currentExpanded, previousStatus, nextStatus } = params
  if (nextStatus === 'running') return true
  if (previousStatus === 'running') return false
  return currentExpanded
}

export function shouldAutoCloseTaskPreview(
  previousStatus: SessionStatus | null | undefined,
  nextStatus: SessionStatus | null | undefined,
): boolean {
  return previousStatus === 'running' && nextStatus === 'completed'
}

export function shouldShowSessionThinking(params: {
  streaming: boolean
  sessionStatus: SessionStatus | null | undefined
  hasInitialMessage: boolean
  timelineLength: number
  hasError: boolean
  hasRunningStep: boolean
}): boolean {
  const {
    streaming,
    sessionStatus,
    hasInitialMessage,
    timelineLength,
    hasError,
    hasRunningStep,
  } = params

  if (hasRunningStep) return false

  return (
    streaming
    || sessionStatus === 'running'
    || (hasInitialMessage && timelineLength === 0 && !hasError)
  )
}

export function shouldAutoScrollToLatest(params: {
  lastAutoScrolledSessionId: string | null
  sessionId: string
  timelineLength: number
  shouldShowThinking: boolean
}): boolean {
  const { lastAutoScrolledSessionId, sessionId, timelineLength, shouldShowThinking } = params
  if (lastAutoScrolledSessionId === sessionId) return false
  return timelineLength > 0 || shouldShowThinking
}
