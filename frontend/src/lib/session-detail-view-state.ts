import type { RuntimeCapabilities, RuntimeInteraction, SessionStatus, StepEvent } from './api/types'
import type { SSEEventData } from './api/types'
import { findLatestWaitEventContext } from './run-timeline'
import { waitEventContextFromRuntimeInteraction, type WaitEventContext } from './wait-event'

export type SessionScopedDetailViewState<TFile = unknown, TTool = unknown> = {
  fileListOpen: boolean
  previewFile: TFile | null
  previewTool: TTool | null
  timelineExpanded: boolean
  vncOpen: boolean
}

export type SessionScopedRuntimeState = {
  initialMessageSent: boolean
  previousToolCount: number
  hasAutoScrolled: boolean
  previousSessionStatus: SessionStatus | null
}

export type SessionActionAvailability = {
  canSendMessage: boolean
  canResume: boolean
  canCancel: boolean
  canContinueCancelled: boolean
}

export function createSessionScopedDetailViewState<TFile = unknown, TTool = unknown>(): SessionScopedDetailViewState<TFile, TTool> {
  return {
    fileListOpen: false,
    previewFile: null,
    previewTool: null,
    timelineExpanded: false,
    vncOpen: false,
  }
}

export function createSessionScopedRuntimeState(): SessionScopedRuntimeState {
  return {
    initialMessageSent: false,
    previousToolCount: 0,
    hasAutoScrolled: false,
    previousSessionStatus: null,
  }
}

export function resolveSessionActionAvailability(
  capabilities: RuntimeCapabilities | null | undefined,
): SessionActionAvailability {
  return {
    canSendMessage: Boolean(capabilities?.can_send_message),
    canResume: Boolean(capabilities?.can_resume),
    canCancel: Boolean(capabilities?.can_cancel),
    canContinueCancelled: Boolean(capabilities?.can_continue_cancelled),
  }
}

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
  return (
    previousStatus === 'running'
    && (nextStatus === 'completed' || nextStatus === 'failed' || nextStatus === 'cancelled')
  )
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
  hasAutoScrolled: boolean
  timelineLength: number
  shouldShowThinking: boolean
}): boolean {
  const { hasAutoScrolled, timelineLength, shouldShowThinking } = params
  if (hasAutoScrolled) return false
  return timelineLength > 0 || shouldShowThinking
}

export function shouldHideWaitResumeCard(params: {
  sessionStatus: SessionStatus | null | undefined
  waitContextAvailable: boolean
  waitResumePending: boolean
}): boolean {
  const { sessionStatus, waitContextAvailable, waitResumePending } = params
  return sessionStatus === 'waiting' && waitContextAvailable && waitResumePending
}

export function shouldResetWaitResumePending(params: {
  waitResumePending: boolean
  sessionStatus: SessionStatus | null | undefined
  streaming: boolean
}): boolean {
  const { waitResumePending, sessionStatus, streaming } = params
  if (!waitResumePending) return false
  if (sessionStatus !== 'waiting') return true
  return !streaming
}

export function resolveWaitResumeContext(params: {
  canResume: boolean
  runtimeInteraction: RuntimeInteraction | null | undefined
  events: SSEEventData[]
}): WaitEventContext | null {
  if (!params.canResume) return null
  return (
    waitEventContextFromRuntimeInteraction(params.runtimeInteraction)
    ?? findLatestWaitEventContext(params.events)
  )
}
