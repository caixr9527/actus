import type { SessionStatus, StepEvent } from './api/types'

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

export function closeTaskPreviewStateOnSend<TFile = unknown, TTool = unknown>(
  prev: SessionScopedDetailViewState<TFile, TTool>,
): SessionScopedDetailViewState<TFile, TTool> {
  return {
    ...prev,
    fileListOpen: false,
    previewFile: null,
    previewTool: null,
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
