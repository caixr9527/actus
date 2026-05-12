'use client'

import { useCallback, useEffect, useMemo, useRef, useState, type UIEvent } from 'react'
import { usePathname, useRouter, useSearchParams } from 'next/navigation'
import { SessionHeader } from '@/components/session-header'
import { ChatInput } from '@/components/chat-input'
import { ChatMessage } from '@/components/chat-message'
import { FilePreviewPanel } from '@/components/file-preview-panel'
import { ToolPreviewPanel } from '@/components/tool-preview-panel'
import { VNCOverlay } from '@/components/vnc-overlay'
import { WaitResumeCard } from '@/components/wait-resume-card'
import { Button } from '@/components/ui/button'
import { useSidebar } from '@/components/ui/sidebar'
import { useSessionDetail } from '@/hooks/use-session-detail'
import { getToolKind } from '@/components/tool-use/utils'
import {
  eventsToTimeline,
  isEvidenceReuseVirtualToolEvent,
} from '@/lib/session-events'
import { timelineToConversationItems } from '@/lib/assistant-turns'
import { buildStepViewState } from '@/lib/run-timeline'
import { resolvePreviewToolFromTimeline } from '@/lib/session-preview-tool'
import {
  createSessionScopedDetailViewState,
  createSessionScopedRuntimeState,
  isNearScrollBottom,
  resolveSessionActionAvailability,
  resolveWaitResumeContext,
  shouldAutoCloseTaskPreview,
  shouldAutoScrollToLatest,
  shouldHideWaitResumeCard,
  shouldResetWaitResumePending,
  shouldShowSessionThinking,
  shouldShowJumpToLatestButton,
  type SessionScopedDetailViewState,
} from '@/lib/session-detail-view-state'
import { cn } from '@/lib/utils'
import { useIsMobile } from '@/hooks/use-mobile'
import { useAuth } from '@/hooks/use-auth'
import { getApiErrorMessage, isApiErrorKey } from '@/lib/api'
import type { ToolEvent, FileInfo } from '@/lib/api/types'
import type { AttachmentFile, TimelineItem } from '@/lib/session-events'
import { sessionApi } from '@/lib/api/session'
import { toast } from 'sonner'
import { ArrowDown, Loader2 } from 'lucide-react'
import { useI18n } from '@/lib/i18n'
import { MarkdownContent } from '@/components/markdown-content'

export interface SessionDetailViewProps {
  sessionId: string
  initialMessage?: string
  initialAttachments?: string[]
  hasInitialMessage?: boolean
}

/**
 * 从 timeline 中找到最后一个非 message 类型的工具事件
 */
function findLatestTool(timeline: TimelineItem[]): ToolEvent | null {
  for (let i = timeline.length - 1; i >= 0; i--) {
    const item = timeline[i]
    if (
      item.kind === 'tool' &&
      getToolKind(item.data) !== 'message' &&
      !isEvidenceReuseVirtualToolEvent(item.data)
    ) {
      return item.data
    }
    if (item.kind === 'step' && item.tools.length > 0) {
      for (let j = item.tools.length - 1; j >= 0; j--) {
        if (
          getToolKind(item.tools[j]) !== 'message' &&
          !isEvidenceReuseVirtualToolEvent(item.tools[j])
        ) {
          return item.tools[j]
        }
      }
    }
  }
  return null
}

function removeInitQueryParamFromUrl(): void {
  if (typeof window === 'undefined') return
  const url = new URL(window.location.href)
  if (!url.searchParams.has('init')) return
  url.searchParams.delete('init')
  const nextUrl = `${url.pathname}${url.search}${url.hash}`
  window.history.replaceState(window.history.state, '', nextUrl)
}

const TIMELINE_WINDOW_SIZE = 120
const PREVIEW_PANEL_ANIMATION_MS = 260

type PreviewPanelState =
  | { kind: 'none' }
  | { kind: 'file'; file: AttachmentFile; closing: boolean }
  | { kind: 'tool'; tool: ToolEvent; closing: boolean }

export function SessionDetailView({ sessionId, initialMessage, initialAttachments, hasInitialMessage }: SessionDetailViewProps) {
  return (
    <SessionDetailViewSessionScope
      key={sessionId}
      sessionId={sessionId}
      initialMessage={initialMessage}
      initialAttachments={initialAttachments}
      hasInitialMessage={hasInitialMessage}
    />
  )
}

function SessionDetailViewSessionScope({ sessionId, initialMessage, initialAttachments, hasInitialMessage }: SessionDetailViewProps) {
  const router = useRouter()
  const pathname = usePathname()
  const searchParams = useSearchParams()
  const { locale, t } = useI18n()
  const { isHydrated, isLoggedIn } = useAuth()
  const isMobile = useIsMobile()
  const { setOpen, setOpenMobile } = useSidebar()
  const currentPath = useMemo(() => {
    const query = searchParams.toString()
    return query ? `${pathname}?${query}` : pathname
  }, [pathname, searchParams])
  const {
    session,
    files,
    availableModels,
    defaultModelId,
    events,
    loading,
    modelsLoading,
    modelUpdating,
    error,
    refresh,
    refreshFiles,
    updateSessionModel,
    sendMessage,
    resumeWaitingRun,
    continueCancelledRun,
    streaming,
    plannerTextStream,
    finalTextStream,
  } = useSessionDetail(sessionId, hasInitialMessage, isHydrated && isLoggedIn)

  const timeline = useMemo(() => eventsToTimeline(events, locale), [events, locale])
  const stepView = useMemo(() => buildStepViewState(events), [events])
  const hasRunningStep = useMemo(() => {
    return stepView.steps.some((step) => step.status === 'running')
  }, [stepView.steps])
  const runtimeStatus = session?.runtime.status
  const actionAvailability = useMemo(
    () => resolveSessionActionAvailability(session?.runtime.capabilities),
    [session?.runtime.capabilities],
  )
  const { canSendMessage, canResume, canCancel, canContinueCancelled } = actionAvailability
  const waitContext = useMemo(() => resolveWaitResumeContext({
    canResume,
    runtimeInteraction: session?.runtime.interaction,
    events,
  }), [canResume, events, session?.runtime.interaction])
  const isWaitingForResume = runtimeStatus === 'waiting' && canResume

  const [sessionUiState, setSessionUiState] = useState<SessionScopedDetailViewState<AttachmentFile, ToolEvent>>(
    () => createSessionScopedDetailViewState<AttachmentFile, ToolEvent>(),
  )
  const [waitResumePending, setWaitResumePending] = useState(false)
  const [previewPanel, setPreviewPanel] = useState<PreviewPanelState>({ kind: 'none' })
  const sessionRuntimeRef = useRef(createSessionScopedRuntimeState())
  const [autoFollowLatest, setAutoFollowLatest] = useState(true)
  const scrollContainerRef = useRef<HTMLDivElement>(null)
  const followScrollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const previewCloseTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const { fileListOpen, previewFile, previewTool, timelineExpanded, vncOpen } = sessionUiState

  const scrollToLatest = useCallback((behavior: ScrollBehavior = 'smooth') => {
    const container = scrollContainerRef.current
    if (!container) return
    sessionRuntimeRef.current.autoFollowLatest = true
    setAutoFollowLatest(true)
    requestAnimationFrame(() => {
      container.scrollTo({ top: container.scrollHeight, behavior })
    })
  }, [])

  const handleTimelineScroll = useCallback((event: UIEvent<HTMLDivElement>) => {
    const container = event.currentTarget
    if (followScrollTimerRef.current) {
      clearTimeout(followScrollTimerRef.current)
    }
    followScrollTimerRef.current = setTimeout(() => {
      const nextAutoFollowLatest = isNearScrollBottom({
        scrollTop: container.scrollTop,
        scrollHeight: container.scrollHeight,
        clientHeight: container.clientHeight,
      })
      sessionRuntimeRef.current.autoFollowLatest = nextAutoFollowLatest
      setAutoFollowLatest(nextAutoFollowLatest)
      followScrollTimerRef.current = null
    }, 80)
  }, [])

  useEffect(() => {
    if (!isHydrated || isLoggedIn) {
      return
    }
    router.replace(`/?auth=login&redirect=${encodeURIComponent(currentPath)}`)
  }, [currentPath, isHydrated, isLoggedIn, router])

  const hasPreview = previewFile !== null || previewTool !== null
  const mainContentWidthClass = hasPreview ? 'max-w-[1200px]' : 'max-w-[1080px]'
  const effectiveWaitResumePending = useMemo(() => {
    if (!waitResumePending) return false
    return !shouldResetWaitResumePending({
      waitResumePending,
      sessionStatus: runtimeStatus,
      streaming,
    })
  }, [waitResumePending, runtimeStatus, streaming])
  const showFullTimeline = timelineExpanded
  const visibleTimeline = useMemo(() => {
    if (showFullTimeline || timeline.length <= TIMELINE_WINDOW_SIZE) return timeline
    return timeline.slice(-TIMELINE_WINDOW_SIZE)
  }, [showFullTimeline, timeline])
  const conversationItems = useMemo(() => timelineToConversationItems(visibleTimeline), [visibleTimeline])
  const hiddenTimelineCount = timeline.length - visibleTimeline.length
  const collapseLeftSidebar = useCallback(() => {
    setOpen(false)
    setOpenMobile(false)
  }, [setOpen, setOpenMobile])

  const panelTool = previewPanel.kind === 'tool'
    ? resolvePreviewToolFromTimeline(previewPanel.tool, timeline) ?? previewPanel.tool
    : null
  const latestTool = useMemo(() => findLatestTool(timeline), [timeline])
  const toolCount = useMemo(() => {
    return timeline.reduce((count, item) => {
      if (item.kind === 'tool') {
        return isEvidenceReuseVirtualToolEvent(item.data) ? count : count + 1
      }
      if (item.kind === 'step') {
        return count + item.tools.filter((tool) => !isEvidenceReuseVirtualToolEvent(tool)).length
      }
      return count
    }, 0)
  }, [timeline])

  // 任务运行中自动追踪最新工具预览（VNC 打开时暂停）
  // 该副作用职责是将流式事件同步到预览 UI。
  useEffect(() => {
    if (runtimeStatus !== 'running' || vncOpen) return

    if (toolCount > sessionRuntimeRef.current.previousToolCount && latestTool) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setSessionUiState((prev) => ({
        ...prev,
        previewTool: latestTool,
        previewFile: null,
      }))
      if (previewCloseTimerRef.current) {
        clearTimeout(previewCloseTimerRef.current)
        previewCloseTimerRef.current = null
      }
      setPreviewPanel({ kind: 'tool', tool: latestTool, closing: false })
      scrollToLatest('smooth')
    }
    sessionRuntimeRef.current.previousToolCount = toolCount
  }, [latestTool, runtimeStatus, scrollToLatest, toolCount, vncOpen])

  useEffect(() => {
    if (!initialMessage || sessionRuntimeRef.current.initialMessageSent || !session || loading || streaming) {
      return
    }

    sessionRuntimeRef.current.initialMessageSent = true

    collapseLeftSidebar()
    sendMessage(initialMessage, initialAttachments || [])
      .then(() => {
        removeInitQueryParamFromUrl()
      })
      .catch((e) => {
        toast.error(getApiErrorMessage(e, 'sessionDetail.sendInitialFailed', t))
      })
  }, [collapseLeftSidebar, initialMessage, initialAttachments, session, loading, streaming, sendMessage, t])

  const handleSend = useCallback(
    async (message: string, uploadedFiles: FileInfo[]) => {
      try {
        const attachmentIds = uploadedFiles.map((f) => f.id)
        collapseLeftSidebar()
        await sendMessage(message, attachmentIds)
      } catch (e) {
        toast.error(getApiErrorMessage(e, 'sessionDetail.sendFailed', t))
        throw e
      }
    },
    [collapseLeftSidebar, sendMessage, t]
  )

  const handleResume = useCallback(
    async (resumeValue: unknown) => {
      setWaitResumePending(true)
      try {
        await resumeWaitingRun(resumeValue)
      } catch (error) {
        setWaitResumePending(false)
        toast.error(getApiErrorMessage(error, 'sessionDetail.sendFailed', t))
        throw error
      }
    },
    [resumeWaitingRun, t]
  )

  const handleContinueCancelledRun = useCallback(
    async () => {
      try {
        await continueCancelledRun()
      } catch (error) {
        toast.error(getApiErrorMessage(error, 'sessionDetail.sendFailed', t))
        throw error
      }
    },
    [continueCancelledRun, t],
  )

  const handleModelChange = useCallback(
    async (modelId: string) => {
      await updateSessionModel(modelId)
    },
    [updateSessionModel],
  )

  const handleViewAllFiles = useCallback(() => {
    refreshFiles()
    setSessionUiState((prev) => ({ ...prev, fileListOpen: true }))
  }, [refreshFiles])

  const handleFileClick = useCallback((file: AttachmentFile) => {
    if (previewCloseTimerRef.current) {
      clearTimeout(previewCloseTimerRef.current)
      previewCloseTimerRef.current = null
    }
    setSessionUiState((prev) => ({
      ...prev,
      previewFile: file,
      previewTool: null,
    }))
    setPreviewPanel({ kind: 'file', file, closing: false })
  }, [])

  const handleToolClick = useCallback((tool: ToolEvent) => {
    const kind = getToolKind(tool)
    if (kind === 'message' || isEvidenceReuseVirtualToolEvent(tool)) return
    if (previewCloseTimerRef.current) {
      clearTimeout(previewCloseTimerRef.current)
      previewCloseTimerRef.current = null
    }
    setSessionUiState((prev) => ({
      ...prev,
      previewTool: tool,
      previewFile: null,
    }))
    setPreviewPanel({ kind: 'tool', tool, closing: false })
  }, [])

  const handleClosePreview = useCallback(() => {
    setSessionUiState((prev) => ({
      ...prev,
      previewFile: null,
      previewTool: null,
    }))
    if (previewCloseTimerRef.current) {
      clearTimeout(previewCloseTimerRef.current)
      previewCloseTimerRef.current = null
    }
    setPreviewPanel((current) => {
      if (current.kind === 'none' || current.closing) return current
      previewCloseTimerRef.current = setTimeout(() => {
        setPreviewPanel({ kind: 'none' })
        previewCloseTimerRef.current = null
      }, PREVIEW_PANEL_ANIMATION_MS)
      return { ...current, closing: true }
    })
  }, [])

  const closePreviewPanel = handleClosePreview

  useEffect(() => {
    return () => {
      if (followScrollTimerRef.current) {
        clearTimeout(followScrollTimerRef.current)
      }
      if (previewCloseTimerRef.current) {
        clearTimeout(previewCloseTimerRef.current)
      }
    }
  }, [])

  const handleJumpToLatest = useCallback(() => {
    if (latestTool) {
      if (previewCloseTimerRef.current) {
        clearTimeout(previewCloseTimerRef.current)
        previewCloseTimerRef.current = null
      }
      setSessionUiState((prev) => ({
        ...prev,
        previewTool: latestTool,
        previewFile: null,
      }))
      setPreviewPanel({ kind: 'tool', tool: latestTool, closing: false })
    }
    scrollToLatest('smooth')
  }, [latestTool, scrollToLatest])

  const handleOpenVNC = useCallback(() => {
    setSessionUiState((prev) => ({ ...prev, vncOpen: true }))
  }, [])

  const handleCloseVNC = useCallback(() => {
    setSessionUiState((prev) => ({ ...prev, vncOpen: false }))
    // 关闭 VNC 后跳转到最新工具
    if (latestTool && runtimeStatus === 'running') {
      if (previewCloseTimerRef.current) {
        clearTimeout(previewCloseTimerRef.current)
        previewCloseTimerRef.current = null
      }
      setSessionUiState((prev) => ({
        ...prev,
        previewTool: latestTool,
        previewFile: null,
        vncOpen: false,
      }))
      setPreviewPanel({ kind: 'tool', tool: latestTool, closing: false })
      setTimeout(() => {
        scrollToLatest('smooth')
      }, 100)
    }
  }, [latestTool, runtimeStatus, scrollToLatest])

  const handleStop = useCallback(async () => {
    if (!session) return
    try {
      await sessionApi.stopSession(sessionId)
      await refresh({ resetRealtime: true })
      toast.success(t('sessionDetail.stopSuccess'))
    } catch (error) {
      toast.error(getApiErrorMessage(error, 'sessionDetail.stopFailed', t))
    }
  }, [session, sessionId, refresh, t])

  const handleRealtimeRecover = useCallback(() => {
    void refresh({ resetRealtime: true })
  }, [refresh])
  const isSessionNotFoundError = Boolean(
    error &&
    (
      isApiErrorKey(error, 'error.session.not_found') ||
      (error as { code?: number }).code === 404
    )
  )
  const errorMessage = error ? getApiErrorMessage(error, 'sessionDetail.loadFailed', t) : null

  const shouldShowThinking = shouldShowSessionThinking({
    streaming,
    sessionStatus: runtimeStatus,
    hasInitialMessage: Boolean(hasInitialMessage),
    timelineLength: timeline.length,
    hasError: Boolean(error),
    hasRunningStep,
  })
  // 草稿流已经承担“正在生成”的可视反馈，避免与通用 thinking 占位重复展示。
  const hasVisibleTextStreamDraft = Boolean(
    plannerTextStream?.text.trim() || finalTextStream?.text.trim()
  )
  const shouldShowJumpToLatest = shouldShowJumpToLatestButton({
    autoFollowLatest,
    timelineLength: timeline.length,
    shouldShowThinking,
    hasVisibleTextStreamDraft,
  })

  useEffect(() => {
    if (!shouldAutoScrollToLatest({
      autoFollowLatest,
      sessionStatus: runtimeStatus,
      streaming,
      timelineLength: timeline.length,
      shouldShowThinking,
      hasVisibleTextStreamDraft,
    })) {
      return
    }

    scrollToLatest('smooth')
  }, [
    conversationItems,
    finalTextStream?.text,
    hasVisibleTextStreamDraft,
    plannerTextStream?.text,
    autoFollowLatest,
    runtimeStatus,
    scrollToLatest,
    shouldShowThinking,
    streaming,
    timeline.length,
  ])

  useEffect(() => {
    const previousStatus = sessionRuntimeRef.current.previousSessionStatus
    const nextStatus = runtimeStatus ?? null
    if (shouldAutoCloseTaskPreview(previousStatus, nextStatus)) {
      setTimeout(() => {
        closePreviewPanel()
      }, 0)
    }
    sessionRuntimeRef.current.previousSessionStatus = nextStatus
  }, [closePreviewPanel, runtimeStatus])

  if (!isHydrated) {
    return (
      <div className="relative flex flex-col h-full flex-1 min-w-0 px-4 items-center justify-center">
        <p className="text-sm text-gray-500">{t('common.loading')}</p>
      </div>
    )
  }

  if (!isLoggedIn) {
    return (
      <div className="relative flex flex-col h-full flex-1 min-w-0 px-4 items-center justify-center">
        <div className="w-full max-w-[420px] rounded-2xl border bg-white px-6 py-7 text-center shadow-sm">
          <p className="text-base font-medium text-gray-700">{t('sessionDetail.loginRequired')}</p>
          <div className="mt-3 inline-flex items-center gap-2 text-sm text-gray-500">
            <Loader2 className="size-4 animate-spin" />
            <span>{t('sessionDetail.openingLogin')}</span>
          </div>
        </div>
      </div>
    )
  }

  if (loading && !session) {
    return (
      <div className="relative flex flex-col h-full flex-1 min-w-0 px-4 items-center justify-center">
        {hasInitialMessage ? (
          <div className="flex items-center gap-2 text-sm text-gray-500">
            <Loader2 className="size-4 animate-spin" />
            <span>{t('sessionDetail.thinking')}</span>
          </div>
        ) : (
          <p className="text-sm text-gray-500">{t('common.loading')}</p>
        )}
      </div>
    )
  }

  if (error && !session) {
    if (isSessionNotFoundError) {
      return (
        <div className="relative flex flex-col h-full flex-1 min-w-0 px-4 items-center justify-center gap-3">
          <p className="text-base text-gray-700">{t('sessionDetail.sessionNotFoundTitle')}</p>
          <p className="text-sm text-gray-500 text-center">
            {t('sessionDetail.sessionNotFoundDescription')}
          </p>
          <div className="flex items-center gap-2">
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="cursor-pointer"
              onClick={() => refresh()}
            >
              {t('common.retry')}
            </Button>
            <Button
              type="button"
              size="sm"
              className="cursor-pointer"
              onClick={() => router.push('/')}
            >
              {t('sessionDetail.backHome')}
            </Button>
          </div>
        </div>
      )
    }

    return (
      <div className="relative flex flex-col h-full flex-1 min-w-0 px-4 items-center justify-center gap-2">
        <p className="text-sm text-red-600">
          {t('sessionDetail.loadSessionFailed', {
            message: errorMessage ?? t('sessionDetail.loadFailed'),
          })}
        </p>
        <button
          type="button"
          onClick={() => refresh()}
          className="text-sm text-primary underline"
        >
          {t('common.retry')}
        </button>
      </div>
    )
  }

  if (!session) {
    return (
      <div className="relative flex flex-col h-full flex-1 min-w-0 px-4 items-center justify-center gap-3">
        <p className="text-sm text-gray-500">{t('sessionDetail.notFound')}</p>
        <button
          type="button"
          className="text-sm text-primary underline underline-offset-2 cursor-pointer"
          onClick={() => router.push('/')}
        >
          {t('sessionDetail.backHome')}
        </button>
      </div>
    )
  }

  return (
    <>
      <div className="flex flex-row h-screen w-full overflow-hidden">
        {/* 主内容区 */}
        <div className="flex flex-col flex-1 min-w-0 h-full overflow-hidden">
          <div className={cn('flex flex-col h-full mx-auto w-full min-w-0 px-4', mainContentWidthClass)}>
            <div className="flex-shrink-0">
              <SessionHeader
                title={session.title}
                files={files}
                fileListOpen={fileListOpen}
                onFileListOpenChange={(open) => {
                  setSessionUiState((prev) => ({ ...prev, fileListOpen: open }))
                }}
                onFetchFiles={refreshFiles}
                onFileClick={handleFileClick}
              />
            </div>

            {error && (
              <div className="mt-2 rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-800">
                <div className="flex items-center justify-between gap-2">
                  <span>{errorMessage}</span>
                  <button
                    type="button"
                    className="text-amber-900 underline underline-offset-2 cursor-pointer"
                    onClick={handleRealtimeRecover}
                  >
                    {t('common.retry')}
                  </button>
                </div>
              </div>
            )}

            <div className="relative flex-1 min-h-0">
              <div
                ref={scrollContainerRef}
                className="scrollbar-hide h-full overflow-y-auto"
                onScroll={handleTimelineScroll}
              >
                <div className="flex flex-col w-full gap-3 pt-3">
                {hiddenTimelineCount > 0 && (
                  <div className="flex justify-center py-1">
                    <button
                      type="button"
                      className="text-xs text-gray-500 hover:text-gray-700 underline underline-offset-2 cursor-pointer"
                      onClick={() => {
                        setSessionUiState((prev) => ({ ...prev, timelineExpanded: true }))
                      }}
                    >
                      {t('sessionDetail.showEarlierRecords', { count: hiddenTimelineCount })}
                    </button>
                  </div>
                )}
                {showFullTimeline && timeline.length > TIMELINE_WINDOW_SIZE && (
                  <div className="flex justify-center py-1">
                    <button
                      type="button"
                      className="text-xs text-gray-500 hover:text-gray-700 underline underline-offset-2 cursor-pointer"
                      onClick={() => {
                        setSessionUiState((prev) => ({ ...prev, timelineExpanded: false }))
                      }}
                    >
                      {t('sessionDetail.showRecentRecords', { count: TIMELINE_WINDOW_SIZE })}
                    </button>
                  </div>
                )}
                {timeline.length === 0 && !streaming && !hasInitialMessage && (
                  <div className="flex items-center justify-center py-8 text-sm text-gray-500">
                    {t('sessionDetail.emptyTimeline')}
                  </div>
                )}
                {conversationItems.map((item) => (
                  <ChatMessage
                    key={item.id}
                    item={item}
                    onViewAllFiles={handleViewAllFiles}
                    onFileClick={handleFileClick}
                    onToolClick={handleToolClick}
                  />
                ))}

                {plannerTextStream && plannerTextStream.text.trim() && (
                  <div className="mt-3 flex flex-col gap-2 w-full">
                    <div className="max-w-none p-0 m-0 text-gray-500">
                      <MarkdownContent content={plannerTextStream.text} className="text-gray-500" />
                    </div>
                  </div>
                )}

                {finalTextStream && finalTextStream.text.trim() && (
                  <div className="mt-3 flex flex-col gap-2 w-full">
                    <div className="max-w-none p-0 m-0 text-gray-700">
                      <MarkdownContent content={finalTextStream.text} className="text-gray-700" />
                    </div>
                  </div>
                )}

                {(shouldShowThinking || hasVisibleTextStreamDraft) && (
                  <div className="flex items-center gap-2 text-sm text-gray-500 py-3">
                    <Loader2 className="size-4 animate-spin" />
                    <span>{t('sessionDetail.thinking')}</span>
                  </div>
                )}

                <div className="h-[140px]" />
                </div>
              </div>
              {shouldShowJumpToLatest && (
                <div className="pointer-events-none absolute inset-x-0 bottom-2 z-20 flex justify-center">
                  <button
                    type="button"
                    className="pointer-events-auto inline-flex h-10 w-10 items-center justify-center rounded-full border border-stone-200 bg-white/95 text-stone-700 shadow-[0_8px_24px_rgba(0,0,0,0.12)]"
                    onClick={() => {
                      scrollToLatest('smooth')
                    }}
                    aria-label={t('sessionDetail.jumpToLatest')}
                  >
                    <ArrowDown className="size-4" />
                  </button>
                </div>
              )}
            </div>

            <div className="flex-shrink-0 bg-[#f8f8f7] py-4">
              {isWaitingForResume && waitContext && !shouldHideWaitResumeCard({
                sessionStatus: runtimeStatus,
                waitContextAvailable: Boolean(waitContext),
                waitResumePending: effectiveWaitResumePending,
              }) ? (
                <WaitResumeCard
                  className="mb-2"
                  waitContext={waitContext}
                  busy={streaming || effectiveWaitResumePending || !canResume}
                  onResume={handleResume}
                  onOpenTakeover={waitContext.suggestUserTakeover ? handleOpenVNC : undefined}
                />
              ) : (
                <>
                  {runtimeStatus === 'cancelled' && canContinueCancelled && (
                    <div className="mb-2 flex items-center justify-between gap-3 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">
                      <span>{t('sessionDetail.cancelledContinueHint')}</span>
                      <Button
                        type="button"
                        size="sm"
                        className="cursor-pointer"
                        disabled={streaming || !canContinueCancelled}
                        onClick={() => {
                          void handleContinueCancelledRun()
                        }}
                      >
                        {t('sessionDetail.continueCancelledTask')}
                      </Button>
                    </div>
                  )}
                  <ChatInput
                    onSend={handleSend}
                    sessionId={sessionId}
                    isRunning={canCancel}
                    disabled={!canSendMessage || streaming || runtimeStatus === 'running'}
                    onStop={canCancel ? handleStop : undefined}
                    modelOptions={availableModels}
                    currentModelId={session.current_model_id}
                    defaultModelId={defaultModelId}
                    modelsLoading={modelsLoading}
                    modelUpdating={modelUpdating}
                    onModelChange={handleModelChange}
                  />
                </>
              )}
            </div>
          </div>
        </div>

        {/* 文件/工具预览面板 */}
        {previewPanel.kind === 'file' && (
          <div
            className={cn(
              previewPanel.closing
                ? 'animate-out slide-out-to-right'
                : 'animate-in slide-in-from-right',
              'duration-300',
              isMobile
                ? 'fixed inset-0 z-40 bg-white'
                : 'flex-shrink-0 h-full w-[420px] lg:w-[520px] xl:w-[600px]'
            )}
          >
            <FilePreviewPanel file={previewPanel.file} onClose={handleClosePreview} />
          </div>
        )}

        {previewPanel.kind === 'tool' && panelTool && (
          <div
            className={cn(
              previewPanel.closing
                ? 'animate-out slide-out-to-right'
                : 'animate-in slide-in-from-right',
              'duration-300',
              isMobile
                ? 'fixed inset-0 z-40 bg-white'
                : 'flex-shrink-0 h-full w-[420px] lg:w-[520px] xl:w-[600px] py-2 pr-2'
            )}
          >
            <ToolPreviewPanel
              tool={panelTool}
              onClose={handleClosePreview}
              onJumpToLatest={handleJumpToLatest}
              onOpenVNC={getToolKind(panelTool) === 'browser' ? handleOpenVNC : undefined}
            />
          </div>
        )}
      </div>

      {/* noVNC 全屏远程桌面覆盖层 */}
      {vncOpen && (
        <VNCOverlay sessionId={sessionId} onClose={handleCloseVNC} />
      )}
    </>
  )
}
