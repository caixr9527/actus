'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { usePathname, useRouter, useSearchParams } from 'next/navigation'
import { SessionHeader } from '@/components/session-header'
import { ChatInput } from '@/components/chat-input'
import { RunTimelinePanel } from '@/components/run-timeline-panel'
import { ChatMessage } from '@/components/chat-message'
import { FilePreviewPanel } from '@/components/file-preview-panel'
import { ToolPreviewPanel } from '@/components/tool-preview-panel'
import { VNCOverlay } from '@/components/vnc-overlay'
import { WaitResumeCard } from '@/components/wait-resume-card'
import { Button } from '@/components/ui/button'
import { useSessionDetail } from '@/hooks/use-session-detail'
import { getToolKind } from '@/components/tool-use/utils'
import {
  eventsToTimeline,
} from '@/lib/session-events'
import { buildStepViewState, findLatestWaitEventContext } from '@/lib/run-timeline'
import { resolvePreviewToolFromTimeline } from '@/lib/session-preview-tool'
import {
  createSessionScopedDetailViewState,
  createSessionScopedRuntimeState,
  shouldAutoCloseTaskPreview,
  shouldAutoScrollToLatest,
  shouldShowSessionThinking,
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
import { Loader2 } from 'lucide-react'
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
    if (item.kind === 'tool' && getToolKind(item.data) !== 'message') {
      return item.data
    }
    if (item.kind === 'step' && item.tools.length > 0) {
      for (let j = item.tools.length - 1; j >= 0; j--) {
        if (getToolKind(item.tools[j]) !== 'message') {
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
  const waitContext = useMemo(() => findLatestWaitEventContext(events), [events])
  const isWaitingForResume = session?.status === 'waiting'

  const [sessionUiState, setSessionUiState] = useState<SessionScopedDetailViewState<AttachmentFile, ToolEvent>>(
    () => createSessionScopedDetailViewState<AttachmentFile, ToolEvent>(),
  )
  const sessionRuntimeRef = useRef(createSessionScopedRuntimeState())
  const scrollContainerRef = useRef<HTMLDivElement>(null)
  const { fileListOpen, previewFile, previewTool, timelineExpanded, vncOpen } = sessionUiState

  useEffect(() => {
    if (!isHydrated || isLoggedIn) {
      return
    }
    router.replace(`/?auth=login&redirect=${encodeURIComponent(currentPath)}`)
  }, [currentPath, isHydrated, isLoggedIn, router])

  const hasPreview = previewFile !== null || previewTool !== null
  const showFullTimeline = timelineExpanded
  const visibleTimeline = useMemo(() => {
    if (showFullTimeline || timeline.length <= TIMELINE_WINDOW_SIZE) return timeline
    return timeline.slice(-TIMELINE_WINDOW_SIZE)
  }, [showFullTimeline, timeline])
  const hiddenTimelineCount = timeline.length - visibleTimeline.length

  /**
   * 将 previewTool 解析为 timeline 中最新版本的工具对象。
   * 自动跟踪设置 previewTool 时工具事件可能尚无 content（如截图），
   * 后续 SSE 更新后 timeline 中对象已刷新但 state 仍为旧引用。
   * 通过 tool_call_id 匹配获取最新版本。
   */
  const resolvedPreviewTool = useMemo(() => {
    return resolvePreviewToolFromTimeline(previewTool, timeline)
  }, [previewTool, timeline])
  const latestTool = useMemo(() => findLatestTool(timeline), [timeline])
  const toolCount = useMemo(() => {
    return timeline.reduce((count, item) => {
      if (item.kind === 'tool') return count + 1
      if (item.kind === 'step') return count + item.tools.length
      return count
    }, 0)
  }, [timeline])

  // 任务运行中自动追踪最新工具预览（VNC 打开时暂停）
  // 该副作用职责是将流式事件同步到预览 UI。
  useEffect(() => {
    if (session?.status !== 'running' || vncOpen) return

    if (toolCount > sessionRuntimeRef.current.previousToolCount && latestTool) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setSessionUiState((prev) => ({
        ...prev,
        previewTool: latestTool,
        previewFile: null,
      }))
      scrollContainerRef.current?.scrollTo({ top: scrollContainerRef.current.scrollHeight, behavior: 'smooth' })
    }
    sessionRuntimeRef.current.previousToolCount = toolCount
  }, [latestTool, session?.status, toolCount, vncOpen])

  useEffect(() => {
    if (!initialMessage || sessionRuntimeRef.current.initialMessageSent || !session || loading || streaming) {
      return
    }

    sessionRuntimeRef.current.initialMessageSent = true

    sendMessage(initialMessage, initialAttachments || [])
      .then(() => {
        removeInitQueryParamFromUrl()
      })
      .catch((e) => {
        toast.error(getApiErrorMessage(e, 'sessionDetail.sendInitialFailed', t))
      })
  }, [initialMessage, initialAttachments, session, loading, streaming, sendMessage, t])

  const handleSend = useCallback(
    async (message: string, uploadedFiles: FileInfo[]) => {
      try {
        const attachmentIds = uploadedFiles.map((f) => f.id)
        await sendMessage(message, attachmentIds)
      } catch (e) {
        toast.error(getApiErrorMessage(e, 'sessionDetail.sendFailed', t))
        throw e
      }
    },
    [sendMessage, t]
  )

  const handleResume = useCallback(
    async (resumeValue: unknown) => {
      try {
        await resumeWaitingRun(resumeValue)
      } catch (error) {
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
    setSessionUiState((prev) => ({
      ...prev,
      previewFile: file,
      previewTool: null,
    }))
  }, [])

  const handleToolClick = useCallback((tool: ToolEvent) => {
    const kind = getToolKind(tool)
    if (kind === 'message') return
    setSessionUiState((prev) => ({
      ...prev,
      previewTool: tool,
      previewFile: null,
    }))
  }, [])

  const handleClosePreview = useCallback(() => {
    setSessionUiState((prev) => ({
      ...prev,
      previewFile: null,
      previewTool: null,
    }))
  }, [])

  const handleJumpToLatest = useCallback(() => {
    if (latestTool) {
      setSessionUiState((prev) => ({
        ...prev,
        previewTool: latestTool,
        previewFile: null,
      }))
    }
    scrollContainerRef.current?.scrollTo({ top: scrollContainerRef.current.scrollHeight, behavior: 'smooth' })
  }, [latestTool])

  const handleOpenVNC = useCallback(() => {
    setSessionUiState((prev) => ({ ...prev, vncOpen: true }))
  }, [])

  const handleCloseVNC = useCallback(() => {
    setSessionUiState((prev) => ({ ...prev, vncOpen: false }))
    // 关闭 VNC 后跳转到最新工具
    if (latestTool && session?.status === 'running') {
      setSessionUiState((prev) => ({
        ...prev,
        previewTool: latestTool,
        previewFile: null,
        vncOpen: false,
      }))
      setTimeout(() => {
        scrollContainerRef.current?.scrollTo({ top: scrollContainerRef.current.scrollHeight, behavior: 'smooth' })
      }, 100)
    }
  }, [latestTool, session?.status])

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
  const isSessionRunning = session?.status === 'running'

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
    sessionStatus: session?.status,
    hasInitialMessage: Boolean(hasInitialMessage),
    timelineLength: timeline.length,
    hasError: Boolean(error),
    hasRunningStep,
  })
  // 草稿流已经承担“正在生成”的可视反馈，避免与通用 thinking 占位重复展示。
  const hasVisibleTextStreamDraft = Boolean(
    plannerTextStream?.text.trim() || finalTextStream?.text.trim()
  )

  useEffect(() => {
    if (!shouldAutoScrollToLatest({
      hasAutoScrolled: sessionRuntimeRef.current.hasAutoScrolled,
      timelineLength: timeline.length,
      shouldShowThinking,
    })) {
      return
    }

    const container = scrollContainerRef.current
    if (!container) return
    sessionRuntimeRef.current.hasAutoScrolled = true
    requestAnimationFrame(() => {
      container.scrollTo({ top: container.scrollHeight, behavior: 'auto' })
    })
  }, [timeline.length, shouldShowThinking])

  useEffect(() => {
    const previousStatus = sessionRuntimeRef.current.previousSessionStatus
    const nextStatus = session?.status ?? null
    if (shouldAutoCloseTaskPreview(previousStatus, nextStatus)) {
      // 任务从 running 收敛到 completed/cancelled 时，预览面板必须立即清空，避免显示过期执行态内容。
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setSessionUiState((prev) => ({
        ...prev,
        previewFile: null,
        previewTool: null,
      }))
    }
    sessionRuntimeRef.current.previousSessionStatus = nextStatus
  }, [session?.status])

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
          <div className={cn('flex flex-col h-full mx-auto w-full min-w-0 px-4', !hasPreview && 'max-w-[768px]')}>
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

            <div ref={scrollContainerRef} className="flex-1 overflow-y-auto">
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
                {visibleTimeline.map((item) => (
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
                    <div className="flex items-center justify-between h-7">
                      <div className="flex items-center justify-center gap-1 text-gray-500">
                        <Loader2 className="size-4 animate-spin" />
                        <span className="text-xs">{t('sessionDetail.thinking')}</span>
                      </div>
                    </div>
                    <div className="max-w-none p-0 m-0 text-gray-500">
                      <MarkdownContent content={plannerTextStream.text} className="text-gray-500" />
                    </div>
                  </div>
                )}

                {finalTextStream && finalTextStream.text.trim() && (
                  <div className="mt-3 flex flex-col gap-2 w-full">
                    <div className="flex items-center justify-between h-7">
                      <div className="flex items-center justify-center gap-1 text-gray-500">
                        <Loader2 className="size-4 animate-spin" />
                        <span className="text-xs">{t('sessionDetail.thinking')}</span>
                      </div>
                    </div>
                    <div className="max-w-none p-0 m-0 text-gray-700">
                      <MarkdownContent content={finalTextStream.text} className="text-gray-700" />
                    </div>
                  </div>
                )}

                {shouldShowThinking && !hasVisibleTextStreamDraft && (
                  <div className="flex items-center gap-2 text-sm text-gray-500 py-3">
                    <Loader2 className="size-4 animate-spin" />
                    <span>{t('sessionDetail.thinking')}</span>
                  </div>
                )}

                <div className="h-[140px]" />
              </div>
            </div>

            <div className="flex-shrink-0 bg-[#f8f8f7] py-4">
              <RunTimelinePanel
                className="mb-2"
                stepView={stepView}
              />
              {isWaitingForResume && waitContext ? (
                <WaitResumeCard
                  className="mb-2"
                  waitContext={waitContext}
                  busy={streaming}
                  onResume={handleResume}
                  onOpenTakeover={waitContext.suggestUserTakeover ? handleOpenVNC : undefined}
                />
              ) : (
                <>
                  {session.status === 'cancelled' && (
                    <div className="mb-2 flex items-center justify-between gap-3 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">
                      <span>{t('sessionDetail.cancelledContinueHint')}</span>
                      <Button
                        type="button"
                        size="sm"
                        className="cursor-pointer"
                        disabled={streaming}
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
                    isRunning={isSessionRunning}
                    disabled={streaming || isSessionRunning}
                    onStop={handleStop}
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

        {/* 文件预览面板 */}
        {previewFile && (
          <div
            className={cn(
              'animate-in slide-in-from-right duration-300',
              isMobile
                ? 'fixed inset-0 z-40 bg-white'
                : 'flex-shrink-0 h-full w-[420px] lg:w-[520px] xl:w-[600px]'
            )}
          >
            <FilePreviewPanel file={previewFile} onClose={handleClosePreview} />
          </div>
        )}

        {/* 工具预览面板 */}
        {resolvedPreviewTool && (
          <div
            className={cn(
              'animate-in slide-in-from-right duration-300',
              isMobile
                ? 'fixed inset-0 z-40 bg-white'
                : 'flex-shrink-0 h-full w-[420px] lg:w-[520px] xl:w-[600px] py-2 pr-2'
            )}
          >
            <ToolPreviewPanel
              tool={resolvedPreviewTool}
              onClose={handleClosePreview}
              onJumpToLatest={handleJumpToLatest}
              onOpenVNC={getToolKind(resolvedPreviewTool) === 'browser' ? handleOpenVNC : undefined}
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
