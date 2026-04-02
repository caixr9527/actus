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
import { findLatestWaitEventContext } from '@/lib/run-timeline'
import { resolvePreviewToolFromTimeline } from '@/lib/session-preview-tool'
import { shouldAutoCloseTaskPreview, shouldAutoScrollToLatest } from '@/lib/session-detail-view-state'
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
    streaming,
  } = useSessionDetail(sessionId, hasInitialMessage, isHydrated && isLoggedIn)

  const timeline = useMemo(() => eventsToTimeline(events, locale), [events, locale])
  const waitContext = useMemo(() => findLatestWaitEventContext(events), [events])
  const isWaitingForResume = session?.status === 'waiting'

  const [fileListOpen, setFileListOpen] = useState(false)
  const [previewFile, setPreviewFile] = useState<AttachmentFile | null>(null)
  const [previewTool, setPreviewTool] = useState<ToolEvent | null>(null)
  const [expandedTimelineSessionId, setExpandedTimelineSessionId] = useState<string | null>(null)
  const [vncOpen, setVncOpen] = useState(false)
  const initialMessageSentRef = useRef(false)
  const scrollContainerRef = useRef<HTMLDivElement>(null)
  const prevToolCountRef = useRef(0)
  const autoScrolledSessionIdRef = useRef<string | null>(null)
  const previousSessionStatusRef = useRef(session?.status ?? null)

  useEffect(() => {
    if (!isHydrated || isLoggedIn) {
      return
    }
    router.replace(`/?auth=login&redirect=${encodeURIComponent(currentPath)}`)
  }, [currentPath, isHydrated, isLoggedIn, router])

  const hasPreview = previewFile !== null || previewTool !== null
  const showFullTimeline = expandedTimelineSessionId === sessionId
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

  // 任务运行中自动追踪最新工具预览（VNC 打开时暂停）
  // 该副作用职责是将流式事件同步到预览 UI。
  useEffect(() => {
    if (session?.status !== 'running' || vncOpen) return

    const latestTool = findLatestTool(timeline)
    const toolCount = timeline.reduce((n, item) => {
      if (item.kind === 'tool') return n + 1
      if (item.kind === 'step') return n + item.tools.length
      return n
    }, 0)

    if (toolCount > prevToolCountRef.current && latestTool) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setPreviewTool(latestTool)
      setPreviewFile(null)
      scrollContainerRef.current?.scrollTo({ top: scrollContainerRef.current.scrollHeight, behavior: 'smooth' })
    }
    prevToolCountRef.current = toolCount
  }, [timeline, session?.status, vncOpen])

  useEffect(() => {
    if (!initialMessage || initialMessageSentRef.current || !session || loading || streaming) {
      return
    }

    initialMessageSentRef.current = true

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

  const handleModelChange = useCallback(
    async (modelId: string) => {
      await updateSessionModel(modelId)
    },
    [updateSessionModel],
  )

  const handleViewAllFiles = useCallback(() => {
    refreshFiles()
    setFileListOpen(true)
  }, [refreshFiles])

  const handleFileClick = useCallback((file: AttachmentFile) => {
    setPreviewFile(file)
    setPreviewTool(null)
  }, [])

  const handleToolClick = useCallback((tool: ToolEvent) => {
    const kind = getToolKind(tool)
    if (kind === 'message') return
    setPreviewTool(tool)
    setPreviewFile(null)
  }, [])

  const handleClosePreview = useCallback(() => {
    setPreviewFile(null)
    setPreviewTool(null)
  }, [])

  const handleJumpToLatest = useCallback(() => {
    const latest = findLatestTool(timeline)
    if (latest) {
      setPreviewTool(latest)
      setPreviewFile(null)
    }
    scrollContainerRef.current?.scrollTo({ top: scrollContainerRef.current.scrollHeight, behavior: 'smooth' })
  }, [timeline])

  const handleOpenVNC = useCallback(() => {
    setVncOpen(true)
  }, [])

  const handleCloseVNC = useCallback(() => {
    setVncOpen(false)
    // 关闭 VNC 后跳转到最新工具
    const latest = findLatestTool(timeline)
    if (latest && session?.status === 'running') {
      setPreviewTool(latest)
      setPreviewFile(null)
      setTimeout(() => {
        scrollContainerRef.current?.scrollTo({ top: scrollContainerRef.current.scrollHeight, behavior: 'smooth' })
      }, 100)
    }
  }, [timeline, session?.status])

  const handleStop = useCallback(async () => {
    if (!session) return
    try {
      await sessionApi.stopSession(sessionId)
      toast.success(t('sessionDetail.stopSuccess'))
      refresh()
    } catch (error) {
      toast.error(getApiErrorMessage(error, 'sessionDetail.stopFailed', t))
    }
  }, [session, sessionId, refresh, t])

  const handleRealtimeRecover = useCallback(() => {
    void refresh()
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

  const shouldShowThinking =
    streaming || session?.status === 'running' || (hasInitialMessage && timeline.length === 0 && !error)

  useEffect(() => {
    if (!shouldAutoScrollToLatest({
      lastAutoScrolledSessionId: autoScrolledSessionIdRef.current,
      sessionId,
      timelineLength: timeline.length,
      shouldShowThinking,
    })) {
      return
    }

    const container = scrollContainerRef.current
    if (!container) return
    autoScrolledSessionIdRef.current = sessionId
    requestAnimationFrame(() => {
      container.scrollTo({ top: container.scrollHeight, behavior: 'auto' })
    })
  }, [sessionId, timeline.length, shouldShowThinking])

  useEffect(() => {
    const previousStatus = previousSessionStatusRef.current
    const nextStatus = session?.status ?? null
    if (shouldAutoCloseTaskPreview(previousStatus, nextStatus)) {
      setPreviewFile(null)
      setPreviewTool(null)
    }
    previousSessionStatusRef.current = nextStatus
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
                onFileListOpenChange={setFileListOpen}
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
                      onClick={() => setExpandedTimelineSessionId(sessionId)}
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
                      onClick={() => setExpandedTimelineSessionId(null)}
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

                {shouldShowThinking && (
                  <div className="flex items-center gap-2 text-sm text-gray-500 py-3">
                    <Loader2 className="size-4 animate-spin" />
                    <span>{t('sessionDetail.thinking')}</span>
                  </div>
                )}

                <div className="h-[140px]" />
              </div>
            </div>

            <div className="flex-shrink-0 bg-[#f8f8f7] py-4">
              <RunTimelinePanel className="mb-2" events={events} />
              {isWaitingForResume && waitContext ? (
                <WaitResumeCard
                  className="mb-2"
                  waitContext={waitContext}
                  busy={streaming}
                  onResume={handleResume}
                  onOpenTakeover={waitContext.suggestUserTakeover ? handleOpenVNC : undefined}
                />
              ) : (
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
