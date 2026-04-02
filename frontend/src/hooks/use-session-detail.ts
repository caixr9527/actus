'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { configApi } from '@/lib/api/config'
import { sessionApi } from '@/lib/api/session'
import { normalizeEvents, unwrapNestedEvent, visitSessionEvent } from '@/lib/session-event-adapter'
import { canRetry, computeRetryDelayMs, shouldStartEmptySessionStream, type RetryPolicy } from '@/lib/session-stream-policy'
import {
  classifyMessageStreamCloseReason,
  reduceSessionRuntimeStateOnEvent,
  shouldReloadSnapshotAfterMessageStreamClose,
} from '@/lib/session-detail-runtime'
import type { ListModelItem, SessionDetail, SessionStatus, SSEEventData, SessionFile } from '@/lib/api/types'
import { useI18n } from '@/lib/i18n'

export type UseSessionDetailResult = {
  session: SessionDetail | null
  files: SessionFile[]
  availableModels: ListModelItem[]
  defaultModelId: string | null
  events: SSEEventData[]
  loading: boolean
  modelsLoading: boolean
  modelUpdating: boolean
  error: Error | null
  refresh: () => Promise<void>
  refreshFiles: () => Promise<void>
  updateSessionModel: (modelId: string) => Promise<void>
  sendMessage: (message: string, attachmentIds: string[]) => Promise<void>
  resumeWaitingRun: (resumeValue: unknown) => Promise<void>
  streaming: boolean
}

const EMPTY_STREAM_RECONNECT_DELAY = 500
const EMPTY_STREAM_RETRY_POLICY: RetryPolicy = {
  maxRetries: 8,
  baseDelayMs: 1000,
  maxDelayMs: 10_000,
}

type SessionSnapshotPayload = {
  detail: SessionDetail
  fileListRaw: unknown
}

const pendingPromiseCache = new Map<string, Promise<unknown>>()

function reusePendingPromise<T>(key: string, factory: () => Promise<T>): Promise<T> {
  const existing = pendingPromiseCache.get(key)
  if (existing) return existing as Promise<T>

  const pending = factory().finally(() => {
    if (pendingPromiseCache.get(key) === pending) {
      pendingPromiseCache.delete(key)
    }
  })
  pendingPromiseCache.set(key, pending)
  return pending
}

function loadSessionSnapshotOnce(sessionId: string): Promise<SessionSnapshotPayload> {
  return reusePendingPromise(`session-snapshot:${sessionId}`, async () => {
    const [detail, fileListRaw] = await Promise.all([
      sessionApi.getSessionDetail(sessionId),
      sessionApi.getSessionFiles(sessionId),
    ])
    return {
      detail,
      fileListRaw,
    }
  })
}

function loadSessionFilesOnce(sessionId: string): Promise<unknown> {
  return reusePendingPromise(`session-files:${sessionId}`, () => sessionApi.getSessionFiles(sessionId))
}

function loadModelsOnce(): Promise<{ models: ListModelItem[]; default_model_id: string | null }> {
  return reusePendingPromise('models', async () => {
    const data = await configApi.getModels()
    return {
      models: data.models,
      default_model_id: data.default_model_id,
    }
  })
}

/**
 * 任务详情：拉取会话详情与文件列表，管理事件列表；
 * 未完成任务会通过 chat 空 body 流式拉取事件，发送消息时通过 chat 带 body 流式追加事件。
 */
export function useSessionDetail(
  sessionId: string | null,
  initialSkipEmptyStream?: boolean,
  enabled: boolean = true,
): UseSessionDetailResult {
  const { t } = useI18n()
  const [session, setSession] = useState<SessionDetail | null>(null)
  const [files, setFiles] = useState<SessionFile[]>([])
  const [availableModels, setAvailableModels] = useState<ListModelItem[]>([])
  const [defaultModelId, setDefaultModelId] = useState<string | null>(null)
  const [events, setEvents] = useState<SSEEventData[]>([])
  const [loading, setLoading] = useState(true)
  const [modelsLoading, setModelsLoading] = useState(true)
  const [modelUpdating, setModelUpdating] = useState(false)
  const [error, setError] = useState<Error | null>(null)
  const [streaming, setStreaming] = useState(false)
  const [skipEmptyStream, setSkipEmptyStream] = useState(initialSkipEmptyStream || false)
  const emptyStreamCleanupRef = useRef<(() => void) | null>(null)
  const messageStreamCleanupRef = useRef<(() => void) | null>(null)
  const emptyStreamReconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const emptyStreamRetryCountRef = useRef(0)
  const emptyStreamInstanceIdRef = useRef(0)
  const isSendMessageRef = useRef(false)
  const lastEventIdRef = useRef<string | null>(null)
  const sessionEpochRef = useRef(0)
  const currentSessionIdRef = useRef<string | null>(sessionId)
  const sessionStatusRef = useRef<SessionStatus | null>(null)
  const streamingRef = useRef(false)
  currentSessionIdRef.current = sessionId

  const setStreamingState = useCallback((nextStreaming: boolean) => {
    streamingRef.current = nextStreaming
    setStreaming(nextStreaming)
  }, [])

  const clearEmptyStreamReconnectTimer = useCallback(() => {
    if (emptyStreamReconnectTimerRef.current) {
      clearTimeout(emptyStreamReconnectTimerRef.current)
      emptyStreamReconnectTimerRef.current = null
    }
  }, [])

  const stopEmptyStream = useCallback(() => {
    emptyStreamInstanceIdRef.current += 1
    clearEmptyStreamReconnectTimer()
    if (emptyStreamCleanupRef.current) {
      emptyStreamCleanupRef.current()
      emptyStreamCleanupRef.current = null
    }
  }, [clearEmptyStreamReconnectTimer])

  const stopMessageStream = useCallback(() => {
    if (messageStreamCleanupRef.current) {
      messageStreamCleanupRef.current()
      messageStreamCleanupRef.current = null
    }
  }, [])

  const appendEvent = useCallback((ev: SSEEventData) => {
    const evToAppend = unwrapNestedEvent(ev)

    const eventId = (evToAppend.data as { event_id?: string })?.event_id
    if (eventId) lastEventIdRef.current = eventId

    setEvents((prev) => [...prev, evToAppend])

    // 更新会话标题（通过统一事件分发表驱动）
    visitSessionEvent(evToAppend, {
      title: (event) => {
        const title = (event.data as { title?: string }).title
        if (typeof title !== 'string') return
        setSession((prev) => (prev ? { ...prev, title } : null))
      },
    })

    // 统一运行时状态迁移（status + streaming）
    const nextRuntime = reduceSessionRuntimeStateOnEvent(
      { status: sessionStatusRef.current, streaming: streamingRef.current },
      evToAppend,
    )
    if (nextRuntime.streaming !== streamingRef.current) {
      setStreamingState(nextRuntime.streaming)
    }
    if (nextRuntime.status !== sessionStatusRef.current) {
      sessionStatusRef.current = nextRuntime.status
      setSession((prev) => {
        if (!prev || !nextRuntime.status || prev.status === nextRuntime.status) return prev
        return { ...prev, status: nextRuntime.status }
      })
    }
  }, [setStreamingState])

  const startEmptyStream = useCallback((expectedEpoch?: number) => {
    const streamSessionId = sessionId
    if (!streamSessionId) return

    const streamEpoch = expectedEpoch ?? sessionEpochRef.current
    if (streamEpoch !== sessionEpochRef.current || streamSessionId !== currentSessionIdRef.current) {
      return
    }

    stopEmptyStream()
    const streamInstanceId = emptyStreamInstanceIdRef.current + 1
    emptyStreamInstanceIdRef.current = streamInstanceId

    emptyStreamCleanupRef.current = sessionApi.chat(
      streamSessionId,
      { event_id: lastEventIdRef.current || undefined },
      (ev) => {
        if (
          streamEpoch !== sessionEpochRef.current ||
          streamSessionId !== currentSessionIdRef.current ||
          streamInstanceId !== emptyStreamInstanceIdRef.current
        ) {
          return
        }
        emptyStreamRetryCountRef.current = 0
        setError(null)
        appendEvent(ev)
      },
      (err) => {
        if (
          streamEpoch !== sessionEpochRef.current ||
          streamSessionId !== currentSessionIdRef.current ||
          streamInstanceId !== emptyStreamInstanceIdRef.current
        ) {
          return
        }
        if (err.name === 'AbortError') {
          return
        }

        const cleanup = emptyStreamCleanupRef.current
        emptyStreamCleanupRef.current = null
        cleanup?.()

        const scheduleReconnect = () => {
          if (streamInstanceId !== emptyStreamInstanceIdRef.current) return
          if (!canRetry(emptyStreamRetryCountRef.current, EMPTY_STREAM_RETRY_POLICY)) {
            setError(new Error(t('sessionDetail.realtimeDisconnected')))
            return
          }

          const retryDelayMs = err.message === 'SSE_STREAM_END'
            ? EMPTY_STREAM_RECONNECT_DELAY
            : computeRetryDelayMs(emptyStreamRetryCountRef.current, EMPTY_STREAM_RETRY_POLICY)

          emptyStreamRetryCountRef.current += 1
          clearEmptyStreamReconnectTimer()
          emptyStreamReconnectTimerRef.current = setTimeout(() => {
            if (
              streamEpoch !== sessionEpochRef.current ||
              streamSessionId !== currentSessionIdRef.current ||
              streamInstanceId !== emptyStreamInstanceIdRef.current
            ) {
              return
            }
            if (!emptyStreamCleanupRef.current && !isSendMessageRef.current) {
              startEmptyStream(streamEpoch)
            }
          }, retryDelayMs)
        }

        // 流正常结束（服务端关闭连接），延迟重连
        if (err.message === 'SSE_STREAM_END') {
          scheduleReconnect()
          return
        }

        console.warn('Session detail empty stream error:', err)
        scheduleReconnect()
      }
    )
  }, [appendEvent, clearEmptyStreamReconnectTimer, sessionId, stopEmptyStream, t])

  const normalizeFileList = useCallback((raw: unknown): SessionFile[] => {
    if (Array.isArray(raw)) return raw as SessionFile[]
    if (raw && typeof raw === 'object' && 'files' in raw && Array.isArray((raw as { files: unknown }).files)) {
      return (raw as { files: SessionFile[] }).files
    }
    if (raw && typeof raw === 'object' && 'data' in raw && Array.isArray((raw as { data: unknown }).data)) {
      return (raw as { data: SessionFile[] }).data
    }
    return []
  }, [])

  const loadSessionSnapshot = useCallback(async (targetSessionId: string, targetEpoch: number): Promise<SessionDetail | null> => {
    try {
      const { detail, fileListRaw } = await loadSessionSnapshotOnce(targetSessionId)

      if (targetEpoch !== sessionEpochRef.current || targetSessionId !== currentSessionIdRef.current) {
        return null
      }

      setError(null)
      setSession(detail)
      sessionStatusRef.current = detail.status
      setFiles(normalizeFileList(fileListRaw))
      const rawEvents = (detail as { events?: unknown }).events
      if (rawEvents && Array.isArray(rawEvents) && rawEvents.length > 0) {
        const normalized = normalizeEvents(rawEvents)
        setEvents(normalized)
        const lastEvId = (normalized[normalized.length - 1]?.data as { event_id?: string })?.event_id
        if (lastEvId) lastEventIdRef.current = lastEvId
      } else {
        setEvents([])
        lastEventIdRef.current = null
      }
      return detail
    } catch (e) {
      if (targetEpoch !== sessionEpochRef.current || targetSessionId !== currentSessionIdRef.current) {
        return null
      }
      setError(e instanceof Error ? e : new Error(t('sessionDetail.loadFailed')))
      return null
    } finally {
      if (targetEpoch === sessionEpochRef.current && targetSessionId === currentSessionIdRef.current) {
        setLoading(false)
      }
    }
  }, [normalizeFileList, t])

  const refresh = useCallback(async () => {
    if (!sessionId || !enabled) return
    const targetEpoch = sessionEpochRef.current
    setLoading(true)
    setError(null)
    const detail = await loadSessionSnapshot(sessionId, targetEpoch)
    if (
      detail &&
      targetEpoch === sessionEpochRef.current &&
      sessionId === currentSessionIdRef.current &&
      shouldStartEmptySessionStream(detail.status, isSendMessageRef.current, skipEmptyStream)
    ) {
      startEmptyStream(targetEpoch)
    }
  }, [enabled, loadSessionSnapshot, sessionId, skipEmptyStream, startEmptyStream])

  const refreshFiles = useCallback(async () => {
    if (!sessionId || !enabled) return
    const targetEpoch = sessionEpochRef.current
    const targetSessionId = sessionId
    try {
      const fileListRaw = await loadSessionFilesOnce(targetSessionId)
      if (targetEpoch !== sessionEpochRef.current || targetSessionId !== currentSessionIdRef.current) {
        return
      }
      setFiles(normalizeFileList(fileListRaw))
    } catch (e) {
      console.error('刷新文件列表失败:', e)
    }
  }, [enabled, sessionId, normalizeFileList])

  const loadModels = useCallback(async (targetSessionId: string, targetEpoch: number) => {
    try {
      const data = await loadModelsOnce()
      if (targetEpoch !== sessionEpochRef.current || targetSessionId !== currentSessionIdRef.current) {
        return
      }
      setAvailableModels(data.models)
      setDefaultModelId(data.default_model_id)
    } catch (e) {
      if (targetEpoch !== sessionEpochRef.current || targetSessionId !== currentSessionIdRef.current) {
        return
      }
      console.error('加载模型列表失败:', e)
      setAvailableModels([])
      setDefaultModelId(null)
    } finally {
      if (targetEpoch === sessionEpochRef.current && targetSessionId === currentSessionIdRef.current) {
        setModelsLoading(false)
      }
    }
  }, [])

  useEffect(() => {
    sessionEpochRef.current += 1
    const currentEpoch = sessionEpochRef.current

    stopMessageStream()
    stopEmptyStream()
    isSendMessageRef.current = false
    setStreamingState(false)
    setSkipEmptyStream(Boolean(initialSkipEmptyStream))
    lastEventIdRef.current = null
    emptyStreamRetryCountRef.current = 0

    if (!sessionId) {
      setLoading(false)
      setModelsLoading(false)
      setSession(null)
      sessionStatusRef.current = null
      setFiles([])
      setAvailableModels([])
      setDefaultModelId(null)
      setEvents([])
      setError(null)
      return
    }

    if (!enabled) {
      setLoading(false)
      setModelsLoading(false)
      setSession(null)
      sessionStatusRef.current = null
      setFiles([])
      setAvailableModels([])
      setDefaultModelId(null)
      setEvents([])
      setError(null)
      return
    }

    setLoading(true)
    setModelsLoading(true)
    setSession(null)
    sessionStatusRef.current = null
    setFiles([])
    setAvailableModels([])
    setDefaultModelId(null)
    setEvents([])
    setError(null)

    void loadSessionSnapshot(sessionId, currentEpoch)
    void loadModels(sessionId, currentEpoch)

    return () => {
      if (sessionEpochRef.current !== currentEpoch) return
      stopMessageStream()
      stopEmptyStream()
      isSendMessageRef.current = false
    }
  }, [enabled, initialSkipEmptyStream, loadModels, loadSessionSnapshot, sessionId, setStreamingState, stopEmptyStream, stopMessageStream])

  useEffect(() => {
    const status = session?.status
    if (!sessionId || !enabled) return
    const currentEpoch = sessionEpochRef.current
    if (shouldStartEmptySessionStream(status, isSendMessageRef.current, skipEmptyStream)) {
      startEmptyStream(currentEpoch)
    }
    return () => {
      if (sessionEpochRef.current !== currentEpoch) return
      stopEmptyStream()
    }
  }, [enabled, sessionId, session?.status, skipEmptyStream, startEmptyStream, stopEmptyStream])

  // 组件卸载时清理所有流，避免连接泄漏
  useEffect(() => {
    return () => {
      stopMessageStream()
      stopEmptyStream()
      isSendMessageRef.current = false
      sessionStatusRef.current = null
    }
  }, [stopEmptyStream, stopMessageStream])

  const openMessageStream = useCallback(
    async (
      chatParams:
        | { message: string; attachments: string[] }
        | { resume: { value: unknown } }
    ) => {
      if (!sessionId || !enabled) return

      const streamEpoch = sessionEpochRef.current
      const streamSessionId = sessionId

      stopEmptyStream()
      stopMessageStream()
      emptyStreamRetryCountRef.current = 0

      // 发送消息时，清除跳过空流的标记
      setSkipEmptyStream(false)
      isSendMessageRef.current = true
      setStreamingState(true)

      // 新消息会立刻启动新一轮执行，可以乐观切到 running。
      // resume 需要先经过后端 checkpoint 预校验，失败时会保持 waiting，
      // 因此前端不能在请求发出前抢先改成本地 running。
      if ('message' in chatParams) {
        sessionStatusRef.current = 'running'
        setSession((prev) => prev ? { ...prev, status: 'running' } : null)
      }

      const finalizeMessageStream = () => {
        if (streamEpoch !== sessionEpochRef.current || streamSessionId !== currentSessionIdRef.current) {
          return
        }
        setStreamingState(false)
        isSendMessageRef.current = false
        stopMessageStream()
      }

      const onEvent = (ev: SSEEventData) => {
        if (streamEpoch !== sessionEpochRef.current || streamSessionId !== currentSessionIdRef.current) {
          return
        }
        appendEvent(ev)
        if (ev.type === 'done') {
          finalizeMessageStream()
          setSession((prev) => prev ? { ...prev } : null)
        }
      }

      const messageStreamCleanup = sessionApi.chat(
        streamSessionId,
        chatParams,
        onEvent,
        (err) => {
          if (streamEpoch !== sessionEpochRef.current || streamSessionId !== currentSessionIdRef.current) {
            return
          }
          const closeReason = classifyMessageStreamCloseReason(err)
          if (closeReason === 'error') {
            setError(err instanceof Error ? err : new Error(t('sessionDetail.streamError')))
          }
          finalizeMessageStream()
          if (shouldReloadSnapshotAfterMessageStreamClose(closeReason)) {
            void loadSessionSnapshot(streamSessionId, streamEpoch)
          }
        }
      )

      if (streamEpoch !== sessionEpochRef.current || streamSessionId !== currentSessionIdRef.current) {
        messageStreamCleanup()
        return
      }

      // 将消息流的 cleanup 存到独立的 ref，不与 emptyStream 混淆
      messageStreamCleanupRef.current = messageStreamCleanup
    },
    [enabled, sessionId, appendEvent, loadSessionSnapshot, setStreamingState, stopEmptyStream, stopMessageStream, t]
  )

  const sendMessage = useCallback(
    async (message: string, attachmentIds: string[]) => {
      await openMessageStream({
        message,
        attachments: attachmentIds,
      })
    },
    [openMessageStream]
  )

  const resumeWaitingRun = useCallback(
    async (resumeValue: unknown) => {
      await openMessageStream({
        resume: {
          value: resumeValue,
        },
      })
    },
    [openMessageStream]
  )

  const updateSessionModel = useCallback(
    async (modelId: string) => {
      if (!sessionId || !enabled) return

      setModelUpdating(true)
      try {
        const response = await sessionApi.updateSessionModel(sessionId, { model_id: modelId })
        setSession((prev) => {
          if (!prev) return prev
          return {
            ...prev,
            current_model_id: response.current_model_id,
          }
        })
      } finally {
        setModelUpdating(false)
      }
    },
    [enabled, sessionId],
  )

  return {
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
  }
}
