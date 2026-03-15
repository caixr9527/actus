'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { sessionApi } from '@/lib/api/session'
import { normalizeEvent, normalizeEvents } from '@/lib/session-events'
import { canRetry, computeRetryDelayMs, shouldStartEmptySessionStream, type RetryPolicy } from '@/lib/session-stream-policy'
import {
  classifyMessageStreamCloseReason,
  reduceSessionRuntimeStateOnEvent,
  shouldReloadSnapshotAfterMessageStreamClose,
} from '@/lib/session-detail-runtime'
import type { SessionDetail, SessionStatus, SSEEventData, SessionFile } from '@/lib/api/types'

export type UseSessionDetailResult = {
  session: SessionDetail | null
  files: SessionFile[]
  events: SSEEventData[]
  loading: boolean
  error: Error | null
  refresh: () => Promise<void>
  refreshFiles: () => Promise<void>
  sendMessage: (message: string, attachmentIds: string[]) => Promise<void>
  streaming: boolean
}

const EMPTY_STREAM_RECONNECT_DELAY = 500
const EMPTY_STREAM_RETRY_POLICY: RetryPolicy = {
  maxRetries: 8,
  baseDelayMs: 1000,
  maxDelayMs: 10_000,
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
  const [session, setSession] = useState<SessionDetail | null>(null)
  const [files, setFiles] = useState<SessionFile[]>([])
  const [events, setEvents] = useState<SSEEventData[]>([])
  const [loading, setLoading] = useState(true)
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
    let evToAppend = ev
    if (ev.data && typeof ev.data === 'object' && ('event' in ev.data || 'type' in ev.data) && 'data' in ev.data) {
      const normalized = normalizeEvent(ev.data as { event?: string; type?: string; data?: unknown })
      if (normalized) evToAppend = normalized
    }

    const eventId = (evToAppend.data as { event_id?: string })?.event_id
    if (eventId) lastEventIdRef.current = eventId

    setEvents((prev) => [...prev, evToAppend])

    // 更新会话标题
    if (evToAppend.type === 'title' && evToAppend.data && typeof (evToAppend.data as { title?: string }).title === 'string') {
      setSession((prev) =>
        prev ? { ...prev, title: (evToAppend.data as { title: string }).title } : null
      )
    }

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
            setError(new Error('会话实时连接中断，请点击重试恢复'))
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
  }, [appendEvent, clearEmptyStreamReconnectTimer, sessionId, stopEmptyStream])

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
      const [detail, fileListRaw] = await Promise.all([
        sessionApi.getSessionDetail(targetSessionId),
        sessionApi.getSessionFiles(targetSessionId),
      ])

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
      setError(e instanceof Error ? e : new Error('加载失败'))
      return null
    } finally {
      if (targetEpoch === sessionEpochRef.current && targetSessionId === currentSessionIdRef.current) {
        setLoading(false)
      }
    }
  }, [normalizeFileList])

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
      const fileListRaw = await sessionApi.getSessionFiles(targetSessionId)
      if (targetEpoch !== sessionEpochRef.current || targetSessionId !== currentSessionIdRef.current) {
        return
      }
      setFiles(normalizeFileList(fileListRaw))
    } catch (e) {
      console.error('刷新文件列表失败:', e)
    }
  }, [enabled, sessionId, normalizeFileList])

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
      setSession(null)
      sessionStatusRef.current = null
      setFiles([])
      setEvents([])
      setError(null)
      return
    }

    if (!enabled) {
      setLoading(false)
      setSession(null)
      sessionStatusRef.current = null
      setFiles([])
      setEvents([])
      setError(null)
      return
    }

    setLoading(true)
    setSession(null)
    sessionStatusRef.current = null
    setFiles([])
    setEvents([])
    setError(null)

    void loadSessionSnapshot(sessionId, currentEpoch)

    return () => {
      if (sessionEpochRef.current !== currentEpoch) return
      stopMessageStream()
      stopEmptyStream()
      isSendMessageRef.current = false
    }
  }, [enabled, initialSkipEmptyStream, loadSessionSnapshot, sessionId, setStreamingState, stopEmptyStream, stopMessageStream])

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

  const sendMessage = useCallback(
    async (message: string, attachmentIds: string[]) => {
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

      // 立即更新状态为 running，不等待 SSE 事件
      sessionStatusRef.current = 'running'
      setSession((prev) => prev ? { ...prev, status: 'running' } : null)

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
        { message, attachments: attachmentIds },
        onEvent,
        (err) => {
          if (streamEpoch !== sessionEpochRef.current || streamSessionId !== currentSessionIdRef.current) {
            return
          }
          const closeReason = classifyMessageStreamCloseReason(err)
          if (closeReason === 'error') {
            setError(err instanceof Error ? err : new Error('流式响应异常'))
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
    [enabled, sessionId, appendEvent, loadSessionSnapshot, setStreamingState, stopEmptyStream, stopMessageStream]
  )

  return {
    session,
    files,
    events,
    loading,
    error,
    refresh,
    refreshFiles,
    sendMessage,
    streaming,
  }
}
