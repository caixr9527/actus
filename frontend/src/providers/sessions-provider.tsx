'use client'

import React, {createContext, useCallback, useContext, useEffect, useRef, useState} from 'react'
import {getApiErrorMessage, sessionApi} from '@/lib/api'
import type {Session} from '@/lib/api'
import {canRetry, computeRetryDelayMs, type RetryPolicy} from '@/lib/session-stream-policy'
import { useI18n } from '@/lib/i18n'

/** 重连配置 */
const RETRY_POLICY: RetryPolicy = {
  maxRetries: 10,
  baseDelayMs: 1000,
  maxDelayMs: 30_000,
}

const FALLBACK_POLL_INTERVAL_MS = 15_000

export type RealtimeStatus = 'connected' | 'reconnecting' | 'degraded'

/**
 * 从 API 返回值中安全提取 Session 数组
 * 兼容 data 为 { sessions: [...] } / 直接数组 / null 等格式
 */
function normalizeSessions(raw: unknown): Session[] {
  if (Array.isArray(raw)) return raw as Session[]
  if (raw && typeof raw === 'object' && 'sessions' in raw) {
    return Array.isArray((raw as Record<string, unknown>).sessions)
      ? ((raw as Record<string, unknown>).sessions as Session[])
      : []
  }
  return []
}

// ==================== Context ====================

type SessionsContextValue = {
  sessions: Session[]
  loading: boolean
  error: string | null
  realtimeStatus: RealtimeStatus
  reconnectCount: number
  realtimeAlert: string | null
  /** 手动刷新（通过 REST 接口拉取一次） */
  refresh: () => Promise<void>
  resumeRealtime: () => void
  deleteSession: (sessionId: string) => Promise<boolean>
}

const SessionsContext = createContext<SessionsContextValue | null>(null)

// ==================== Provider ====================

/**
 * 会话列表数据 Provider
 *
 * 放置在 root layout 中，确保不会因为侧边栏展开/折叠而重新挂载。
 *
 * 数据流:
 *  1. 挂载后立即通过 REST GET /sessions 获取初始数据（仅一次）
 *  2. 同时建立 SSE POST /sessions/stream 长连接，接收实时推送
 *  3. SSE 断开后自动指数退避重连
 *  4. refresh() 可手动通过 REST 拉取
 */
export function SessionsProvider({children}: { children: React.ReactNode }) {
  const { t } = useI18n()
  const [sessions, setSessions] = useState<Session[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [realtimeStatus, setRealtimeStatus] = useState<RealtimeStatus>('connected')
  const [reconnectCount, setReconnectCount] = useState(0)
  const [realtimeAlert, setRealtimeAlert] = useState<string | null>(null)

  const cleanupRef = useRef<(() => void) | null>(null)
  const retryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const fallbackPollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const retryCountRef = useRef(0)
  const connectRef = useRef<(() => void) | null>(null)
  const unmountedRef = useRef(false)
  /** 确保 REST 初始请求只发起一次（防止 Strict Mode 重复） */
  const initialFetchedRef = useRef(false)
  /** 标记 SSE 是否已经推送过数据，防止 REST 回调覆盖更新的 SSE 数据 */
  const sseReceivedRef = useRef(false)

  const clearRetryTimer = useCallback(() => {
    if (retryTimerRef.current) {
      clearTimeout(retryTimerRef.current)
      retryTimerRef.current = null
    }
  }, [])

  const clearFallbackPollTimer = useCallback(() => {
    if (fallbackPollTimerRef.current) {
      clearInterval(fallbackPollTimerRef.current)
      fallbackPollTimerRef.current = null
    }
  }, [])

  const fetchSessionsSnapshot = useCallback(async () => {
    const raw = await sessionApi.getSessions()
    return normalizeSessions(raw)
  }, [])

  const startFallbackPolling = useCallback(() => {
    if (fallbackPollTimerRef.current) return
    fallbackPollTimerRef.current = setInterval(() => {
      if (unmountedRef.current) return
      void fetchSessionsSnapshot()
        .then((nextSessions) => {
          if (unmountedRef.current) return
          setSessions(nextSessions)
          setLoading(false)
        })
        .catch((err) => {
          console.warn('[Sessions] 降级同步失败:', err)
        })
    }, FALLBACK_POLL_INTERVAL_MS)
  }, [fetchSessionsSnapshot])

  const markRealtimeConnected = useCallback(() => {
    retryCountRef.current = 0
    setReconnectCount(0)
    setRealtimeStatus('connected')
    setRealtimeAlert(null)
    clearFallbackPollTimer()
  }, [clearFallbackPollTimer])

  const resumeRealtime = useCallback(() => {
    retryCountRef.current = 0
    setReconnectCount(0)
    setRealtimeStatus('reconnecting')
    setRealtimeAlert(t('sessionsProvider.resumingRealtime'))
    clearRetryTimer()
    clearFallbackPollTimer()
    connectRef.current?.()
  }, [clearFallbackPollTimer, clearRetryTimer, t])

  // ---------- 手动刷新 ----------
  const refresh = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const nextSessions = await fetchSessionsSnapshot()
      setSessions(nextSessions)
      if (realtimeStatus === 'degraded') {
        resumeRealtime()
      }
    } catch (err) {
      console.error('[Sessions] REST 获取失败:', err)
      setError(getApiErrorMessage(err, 'sessionsProvider.fetchFailed', t))
    } finally {
      setLoading(false)
    }
  }, [fetchSessionsSnapshot, realtimeStatus, resumeRealtime, t])

  // ---------- 初始 REST 请求（仅一次） ----------
  useEffect(() => {
    if (initialFetchedRef.current) return
    initialFetchedRef.current = true

    fetchSessionsSnapshot()
      .then((raw) => {
        // 仅在 SSE 尚未推送过数据时更新，防止用旧数据覆盖 SSE 已推送的新数据
        if (!sseReceivedRef.current) {
          setSessions(raw)
        }
        setLoading(false)
        setError(null)
      })
      .catch((err) => {
        console.error('[Sessions] 初始获取失败:', err)
        setError(getApiErrorMessage(err, 'sessionsProvider.fetchFailed', t))
        setLoading(false)
      })
  }, [fetchSessionsSnapshot, t])

  // ---------- SSE 实时订阅 ----------
  useEffect(() => {
    unmountedRef.current = false
    let mounted = true

    const connect = () => {
      if (!mounted) return

      clearRetryTimer()

      // 清理上一次连接
      if (cleanupRef.current) {
        cleanupRef.current()
        cleanupRef.current = null
      }

      const cleanup = sessionApi.streamSessions(
        // onSessions
        (newSessions) => {
          if (!mounted) return
          markRealtimeConnected()
          sseReceivedRef.current = true
          setSessions(newSessions)
          setLoading(false)
          setError(null)
        },
        // onError / onEnd
        (err) => {
          if (!mounted) return
          console.warn('[Sessions] SSE 断开:', err.message)

          if (!canRetry(retryCountRef.current, RETRY_POLICY)) {
            setRealtimeStatus('degraded')
            setRealtimeAlert(t('sessionsProvider.realtimeDisconnectedFallback'))
            startFallbackPolling()
            console.error('[Sessions] 超过最大重试次数，已切换为降级同步')
            return
          }

          const delay = computeRetryDelayMs(retryCountRef.current, RETRY_POLICY)
          retryCountRef.current += 1
          setReconnectCount(retryCountRef.current)
          setRealtimeStatus('reconnecting')
          setRealtimeAlert(
            t('sessionsProvider.reconnectingInSeconds', {
              seconds: Math.ceil(delay / 1000),
              count: retryCountRef.current,
            }),
          )
          console.log(`[Sessions] ${delay}ms 后尝试重连（第 ${retryCountRef.current} 次）`)
          retryTimerRef.current = setTimeout(connect, delay)
        },
      )

      cleanupRef.current = cleanup
    }

    connectRef.current = connect
    connect()

    return () => {
      mounted = false
      unmountedRef.current = true
      connectRef.current = null
      if (cleanupRef.current) {
        cleanupRef.current()
        cleanupRef.current = null
      }
      clearRetryTimer()
      clearFallbackPollTimer()
    }
  }, [clearFallbackPollTimer, clearRetryTimer, markRealtimeConnected, startFallbackPolling, t])

  // ---------- 删除会话 ----------
  const deleteSession = useCallback(async (sessionId: string): Promise<boolean> => {
    try {
      await sessionApi.deleteSession(sessionId)
      setSessions((prev) => prev.filter((s) => s.session_id !== sessionId))
      return true
    } catch {
      return false
    }
  }, [])

  return (
    <SessionsContext.Provider value={{
      sessions,
      loading,
      error,
      realtimeStatus,
      reconnectCount,
      realtimeAlert,
      refresh,
      resumeRealtime,
      deleteSession,
    }}>
      {children}
    </SessionsContext.Provider>
  )
}

// ==================== Hook ====================

/**
 * 获取会话列表数据的 Hook
 *
 * 必须在 <SessionsProvider> 内使用
 */
export function useSessions(): SessionsContextValue {
  const ctx = useContext(SessionsContext)
  if (!ctx) {
    throw new Error('useSessions 必须在 SessionsProvider 内使用')
  }
  return ctx
}
