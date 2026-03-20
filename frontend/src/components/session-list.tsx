'use client'

import {useCallback, useState} from 'react'
import {useParams, useRouter} from 'next/navigation'
import {toast} from 'sonner'
import {ItemGroup} from '@/components/ui/item'
import {SessionItem} from '@/components/session-item'
import {DeleteSessionDialog} from '@/components/delete-session-dialog'
import {useSessions} from '@/hooks/use-sessions'
import type {Session} from '@/lib/api'
import { useI18n } from '@/lib/i18n'

/**
 * 会话列表组件
 * 负责渲染列表、处理路由导航及删除操作
 */
export function SessionList() {
  const router = useRouter()
  const params = useParams()
  const { t } = useI18n()
  const {sessions, loading, error, realtimeStatus, reconnectCount, realtimeAlert, refresh, resumeRealtime, deleteSession} = useSessions()

  // 待删除的会话
  const [pendingDeleteSession, setPendingDeleteSession] = useState<Session | null>(null)

  const handleSessionClick = useCallback((sessionId: string) => {
    router.push(`/sessions/${sessionId}`)
  }, [router])

  const handleDeleteRequest = useCallback((session: Session) => {
    setPendingDeleteSession(session)
  }, [])

  const handleDeleteConfirm = useCallback(async () => {
    if (!pendingDeleteSession) return

    const sessionTitle = pendingDeleteSession.title || t('session.newTask')
    const success = await deleteSession(pendingDeleteSession.session_id)

    if (success) {
      toast.success(t('sessionList.deleteSuccess', { title: sessionTitle }))
      // 如果删除的是当前正在查看的会话，跳转到首页
      if (params?.id === pendingDeleteSession.session_id) {
        router.push('/')
      }
    } else {
      toast.error(t('sessionList.deleteFailed', { title: sessionTitle }))
    }

    setPendingDeleteSession(null)
  }, [pendingDeleteSession, deleteSession, params?.id, router, t])

  const handleDialogOpenChange = useCallback((open: boolean) => {
    if (!open) {
      setPendingDeleteSession(null)
    }
  }, [])

  const handleRealtimeRetry = useCallback(() => {
    resumeRealtime()
    void refresh()
  }, [refresh, resumeRealtime])

  // 加载态：骨架屏
  if (loading) {
    return (
      <ItemGroup className="gap-1">
        {Array.from({length: 3}).map((_, i) => (
          <div
            key={i}
            className="flex items-center gap-2 p-2 animate-pulse"
          >
            <div className="size-8 rounded-full bg-muted"/>
            <div className="flex-1 space-y-1.5">
              <div className="h-3.5 bg-muted rounded w-3/4"/>
              <div className="h-3 bg-muted rounded w-1/2"/>
            </div>
          </div>
        ))}
      </ItemGroup>
    )
  }

  // 错误态
  if (error) {
    return (
      <div className="flex flex-col items-center gap-2 py-8 text-sm text-muted-foreground">
        <p>{t('sessionList.loadFailed')}</p>
        <button
          className="text-primary underline underline-offset-4 cursor-pointer"
          onClick={refresh}
        >
          {t('common.retry')}
        </button>
      </div>
    )
  }

  // 空态
  if (sessions.length === 0) {
    return (
      <div className="space-y-3">
        {realtimeAlert && (
          <div className="rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-800">
            <div className="flex items-center justify-between gap-2">
              <span>{realtimeAlert}</span>
              <button
                type="button"
                className="text-amber-900 underline underline-offset-2 cursor-pointer"
                onClick={handleRealtimeRetry}
              >
                {t('common.retry')}
              </button>
            </div>
          </div>
        )}
        <div className="py-8 text-center text-sm text-muted-foreground">
          {t('sessionList.empty')}
        </div>
      </div>
    )
  }

  return (
    <>
      {realtimeAlert && (
        <div className="mb-2 rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-800">
          <div className="flex items-center justify-between gap-2">
            <span>
              {realtimeAlert}
              {realtimeStatus === 'reconnecting'
                ? t('sessionList.reconnectingCount', { count: reconnectCount })
                : ''}
            </span>
            <button
              type="button"
              className="text-amber-900 underline underline-offset-2 cursor-pointer"
              onClick={handleRealtimeRetry}
            >
              {t('common.retry')}
            </button>
          </div>
        </div>
      )}
      <ItemGroup className="gap-1">
        {sessions.map((session) => (
          <SessionItem
            key={session.session_id}
            session={session}
            isActive={session.session_id === String(params?.id ?? '')}
            onClick={handleSessionClick}
            onDelete={handleDeleteRequest}
          />
        ))}
      </ItemGroup>

      {/* 删除确认弹窗 */}
      <DeleteSessionDialog
        open={!!pendingDeleteSession}
        onOpenChange={handleDialogOpenChange}
        onConfirm={handleDeleteConfirm}
      />
    </>
  )
}
