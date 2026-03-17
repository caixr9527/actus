'use client'

import { useEffect, useState } from 'react'
import { useSearchParams } from 'next/navigation'
import { SessionDetailView } from '@/components/session-detail-view'
import {
  consumeInitialMessageDraft,
  parseLegacyInitQueryParam,
} from '@/lib/initial-message-draft'
import { useI18n } from '@/lib/i18n'

interface PageProps {
  params: Promise<{ id: string }>
}

/**
 * 任务详情页：展示会话标题、事件时间线、任务进度与输入框。
 * - 通过 getSessionDetail 获取任务详情与事件列表（若后端返回 events）
 * - 未完成任务通过 chat 空 body 流式拉取事件
 * - 发送消息通过 chat 带 message/attachments 流式追加事件
 * - 首页跳转优先使用 sessionStorage 传递初始消息（URL 仅作兼容回退）
 */
export default function SessionDetailPage({ params }: PageProps) {
  const { t } = useI18n()
  const searchParams = useSearchParams()
  const [sessionData, setSessionData] = useState<{
    id: string
    initialMessage?: string
    initialAttachments?: string[]
    hasInitialMessage: boolean
  } | null>(null)
  
  useEffect(() => {
    let cancelled = false

    const loadSessionData = async () => {
      const p = await params
      if (cancelled) return

      const storageDraft = consumeInitialMessageDraft(p.id)
      if (storageDraft) {
        setSessionData({
          id: p.id,
          initialMessage: storageDraft.message,
          initialAttachments: storageDraft.attachments,
          hasInitialMessage: true,
        })
        return
      }

      const legacyDraft = parseLegacyInitQueryParam(searchParams.get('init'))
      if (legacyDraft) {
        setSessionData({
          id: p.id,
          initialMessage: legacyDraft.message,
          initialAttachments: legacyDraft.attachments,
          hasInitialMessage: true,
        })
        return
      }

      setSessionData({
        id: p.id,
        hasInitialMessage: false,
      })
    }

    void loadSessionData()
    return () => {
      cancelled = true
    }
  }, [params, searchParams])

  if (!sessionData) {
    return <div className="flex items-center justify-center h-full">{t('common.loading')}</div>
  }

  return (
    <SessionDetailView
      sessionId={sessionData.id}
      initialMessage={sessionData.initialMessage}
      initialAttachments={sessionData.initialAttachments}
      hasInitialMessage={sessionData.hasInitialMessage}
    />
  )
}
