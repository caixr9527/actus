'use client'

import { useMemo, useState } from 'react'
import { CheckCircle2, ChevronDown, ChevronUp, Circle, Clock3, Loader2, XCircle } from 'lucide-react'
import { cn } from '@/lib/utils'
import { buildRunTimeline, buildStepViewState, type RunTimelineItem, type StepViewStatus } from '@/lib/run-timeline'
import type { SSEEventData } from '@/lib/api/types'
import { useI18n } from '@/lib/i18n'
import type { Translate } from '@/lib/i18n'

export interface RunTimelinePanelProps {
  className?: string
  events: SSEEventData[]
}

const TIMELINE_PREVIEW_COUNT = 12

function formatRelativeTime(ts: number | null, locale: 'zh-CN' | 'en-US'): string {
  if (ts == null) return locale === 'en-US' ? 'No timestamp' : '无时间戳'
  const now = Date.now()
  const diff = now - ts
  if (diff < 60_000) return locale === 'en-US' ? 'Just now' : '刚刚'
  if (diff < 3_600_000) {
    const m = Math.floor(diff / 60_000)
    return locale === 'en-US' ? `${m}m ago` : `${m}分钟前`
  }
  if (diff < 86_400_000) {
    const h = Math.floor(diff / 3_600_000)
    return locale === 'en-US' ? `${h}h ago` : `${h}小时前`
  }
  const d = Math.floor(diff / 86_400_000)
  return locale === 'en-US' ? `${d}d ago` : `${d}天前`
}

function kindLabel(item: RunTimelineItem, t: Translate): string {
  return t(`sessionDetail.runTimelineKind.${item.kind}`)
}

function statusLabel(status: StepViewStatus, t: Translate): string {
  return t(`sessionDetail.stepStatus.${status}`)
}

function statusIcon(status: StepViewStatus) {
  if (status === 'completed') return <CheckCircle2 className="size-4 text-emerald-600" />
  if (status === 'failed') return <XCircle className="size-4 text-red-600" />
  if (status === 'running') return <Loader2 className="size-4 animate-spin text-blue-600" />
  if (status === 'waiting') return <Clock3 className="size-4 text-amber-600" />
  return <Circle className="size-4 text-gray-400" />
}

function kindBadgeClass(kind: RunTimelineItem['kind']): string {
  if (kind === 'error') return 'bg-red-50 text-red-700 border-red-200'
  if (kind === 'done') return 'bg-emerald-50 text-emerald-700 border-emerald-200'
  if (kind === 'wait') return 'bg-amber-50 text-amber-700 border-amber-200'
  if (kind === 'tool') return 'bg-blue-50 text-blue-700 border-blue-200'
  if (kind === 'step') return 'bg-cyan-50 text-cyan-700 border-cyan-200'
  if (kind === 'plan') return 'bg-purple-50 text-purple-700 border-purple-200'
  return 'bg-gray-50 text-gray-700 border-gray-200'
}

export function RunTimelinePanel({ className, events }: RunTimelinePanelProps) {
  const { locale, t } = useI18n()
  const [expanded, setExpanded] = useState(false)
  const timelineItems = useMemo(() => buildRunTimeline(events, locale), [events, locale])
  const stepView = useMemo(() => buildStepViewState(events), [events])
  const visibleTimelineItems = expanded ? timelineItems : timelineItems.slice(-TIMELINE_PREVIEW_COUNT)
  const hiddenCount = timelineItems.length - visibleTimelineItems.length
  const latestTimelineItem = timelineItems.length > 0 ? timelineItems[timelineItems.length - 1] : null

  return (
    <div className={cn('rounded-xl border bg-white p-3', className)}>
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-800">{t('sessionDetail.runTimelineTitle')}</h3>
        <button
          type="button"
          className="inline-flex cursor-pointer items-center gap-1 text-xs text-gray-500 hover:text-gray-700"
          onClick={() => setExpanded((prev) => !prev)}
        >
          {expanded ? <ChevronUp className="size-4" /> : <ChevronDown className="size-4" />}
          <span>{expanded ? t('sessionDetail.runTimelineCollapse') : t('sessionDetail.runTimelineExpand')}</span>
        </button>
      </div>

      {!expanded ? (
        <div className="mt-2 rounded-lg border bg-gray-50 px-3 py-2">
          {latestTimelineItem ? (
            <>
              <div className="flex items-center justify-between gap-2">
                <div className="min-w-0 flex items-center gap-2">
                  <span className={cn('rounded-full border px-2 py-0.5 text-[11px]', kindBadgeClass(latestTimelineItem.kind))}>
                    {kindLabel(latestTimelineItem, t)}
                  </span>
                  <p className="truncate text-xs text-gray-700">{latestTimelineItem.summary}</p>
                </div>
                <span className="shrink-0 text-[11px] text-gray-500">
                  {formatRelativeTime(latestTimelineItem.timestamp, locale)}
                </span>
              </div>
              <div className="mt-1 flex items-center gap-2 text-[11px] text-gray-500">
                <span>{t('sessionDetail.runTimelineCompactSteps', { completed: stepView.completedCount, total: stepView.totalCount })}</span>
                <span>·</span>
                <span>{t('sessionDetail.runTimelineCompactEvents', { count: timelineItems.length })}</span>
              </div>
            </>
          ) : (
            <div className="text-xs text-gray-500">{t('sessionDetail.runTimelineEmpty')}</div>
          )}
        </div>
      ) : (
        <>
          <div className="mt-3 rounded-lg border bg-gray-50 px-3 py-2">
            <div className="mb-2 flex items-center justify-between">
              <span className="text-xs font-medium text-gray-600">{t('sessionDetail.stepViewTitle')}</span>
              <span className="text-xs text-gray-500">
                {stepView.completedCount}/{stepView.totalCount}
              </span>
            </div>
            {stepView.steps.length === 0 ? (
              <div className="text-xs text-gray-500">{t('sessionDetail.stepViewEmpty')}</div>
            ) : (
              <div className="flex max-h-32 flex-col gap-1 overflow-y-auto pr-1">
                {stepView.steps.map((step) => (
                  <div key={step.id} className="flex items-center gap-2 rounded-md bg-white px-2 py-1.5 text-xs">
                    {statusIcon(step.status)}
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-gray-700">{step.description}</p>
                      {step.detail && (
                        <p className="mt-0.5 line-clamp-2 text-[11px] text-gray-500">{step.detail}</p>
                      )}
                    </div>
                    <span className="shrink-0 text-[11px] text-gray-500">{statusLabel(step.status, t)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="mt-3">
            {timelineItems.length === 0 ? (
              <div className="rounded-lg border border-dashed px-3 py-4 text-xs text-gray-500">
                {t('sessionDetail.runTimelineEmpty')}
              </div>
            ) : (
              <div className="flex max-h-56 flex-col gap-1 overflow-y-auto pr-1">
                {hiddenCount > 0 && (
                  <div className="text-center text-[11px] text-gray-500">
                    {t('sessionDetail.runTimelineHiddenCount', { count: hiddenCount })}
                  </div>
                )}
                {visibleTimelineItems.map((item) => (
                  <div key={item.id} className="rounded-lg border bg-white px-2.5 py-2">
                    <div className="mb-1 flex items-center gap-2">
                      <span className={cn('rounded-full border px-2 py-0.5 text-[11px]', kindBadgeClass(item.kind))}>
                        {kindLabel(item, t)}
                      </span>
                      <span className="text-[11px] text-gray-500">{formatRelativeTime(item.timestamp, locale)}</span>
                    </div>
                    <p className="text-xs text-gray-700">{item.summary}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}
