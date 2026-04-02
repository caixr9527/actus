'use client'

import { useState } from 'react'
import { CheckCircle2, ChevronDown, Circle, Clock3, Loader2, XCircle } from 'lucide-react'
import { cn } from '@/lib/utils'
import { type StepViewState, type StepViewStatus } from '@/lib/run-timeline'
import { useI18n } from '@/lib/i18n'
import type { Translate } from '@/lib/i18n'

export interface RunTimelinePanelProps {
  className?: string
  stepView: StepViewState
}

type StepTone = {
  badgeClassName: string
  markerClassName: string
  titleClassName: string
}

function statusLabel(status: StepViewStatus, t: Translate): string {
  return t(`sessionDetail.stepStatus.${status}`)
}

function statusIcon(status: StepViewStatus) {
  if (status === 'completed') return <CheckCircle2 className="size-4 text-emerald-600" />
  if (status === 'failed') return <XCircle className="size-4 text-rose-600" />
  if (status === 'running') return <Loader2 className="size-4 animate-spin text-sky-600" />
  if (status === 'waiting') return <Clock3 className="size-4 text-amber-600" />
  return <Circle className="size-4 text-stone-400" />
}

function getStepTone(status: StepViewStatus): StepTone {
  if (status === 'completed') {
    return {
      badgeClassName: 'border-emerald-200 bg-emerald-50 text-emerald-700',
      markerClassName: 'bg-emerald-500',
      titleClassName: 'text-stone-700',
    }
  }
  if (status === 'failed') {
    return {
      badgeClassName: 'border-rose-200 bg-rose-50 text-rose-700',
      markerClassName: 'bg-rose-500',
      titleClassName: 'text-stone-700',
    }
  }
  if (status === 'running') {
    return {
      badgeClassName: 'border-sky-200 bg-sky-50 text-sky-700',
      markerClassName: 'bg-sky-500',
      titleClassName: 'text-stone-900',
    }
  }
  if (status === 'waiting') {
    return {
      badgeClassName: 'border-amber-200 bg-amber-50 text-amber-700',
      markerClassName: 'bg-amber-500',
      titleClassName: 'text-stone-800',
    }
  }
  return {
    badgeClassName: 'border-stone-200 bg-stone-50 text-stone-600',
    markerClassName: 'bg-stone-300',
    titleClassName: 'text-stone-500',
  }
}

export function RunTimelinePanel({ className, stepView }: RunTimelinePanelProps) {
  const { t } = useI18n()
  const [expanded, setExpanded] = useState(false)

  const totalCount = stepView.totalCount
  const completedCount = stepView.completedCount
  const progressPercent = totalCount > 0 ? Math.round((completedCount / totalCount) * 100) : 0

  return (
    <div className={cn('px-1', className)}>
      <div className="overflow-hidden rounded-2xl border border-stone-200/80 bg-white/92">
        <button
          type="button"
          className="flex w-full items-center gap-3 px-3 py-2.5 text-left transition-colors hover:bg-stone-50/70 cursor-pointer sm:px-4"
          onClick={() => setExpanded((prev) => !prev)}
        >
          <span className="shrink-0 text-xs font-medium tracking-[0.14em] text-stone-500 uppercase">
            {t('sessionDetail.runTimelineTitle')}
          </span>

          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-3">
              <span className="shrink-0 text-xs text-stone-500">
                {completedCount}/{totalCount || 0}
              </span>
              <div className="min-w-0 flex-1">
                {!expanded ? (
                  <div className="h-1 rounded-full bg-stone-200">
                    <div
                      className="h-full rounded-full bg-stone-900 transition-[width] duration-300"
                      style={{ width: `${progressPercent}%` }}
                    />
                  </div>
                ) : (
                  <div className="h-px bg-transparent" />
                )}
              </div>
            </div>
          </div>

          <div className="grid size-7 shrink-0 place-items-center rounded-full border border-stone-200 bg-white text-stone-500">
            <ChevronDown className={cn('size-4 transition-transform duration-200', expanded && 'rotate-180')} />
          </div>
        </button>

        {expanded ? (
          <div className="border-t border-stone-200/80 px-3 py-2.5 sm:px-4">
            {stepView.steps.length === 0 ? (
              <div className="py-3 text-sm text-stone-500">
                {t('sessionDetail.stepViewEmpty')}
              </div>
            ) : (
              <div className="relative">
                <div className="absolute bottom-4 left-[11px] top-4 w-px bg-stone-200" />
                <div className="space-y-1">
                {stepView.steps.map((step) => {
                  const tone = getStepTone(step.status)
                  return (
                    <div
                      key={step.id}
                      className="relative pl-8"
                    >
                      <span className={cn('absolute left-0 top-3 size-[7px] rounded-full', tone.markerClassName)} />
                      <div className="flex min-h-8 items-center justify-between gap-3 py-1">
                        <div className="min-w-0 flex items-center gap-3">
                          <span className="w-6 shrink-0 text-xs font-medium tabular-nums text-stone-400">
                            {String(step.stepIndex + 1).padStart(2, '0')}
                          </span>
                          <p className={cn('truncate text-sm', tone.titleClassName)}>
                            {step.description}
                          </p>
                        </div>
                        <span
                          className={cn(
                            'shrink-0 rounded-full border px-2 py-0.5 text-[11px] font-medium',
                            tone.badgeClassName,
                          )}
                        >
                          <span className="inline-flex items-center gap-1.5">
                            {statusIcon(step.status)}
                            <span>{statusLabel(step.status, t)}</span>
                          </span>
                        </span>
                      </div>
                    </div>
                  )
                })}
                </div>
              </div>
            )}
          </div>
        ) : null}
      </div>
    </div>
  )
}
