'use client'

import { useState } from 'react'
import { Ban, ChevronDown, Clock3, Loader2, XCircle } from 'lucide-react'
import { cn } from '@/lib/utils'
import { type StepViewState, type StepViewStatus } from '@/lib/run-timeline'
import { useI18n } from '@/lib/i18n'

export interface RunTimelinePanelProps {
  className?: string
  stepView: StepViewState
}

function renderStepStatusIcon(status: StepViewStatus) {
  if (status === 'running') {
    return <Loader2 className="size-3.5 animate-spin text-stone-900" />
  }
  if (status === 'completed') {
    return <span className="block size-2.5 rounded-full bg-stone-900" />
  }
  if (status === 'failed') {
    return <XCircle className="size-3.5 text-rose-600" />
  }
  if (status === 'cancelled') {
    return <Ban className="size-3.5 text-stone-500" />
  }
  if (status === 'waiting') {
    return <Clock3 className="size-3.5 text-amber-600" />
  }
  return <span className="block size-2.5 rounded-full bg-stone-300" />
}

function getStepTitleClassName(status: StepViewStatus) {
  if (status === 'running') return 'text-stone-950'
  if (status === 'completed') return 'text-stone-700'
  if (status === 'failed') return 'text-rose-700'
  if (status === 'cancelled') return 'text-stone-500'
  if (status === 'waiting') return 'text-stone-800'
  return 'text-stone-500'
}

export function RunTimelinePanel({ className, stepView }: RunTimelinePanelProps) {
  const { t } = useI18n()
  const [expanded, setExpanded] = useState(false)

  const totalCount = stepView.totalCount
  const progressCount = stepView.progressCount
  const progressPercent = totalCount > 0 ? Math.round((progressCount / totalCount) * 100) : 0

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
                {progressCount}/{totalCount || 0}
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
                <div className="absolute bottom-3 left-[7px] top-3 w-px bg-stone-200" />
                <div className="space-y-1.5">
                {stepView.steps.map((step) => {
                  return (
                    <div
                      key={step.id}
                      className="relative pl-6"
                    >
                      <span className="absolute left-0 top-2.5 inline-flex size-4 items-center justify-center">
                        {renderStepStatusIcon(step.status)}
                      </span>
                      <div className="flex min-h-8 items-center gap-3 py-0.5">
                        <div className="min-w-0 flex items-center gap-2.5">
                          <span className="w-6 shrink-0 text-[11px] font-medium tabular-nums tracking-[0.14em] text-stone-400">
                            {String(step.stepIndex + 1).padStart(2, '0')}
                          </span>
                          <p className={cn('truncate text-sm leading-6', getStepTitleClassName(step.status))}>
                            {step.description}
                          </p>
                        </div>
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
