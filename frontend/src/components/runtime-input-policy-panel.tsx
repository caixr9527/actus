'use client'

import { useMemo } from 'react'
import { cn } from '@/lib/utils'
import type { SSEEventData } from '@/lib/api/types'
import { useI18n } from '@/lib/i18n'
import {
  formatRuntimeInputPartSummary,
  hasRuntimeInputPolicySignal,
  pickLatestRuntimeInputPolicySnapshot,
} from '@/lib/runtime-input-policy'

export interface RuntimeInputPolicyPanelProps {
  className?: string
  events: SSEEventData[]
}

export function RuntimeInputPolicyPanel({ className, events }: RuntimeInputPolicyPanelProps) {
  const { t } = useI18n()
  const snapshot = useMemo(() => pickLatestRuntimeInputPolicySnapshot(events), [events])
  if (!hasRuntimeInputPolicySignal(snapshot)) return null

  return (
    <div className={cn('rounded-xl border bg-white px-3 py-2', className)}>
      <p className="text-xs font-semibold text-gray-800">{t('sessionDetail.runtimePolicyTitle')}</p>
      <div className="mt-1.5 flex flex-wrap items-center gap-1.5 text-[11px] text-gray-600">
        {snapshot.inputPartSummary && (
          <span className="rounded-md bg-gray-100 px-1.5 py-0.5">
            {t('sessionDetail.runtimeInputParts', {
              summary: formatRuntimeInputPartSummary(snapshot.inputPartSummary),
            })}
          </span>
        )}
        {snapshot.downgradeReason && (
          <span className="rounded-md bg-amber-50 px-1.5 py-0.5 text-amber-700">
            {t('sessionDetail.runtimeDowngradeReason', { reason: snapshot.downgradeReason })}
          </span>
        )}
        {snapshot.unsupportedParts.length > 0 && (
          <span className="rounded-md bg-red-50 px-1.5 py-0.5 text-red-700">
            {t('sessionDetail.runtimeUnsupportedParts', { count: snapshot.unsupportedParts.length })}
          </span>
        )}
      </div>
      {snapshot.unsupportedParts.length > 0 && (
        <div className="mt-2 rounded-md border border-red-100 bg-red-50/40 px-2 py-1.5">
          <p className="text-[11px] font-medium text-red-800">{t('sessionDetail.runtimeUnsupportedListTitle')}</p>
          <div className="mt-1 space-y-1">
            {snapshot.unsupportedParts.map((part, index) => (
              <div key={`${part.type}-${part.filepath ?? 'na'}-${index}`} className="text-[11px] text-red-800">
                <span>{t('sessionDetail.runtimeUnsupportedType', { type: part.type || 'unknown' })}</span>
                {part.reason && (
                  <span className="ml-2">{t('sessionDetail.runtimeUnsupportedReason', { reason: part.reason })}</span>
                )}
                {part.filepath && (
                  <div className="text-red-700">{t('sessionDetail.runtimeUnsupportedPath', { path: part.filepath })}</div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
