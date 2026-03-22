'use client'

import { useMemo, useState } from 'react'
import { ChevronDown, ChevronUp } from 'lucide-react'
import { cn } from '@/lib/utils'
import { buildSkillDebugItems, type SkillDebugItem } from '@/lib/skill-debug'
import type { SSEEventData } from '@/lib/api/types'
import { useI18n } from '@/lib/i18n'
import type { Translate } from '@/lib/i18n'

export interface SkillDebugPanelProps {
  className?: string
  events: SSEEventData[]
}

const PREVIEW_COUNT = 3
const DEBUG_VALUE_LIMIT = 1200

function formatTime(ts: number | null, locale: 'zh-CN' | 'en-US'): string {
  if (ts == null) return locale === 'en-US' ? 'No timestamp' : '无时间戳'
  const date = new Date(ts)
  if (Number.isNaN(date.getTime())) return locale === 'en-US' ? 'No timestamp' : '无时间戳'
  return date.toLocaleString(locale === 'en-US' ? 'en-US' : 'zh-CN', { hour12: false })
}

function stringifyDebugValue(value: unknown): string | null {
  if (value === null || value === undefined) return null
  if (typeof value === 'string') {
    const trimmed = value.trim()
    if (!trimmed) return null
    return trimmed.length > DEBUG_VALUE_LIMIT ? `${trimmed.slice(0, DEBUG_VALUE_LIMIT)}...` : trimmed
  }
  try {
    const json = JSON.stringify(value, null, 2)
    if (!json) return null
    return json.length > DEBUG_VALUE_LIMIT ? `${json.slice(0, DEBUG_VALUE_LIMIT)}...` : json
  } catch {
    return String(value)
  }
}

function metadataRow(label: string, value: string | null) {
  if (!value) return null
  return (
    <p className="text-[11px] text-gray-600">
      <span className="text-gray-500">{label}</span>
      <span className="ml-1 text-gray-700">{value}</span>
    </p>
  )
}

function typeLabel(item: SkillDebugItem, t: Translate): string {
  return t(`sessionDetail.runTimelineKind.${item.eventType}`)
}

export function SkillDebugPanel({ className, events }: SkillDebugPanelProps) {
  const { locale, t } = useI18n()
  const [expanded, setExpanded] = useState(false)
  const items = useMemo(() => buildSkillDebugItems(events), [events])
  const reversedItems = useMemo(() => [...items].reverse(), [items])
  const visibleItems = expanded ? reversedItems : reversedItems.slice(0, PREVIEW_COUNT)
  const hiddenCount = reversedItems.length - visibleItems.length

  return (
    <div className={cn('rounded-xl border bg-white p-3', className)}>
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-800">
          {t('sessionDetail.skillDebugTitle')}
        </h3>
        {items.length > PREVIEW_COUNT && (
          <button
            type="button"
            className="inline-flex cursor-pointer items-center gap-1 text-xs text-gray-500 hover:text-gray-700"
            onClick={() => setExpanded((prev) => !prev)}
          >
            {expanded ? <ChevronUp className="size-4" /> : <ChevronDown className="size-4" />}
            <span>{expanded ? t('sessionDetail.skillDebugCollapse') : t('sessionDetail.skillDebugExpand')}</span>
          </button>
        )}
      </div>

      {items.length === 0 ? (
        <div className="mt-2 rounded-lg border border-dashed bg-gray-50 px-3 py-3">
          <p className="text-xs text-gray-500">{t('sessionDetail.skillDebugEmpty')}</p>
          <p className="mt-1 text-[11px] text-gray-400">{t('sessionDetail.skillDebugHint')}</p>
        </div>
      ) : (
        <div className="mt-2 flex max-h-64 flex-col gap-2 overflow-y-auto pr-1">
          {hiddenCount > 0 && (
            <p className="text-center text-[11px] text-gray-500">
              {t('sessionDetail.skillDebugHiddenCount', { count: hiddenCount })}
            </p>
          )}

          {visibleItems.map((item) => {
            const inputText = stringifyDebugValue(item.input)
            const outputText = stringifyDebugValue(item.output)
            return (
              <div key={item.id} className="rounded-lg border bg-gray-50 px-2.5 py-2">
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="rounded-md bg-blue-50 px-1.5 py-0.5 text-[11px] font-medium text-blue-700">
                    {item.skillId ?? t('sessionDetail.skillDebugUnknown')}
                  </span>
                  {item.skillVersion && (
                    <span className="rounded-md border border-gray-200 bg-white px-1.5 py-0.5 text-[11px] text-gray-600">
                      {t('sessionDetail.skillDebugVersion', { version: item.skillVersion })}
                    </span>
                  )}
                  <span className="rounded-md border border-gray-200 bg-white px-1.5 py-0.5 text-[11px] text-gray-600">
                    {t('sessionDetail.skillDebugEvent', { type: typeLabel(item, t) })}
                  </span>
                  <span className="ml-auto text-[11px] text-gray-500">{formatTime(item.timestamp, locale)}</span>
                </div>

                <div className="mt-1 space-y-0.5">
                  {metadataRow(t('sessionDetail.skillDebugStatusLabel'), item.status)}
                  {metadataRow(t('sessionDetail.skillDebugSubgraphLabel'), item.subgraph)}
                  {metadataRow(t('sessionDetail.skillDebugStepLabel'), item.stepId)}
                  {metadataRow(t('sessionDetail.skillDebugNodeLabel'), item.node)}
                  {metadataRow(t('sessionDetail.skillDebugFailedNodeLabel'), item.failedNode)}
                </div>

                {item.error && (
                  <div className="mt-1 rounded-md border border-red-200 bg-red-50 px-2 py-1.5 text-[11px] text-red-700">
                    <span className="text-red-600">{t('sessionDetail.skillDebugErrorLabel')}</span>
                    <span className="ml-1">{item.error}</span>
                  </div>
                )}

                {inputText && (
                  <div className="mt-1">
                    <p className="text-[11px] text-gray-500">{t('sessionDetail.skillDebugInputLabel')}</p>
                    <pre className="mt-1 overflow-x-auto rounded-md bg-white p-2 text-[11px] text-gray-700">{inputText}</pre>
                  </div>
                )}

                {outputText && (
                  <div className="mt-1">
                    <p className="text-[11px] text-gray-500">{t('sessionDetail.skillDebugOutputLabel')}</p>
                    <pre className="mt-1 overflow-x-auto rounded-md bg-white p-2 text-[11px] text-gray-700">{outputText}</pre>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
