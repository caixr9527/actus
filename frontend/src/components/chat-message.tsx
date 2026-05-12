'use client'

import { useEffect, useRef, useState } from 'react'
import { cn } from '@/lib/utils'
import { Ban, CheckCircle2, ChevronDown, Clock3, Languages, Loader2, XCircle } from 'lucide-react'
import { ManusIcon } from '@/components/manus-icon'
import { ToolUse } from '@/components/tool-use'
import { AttachmentsMessage } from '@/components/attachments-message'
import { MarkdownContent } from '@/components/markdown-content'
import { useI18n } from '@/lib/i18n'
import type { AppLocale } from '@/lib/i18n'
import type { StepEvent, ToolEvent } from '@/lib/api/types'
import { type TimelineItem, type AttachmentFile, getToolTimeLabel } from '@/lib/session-events'
import { resolveStepDetail } from '@/lib/run-timeline'
import {
  getInitialAssistantTurnExpandedState,
  resolveAssistantTurnExpandedState,
  resolveStepExpandedState,
  shouldAutoExpandStep,
} from '@/lib/session-detail-view-state'
import type { AssistantTurnItem, ConversationItem } from '@/lib/assistant-turns'

export interface ChatMessageProps {
  className?: string
  item: ConversationItem
  onViewAllFiles?: () => void
  onFileClick?: (file: AttachmentFile) => void
  onToolClick?: (tool: ToolEvent) => void
}

function getStepStatusIcon(status: StepEvent['status']) {
  if (status === 'completed') return <CheckCircle2 className="size-4 text-emerald-600" />
  if (status === 'failed') return <XCircle className="size-4 text-red-600" />
  if (status === 'cancelled') return <Ban className="size-4 text-stone-500" />
  if (status === 'running') return <Loader2 className="size-4 animate-spin text-blue-600" />
  return <Clock3 className="size-4 text-amber-600" />
}

function ToolRow({
  className,
  timeLabel,
  fallbackTimeLabel,
  children,
}: {
  className?: string
  timeLabel?: string
  fallbackTimeLabel: string
  children: React.ReactNode
}) {
  const [hovered, setHovered] = useState(false)
  return (
    <div
      className={cn(
        'flex items-center justify-between gap-2 mt-3 w-full min-w-0',
        className
      )}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div className="min-w-0 flex-shrink-0">{children}</div>
      <span
        className={cn(
          'flex-shrink-0 text-xs text-gray-500 min-w-[2.5rem] text-right transition-opacity duration-150',
          hovered ? 'opacity-100' : 'opacity-0'
        )}
      >
        {timeLabel ?? fallbackTimeLabel}
      </span>
    </div>
  )
}

export function ChatMessage({
  className,
  item,
  onViewAllFiles,
  onFileClick,
  onToolClick,
}: ChatMessageProps) {
  const { locale, t } = useI18n()
  if (item.kind === 'user') {
    return (
      <div
        className={cn(
          'flex w-full flex-col items-end justify-end gap-1 group mt-3',
          className
        )}
      >
        <div className="flex max-w-[90%] relative flex-col gap-2 items-end">
          <div className="text-gray-700 relative flex items-center rounded-lg overflow-hidden bg-white p-3 border whitespace-pre-wrap break-words text-sm leading-relaxed">
            {item.data.message ?? ''}
          </div>
        </div>
      </div>
    )
  }

  if (item.kind === 'assistant') {
    return <AssistantMessageBlock item={item} className={className} />
  }

  if (item.kind === 'assistant_turn') {
    return (
      <AssistantTurnBlock
        item={item}
        className={className}
        onViewAllFiles={onViewAllFiles}
        onFileClick={onFileClick}
        onToolClick={onToolClick}
        locale={locale}
      />
    )
  }

  if (item.kind === 'tool') {
    return (
      <ToolRow
        className={className}
        timeLabel={item.timeLabel}
        fallbackTimeLabel={t('common.justNow')}
      >
        <ToolUse data={item.data} onClick={onToolClick ? () => onToolClick(item.data) : undefined} />
      </ToolRow>
    )
  }

  if (item.kind === 'step') {
    return (
      <StepBlock
        stepItem={item}
        className={className}
        onToolClick={onToolClick}
        locale={locale}
      />
    )
  }

  if (item.kind === 'attachments') {
    return (
      <div className={cn('mt-3', className)}>
        <AttachmentsMessage
          role={item.role}
          files={item.files}
          onViewAllFiles={item.role === 'assistant' ? onViewAllFiles : undefined}
          onFileClick={onFileClick}
        />
      </div>
    )
  }

  if (item.kind === 'error') {
    return (
      <div
        className={cn('flex flex-col gap-2 w-full group mt-3', className)}
      >
        <div className="flex items-center justify-between h-7 group">
          <div className="flex items-center justify-center gap-1 text-red-600">
            <Languages size={18} />
            <ManusIcon />
          </div>
        </div>
        <div className="max-w-none p-0 m-0 text-red-600">
          <MarkdownContent content={item.error} />
        </div>
      </div>
    )
  }

  return null
}

function AssistantMessageBlock({
  item,
  className,
}: {
  item: Extract<TimelineItem, { kind: 'assistant' }>
  className?: string
}) {
  return (
    <div
      className={cn('flex flex-col gap-2 w-full group mt-3', className)}
    >
      <div className="flex items-center justify-between h-7 group">
        <div className="flex items-center justify-center gap-1 text-gray-700">
          <Languages size={18} />
          <ManusIcon />
        </div>
      </div>
      <div className="max-w-none p-0 m-0 text-gray-700">
        <MarkdownContent content={item.data.message ?? ''} />
      </div>
    </div>
  )
}

function getProcessStatusIcon(status: StepEvent['status']) {
  if (status === 'completed') return <CheckCircle2 className="size-4 text-stone-400" />
  if (status === 'failed') return <XCircle className="size-4 text-red-500" />
  if (status === 'cancelled') return <Ban className="size-4 text-stone-400" />
  if (status === 'running') return <Loader2 className="size-4 animate-spin text-stone-400" />
  return <Clock3 className="size-4 text-amber-500" />
}

function getAssistantTurnSummary(item: AssistantTurnItem, t: ReturnType<typeof useI18n>['t']): string {
  if (item.stepCount === 0 && item.toolCount === 0) {
    if (item.status === 'running') return t('sessionDetail.turnSummary.runningNoProcess')
    if (item.status === 'failed') return t('sessionDetail.turnSummary.failedNoProcess')
    return t('sessionDetail.turnSummary.completedNoProcess')
  }

  if (item.status === 'running') {
    return item.stepCount > 0
      ? t('sessionDetail.turnSummary.runningSteps', {
        completed: item.completedStepCount,
        total: item.stepCount,
        tools: item.toolCount,
      })
      : t('sessionDetail.turnSummary.runningTools', { tools: item.toolCount })
  }

  if (item.status === 'waiting') {
    return item.stepCount > 0
      ? t('sessionDetail.turnSummary.waitingSteps', {
        completed: item.completedStepCount,
        total: item.stepCount,
      })
      : t('sessionDetail.turnSummary.waiting')
  }

  if (item.status === 'failed') {
    return item.stepCount > 0
      ? t('sessionDetail.turnSummary.failedSteps', {
        completed: item.completedStepCount,
        total: item.stepCount,
      })
      : t('sessionDetail.turnSummary.failedNoProcess')
  }

  if (item.status === 'cancelled') {
    return item.stepCount > 0
      ? t('sessionDetail.turnSummary.cancelledSteps', {
        completed: item.completedStepCount,
        total: item.stepCount,
      })
      : t('sessionDetail.turnSummary.cancelled')
  }

  return item.stepCount > 0
    ? t('sessionDetail.turnSummary.completedSteps', {
      total: item.stepCount,
      tools: item.toolCount,
    })
    : t('sessionDetail.turnSummary.completedTools', { tools: item.toolCount })
}

function AssistantTurnBlock({
  item,
  className,
  onViewAllFiles,
  onFileClick,
  onToolClick,
  locale,
}: {
  item: AssistantTurnItem
  className?: string
  onViewAllFiles?: () => void
  onFileClick?: (file: AttachmentFile) => void
  onToolClick?: (tool: ToolEvent) => void
  locale: AppLocale
}) {
  const { t } = useI18n()
  const hasFinalMessage = Boolean(item.finalMessage)
  const [expanded, setExpanded] = useState(() => getInitialAssistantTurnExpandedState({
    hasFinalMessage,
  }))
  const previousStatusRef = useRef<AssistantTurnItem['status'] | null>(null)
  const hasProcess = item.processItems.length > 0

  useEffect(() => {
    const previousStatus = previousStatusRef.current
    previousStatusRef.current = item.status
    setExpanded((current) => resolveAssistantTurnExpandedState({
      currentExpanded: current,
      previousStatus,
      nextStatus: item.status,
      hasFinalMessage,
    }))
  }, [item.status, hasFinalMessage])

  return (
    <div className={cn('flex flex-col w-full group mt-4', className)}>
      {hasProcess && (
        <div className="w-full">
          <button
            type="button"
            className="flex w-full items-center gap-1 border-b border-stone-200/80 pb-3 text-left text-[17px] leading-6 text-stone-500 transition-colors hover:text-stone-700 cursor-pointer outline-none focus-visible:ring-2 focus-visible:ring-stone-300"
            onClick={() => setExpanded((prev) => !prev)}
          >
            <span className="min-w-0 truncate">
              {getAssistantTurnSummary(item, t)}
            </span>
            <ChevronDown className={cn('size-5 shrink-0 text-stone-400 transition-transform duration-200', expanded ? 'rotate-180' : '-rotate-90')} />
          </button>

          {expanded && (
            <div className="flex flex-col gap-4 py-4">
              {item.processItems.map((processItem) => (
                <ProcessItem
                  key={processItem.id}
                  item={processItem}
                  onToolClick={onToolClick}
                  locale={locale}
                />
              ))}
            </div>
          )}
        </div>
      )}

      {item.finalMessage && (
        <AssistantTurnFinalMessage item={item.finalMessage} className={hasProcess && !expanded ? 'pt-4' : undefined} />
      )}

      {item.attachments.map((attachment) => (
        <AttachmentsMessage
          key={attachment.id}
          role={attachment.role}
          files={attachment.files}
          onViewAllFiles={attachment.role === 'assistant' ? onViewAllFiles : undefined}
          onFileClick={onFileClick}
        />
      ))}
    </div>
  )
}

function AssistantTurnFinalMessage({
  item,
  className,
}: {
  item: Extract<TimelineItem, { kind: 'assistant' }>
  className?: string
}) {
  return (
    <div className={cn('max-w-none p-0 m-0 text-gray-700', className)}>
      <MarkdownContent content={item.data.message ?? ''} />
    </div>
  )
}

function ProcessItem({
  item,
  onToolClick,
  locale,
}: {
  item: AssistantTurnItem['processItems'][number]
  onToolClick?: (tool: ToolEvent) => void
  locale: AppLocale
}) {
  const { t } = useI18n()

  if (item.kind === 'step') {
    return (
      <ProcessStepItem
        stepItem={item}
        onToolClick={onToolClick}
        locale={locale}
      />
    )
  }

  if (item.kind === 'tool') {
    return (
      <ToolRow
        className="mt-0 pl-1"
        timeLabel={item.timeLabel}
        fallbackTimeLabel={t('common.justNow')}
      >
        <ToolUse data={item.data} onClick={onToolClick ? () => onToolClick(item.data) : undefined} />
      </ToolRow>
    )
  }

  if (item.kind === 'assistant') {
    return (
      <div className="max-w-none text-gray-700">
        <MarkdownContent content={item.data.message ?? ''} compact />
      </div>
    )
  }

  if (item.kind === 'error') {
    return (
      <div className="text-sm leading-relaxed text-red-600">
        {item.error}
      </div>
    )
  }

  return null
}

function ProcessStepItem({
  stepItem,
  onToolClick,
  locale,
}: {
  stepItem: Extract<TimelineItem, { kind: 'step' }>
  onToolClick?: (tool: ToolEvent) => void
  locale: AppLocale
}) {
  const { t } = useI18n()
  const { data, tools } = stepItem
  const detail = resolveStepDetail(data.outcome)

  return (
    <div className="flex flex-col gap-2">
      <div className="flex min-w-0 items-center gap-2 text-stone-400">
        <span className="shrink-0">
          {getProcessStatusIcon(data.status)}
        </span>
        <span className="min-w-0 truncate text-[15px] leading-6">
          {data.description}
        </span>
      </div>
      {detail && (
        <div className="pl-6 text-sm leading-relaxed text-stone-500">
          {detail}
        </div>
      )}
      {tools.length > 0 && (
        <div className="flex flex-col gap-2 pl-6">
          {tools.map((tool, idx) => (
            <ToolRow
              key={`${data.id}-process-tool-${idx}`}
              className="mt-0"
              timeLabel={getToolTimeLabel(tool, locale)}
              fallbackTimeLabel={t('common.justNow')}
            >
              <ToolUse data={tool} onClick={onToolClick ? () => onToolClick(tool) : undefined} />
            </ToolRow>
          ))}
        </div>
      )}
    </div>
  )
}

function StepBlock({
  stepItem,
  className,
  onToolClick,
  locale,
}: {
  stepItem: Extract<TimelineItem, { kind: 'step' }>
  className?: string
  onToolClick?: (tool: ToolEvent) => void
  locale: AppLocale
}) {
  const [expanded, setExpanded] = useState(() => shouldAutoExpandStep(stepItem.data.status))
  const { t } = useI18n()
  const { data, tools } = stepItem
  const detail = resolveStepDetail(data.outcome)
  const shouldShowBody = Boolean(detail) || tools.length > 0
  const previousStatusRef = useRef<StepEvent['status'] | null>(null)

  useEffect(() => {
    setExpanded((currentExpanded) => resolveStepExpandedState({
      currentExpanded,
      previousStatus: previousStatusRef.current,
      nextStatus: data.status,
    }))
    previousStatusRef.current = data.status
  }, [data.status])

  return (
    <div className={cn('flex flex-col mt-3', className)}>
      <div
        role="button"
        tabIndex={0}
        onClick={() => setExpanded(!expanded)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault()
            setExpanded((prev) => !prev)
          }
        }}
        className="text-sm w-full cursor-pointer flex gap-2 justify-between group/header truncate text-gray-700 rounded-md hover:bg-gray-50/80 transition-colors outline-none focus-visible:ring-2 focus-visible:ring-gray-300"
      >
        <div className="flex flex-row gap-2 justify-start items-center truncate min-w-0 flex-1">
          <div className="flex-shrink-0">
            {getStepStatusIcon(data.status)}
          </div>
          <div className="min-w-0 flex-1">
            <div className="truncate font-medium markdown-content">
              {data.description}
            </div>
            {detail && (
              <div className="mt-0.5 line-clamp-2 text-xs text-gray-500">
                {detail}
              </div>
            )}
          </div>
          <ChevronDown
            className={cn('flex-shrink-0 transition-transform text-gray-500', expanded && 'rotate-180')}
          />
        </div>
      </div>
      {expanded && shouldShowBody && (
        <div className="flex">
          <div className="w-6 relative flex-shrink-0">
            <div className="absolute left-[7px] top-2 bottom-0 w-[1px] border-l border-dashed border-gray-300" />
          </div>
          <div className="flex flex-col gap-3 flex-1 min-w-0 overflow-hidden pt-2 transition-[max-height,opacity] duration-150 ease-in-out">
            {detail && (
              <div className="rounded-md border border-gray-200 bg-gray-50 px-3 py-2 text-xs text-gray-600">
                {detail}
              </div>
            )}
            {tools.map((tool, idx) => (
              <ToolRow
                key={`${data.id}-tool-${idx}`}
                timeLabel={getToolTimeLabel(tool, locale)}
                fallbackTimeLabel={t('common.justNow')}
              >
                <ToolUse data={tool} onClick={onToolClick ? () => onToolClick(tool) : undefined} />
              </ToolRow>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
