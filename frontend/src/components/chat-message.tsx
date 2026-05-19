'use client'

import { useEffect, useRef, useState } from 'react'
import { cn } from '@/lib/utils'
import { Ban, CheckCircle2, ChevronDown, Clock3, Languages, Loader2, PencilLine, ThumbsDown, ThumbsUp, XCircle } from 'lucide-react'
import { ManusIcon } from '@/components/manus-icon'
import { ToolUse } from '@/components/tool-use'
import { AttachmentsMessage } from '@/components/attachments-message'
import { MarkdownContent } from '@/components/markdown-content'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/lib/i18n'
import { getUserMessageTextClassName, normalizeUserMessageText } from '@/lib/user-message-text'
import type { AppLocale } from '@/lib/i18n'
import type { StepEvent, SubmitFeedbackParams, ToolEvent } from '@/lib/api/types'
import { type TimelineItem, type AttachmentFile, getToolTimeLabel } from '@/lib/session-events'
import { resolveStepDetail } from '@/lib/run-timeline'
import { resolveStepProcessSectionOrder } from '@/lib/step-process-display'
import { resolveAssistantTurnSummary } from '@/lib/assistant-turn-summary'
import { buildFinalMessageFeedbackTarget } from '@/lib/feedback-targets'
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
  onSubmitFeedback?: (params: SubmitFeedbackParams) => Promise<void>
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
  onSubmitFeedback,
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
          <UserMessageText message={item.data.message} />
        </div>
      </div>
    )
  }

  if (item.kind === 'assistant') {
    return (
      <AssistantMessageBlock
        item={item}
        className={className}
        onSubmitFeedback={onSubmitFeedback}
      />
    )
  }

  if (item.kind === 'assistant_turn') {
    return (
        <AssistantTurnBlock
          item={item}
          className={className}
          onSubmitFeedback={onSubmitFeedback}
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

function UserMessageText({ message }: { message?: string | null }) {
  return (
    <div
      className={cn(
        'text-gray-700 relative rounded-lg overflow-hidden bg-white p-3 border',
        getUserMessageTextClassName(),
      )}
    >
      {normalizeUserMessageText(message)}
    </div>
  )
}

function AssistantMessageBlock({
  item,
  className,
  onSubmitFeedback,
}: {
  item: Extract<TimelineItem, { kind: 'assistant' }>
  className?: string
  onSubmitFeedback?: (params: SubmitFeedbackParams) => Promise<void>
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
      <AssistantFinalMessageContent item={item} />
      <AssistantFinalFeedbackActions item={item} onSubmitFeedback={onSubmitFeedback} />
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
  return resolveAssistantTurnSummary(item.status, t)
}

function AssistantTurnBlock({
  item,
  className,
  onSubmitFeedback,
  onViewAllFiles,
  onFileClick,
  onToolClick,
  locale,
}: {
  item: AssistantTurnItem
  className?: string
  onSubmitFeedback?: (params: SubmitFeedbackParams) => Promise<void>
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
        <AssistantFinalMessageContent
          item={item.finalMessage}
          className={hasProcess && !expanded ? 'pt-4' : undefined}
        />
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

      {item.finalMessage && (
        <AssistantFinalFeedbackActions
          item={item.finalMessage}
          onSubmitFeedback={onSubmitFeedback}
        />
      )}
    </div>
  )
}

function AssistantFinalMessageContent({
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

function AssistantFinalFeedbackActions({
  item,
  onSubmitFeedback,
}: {
  item: Extract<TimelineItem, { kind: 'assistant' }>
  onSubmitFeedback?: (params: SubmitFeedbackParams) => Promise<void>
}) {
  const { t } = useI18n()
  const [pendingAction, setPendingAction] = useState<string | null>(null)
  const finalTarget = buildFinalMessageFeedbackTarget(item.data)

  const submitFinalFeedback = async (
    intentKind: SubmitFeedbackParams['intent_kind'],
    reasonCode: SubmitFeedbackParams['reason_code'],
  ) => {
    if (!onSubmitFeedback || !finalTarget) return
    const actionKey = `final:${intentKind}`
    setPendingAction(actionKey)
    try {
      await onSubmitFeedback({
        source_action: 'final_satisfaction',
        intent_kind: intentKind,
        target_ref: finalTarget,
        reason_code: reasonCode,
      })
    } finally {
      setPendingAction(null)
    }
  }

  if (!onSubmitFeedback || !finalTarget) return null

  return (
    <div
      className="mt-2 flex items-center gap-3"
      aria-label="final message feedback"
    >
      <FeedbackIconButton
        icon={<ThumbsUp className="size-3.5" />}
        label={t('feedback.finalSatisfied')}
        disabled={pendingAction !== null}
        onClick={() => void submitFinalFeedback('satisfaction', 'user_reported_satisfaction')}
      />
      <FeedbackIconButton
        icon={<ThumbsDown className="size-3.5" />}
        label={t('feedback.finalDissatisfied')}
        disabled={pendingAction !== null}
        onClick={() => void submitFinalFeedback('dissatisfaction', 'user_reported_dissatisfaction')}
      />
      <FeedbackIconButton
        icon={<PencilLine className="size-3.5" />}
        label={t('feedback.finalCorrection')}
        disabled={pendingAction !== null}
        onClick={() => void submitFinalFeedback('correction', 'user_corrected_requirement')}
      />
    </div>
  )
}

function FeedbackIconButton({
  icon,
  label,
  disabled,
  onClick,
}: {
  icon: React.ReactNode
  label: string
  disabled?: boolean
  onClick: () => void
}) {
  return (
    <Button
      type="button"
      variant="ghost"
      size="icon-sm"
      title={label}
      aria-label={label}
      disabled={disabled}
      onClick={onClick}
      className={cn(
        'group/feedback relative h-7 w-7 rounded-none bg-transparent p-0 text-stone-400 shadow-none',
        'hover:bg-transparent hover:text-stone-700 focus-visible:bg-transparent',
      )}
    >
      {icon}
    </Button>
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
  const detail = resolveStepDetail(data.outcome, data.status)
  const sectionOrder = resolveStepProcessSectionOrder({
    hasTools: tools.length > 0,
    hasSummary: Boolean(detail),
  })

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
      {sectionOrder.map((section) => {
        if (section === 'tools') {
          return (
            <div key="tools" className="flex flex-col gap-2 pl-6">
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
          )
        }

        if (!detail) return null
        return (
          <div key="summary" className="pl-6 text-sm leading-relaxed text-stone-500">
            {detail}
          </div>
        )
      })}
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
  const detail = resolveStepDetail(data.outcome, data.status)
  const shouldShowBody = Boolean(detail) || tools.length > 0
  const sectionOrder = resolveStepProcessSectionOrder({
    hasTools: tools.length > 0,
    hasSummary: Boolean(detail),
  })
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
            {sectionOrder.map((section) => {
              if (section === 'tools') {
                return tools.map((tool, idx) => (
                  <ToolRow
                    key={`${data.id}-tool-${idx}`}
                    timeLabel={getToolTimeLabel(tool, locale)}
                    fallbackTimeLabel={t('common.justNow')}
                  >
                    <ToolUse data={tool} onClick={onToolClick ? () => onToolClick(tool) : undefined} />
                  </ToolRow>
                ))
              }

              if (!detail) return null
              return (
                <div key="summary" className="rounded-md border border-gray-200 bg-gray-50 px-3 py-2 text-xs text-gray-600">
                  {detail}
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
