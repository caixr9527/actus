'use client'

import { useEffect, useState } from 'react'
import { MonitorSmartphone, Paperclip } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { cn } from '@/lib/utils'
import { buildResumeValueFromWaitPayload, type WaitEventContext } from '@/lib/wait-event'
import { useI18n } from '@/lib/i18n'

export interface WaitResumeCardProps {
  className?: string
  waitContext: WaitEventContext
  busy?: boolean
  onResume: (resumeValue: unknown) => Promise<void> | void
  onOpenTakeover?: () => void
}

export function WaitResumeCard({
  className,
  waitContext,
  busy = false,
  onResume,
  onOpenTakeover,
}: WaitResumeCardProps) {
  const { t } = useI18n()
  const payload = waitContext.payload
  const [inputValue, setInputValue] = useState(
    payload?.kind === 'input_text' ? (payload.default_value ?? '') : '',
  )

  useEffect(() => {
    if (payload?.kind !== 'input_text') {
      setInputValue('')
      return
    }
    setInputValue(payload.default_value ?? '')
  }, [payload, waitContext.interruptId])

  if (!payload) return null

  const handleSubmitText = async () => {
    if (payload.kind !== 'input_text') return
    const nextValue = inputValue.trim()
    if (!payload.allow_empty && nextValue.length === 0) {
      return
    }
    try {
      await onResume(buildResumeValueFromWaitPayload(payload, nextValue))
    } catch {
      return
    }
  }

  const handleResumeAction = (resumeValue: unknown) => {
    void Promise.resolve(onResume(resumeValue)).catch(() => undefined)
  }

  return (
    <div className={cn('rounded-xl border border-amber-200 bg-amber-50 px-3 py-3', className)}>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-xs font-semibold text-amber-900">
            {payload.title || t('sessionDetail.waitCardTitle')}
          </p>
          <p className="mt-1 text-sm text-amber-950">{payload.prompt}</p>
          {payload.details ? (
            <p className="mt-1 text-xs leading-5 text-amber-800">{payload.details}</p>
          ) : null}
        </div>
        {waitContext.suggestUserTakeover && onOpenTakeover ? (
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="shrink-0 border-amber-300 bg-white/80 text-amber-900 hover:bg-white"
            onClick={onOpenTakeover}
            disabled={busy}
          >
            <MonitorSmartphone className="size-4" />
            <span>{t('sessionDetail.waitCardTakeoverAction')}</span>
          </Button>
        ) : null}
      </div>

      {waitContext.attachments.length > 0 ? (
        <div className="mt-3">
          <p className="text-[11px] font-medium uppercase tracking-[0.08em] text-amber-700">
            {t('sessionDetail.waitCardAttachmentsTitle')}
          </p>
          <div className="mt-1.5 flex flex-wrap gap-2">
            {waitContext.attachments.map((attachment) => (
              <div
                key={attachment}
                className="inline-flex max-w-full items-center gap-1 rounded-full border border-amber-200 bg-white/80 px-2 py-1 text-[11px] text-amber-900"
              >
                <Paperclip className="size-3" />
                <span className="truncate">{attachment}</span>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {waitContext.suggestUserTakeover ? (
        <p className="mt-2 text-[11px] text-amber-800">{t('sessionDetail.waitCardTakeoverHint')}</p>
      ) : null}

      {payload.kind === 'input_text' ? (
        <div className="mt-3 space-y-2">
          <Textarea
            value={inputValue}
            onChange={(event) => setInputValue(event.target.value)}
            placeholder={payload.placeholder || t('sessionDetail.waitCardInputPlaceholder')}
            rows={payload.multiline === false ? 2 : 4}
            disabled={busy}
            className="border-amber-200 bg-white/90 text-sm text-gray-900"
          />
          <div className="flex justify-end">
            <Button
              type="button"
              size="sm"
              onClick={() => {
                void handleSubmitText()
              }}
              disabled={busy || (!payload.allow_empty && inputValue.trim().length === 0)}
            >
              {payload.submit_label || t('sessionDetail.waitCardSubmit')}
            </Button>
          </div>
        </div>
      ) : null}

      {payload.kind === 'confirm' ? (
        <div className="mt-3 flex flex-wrap gap-2">
          <Button
            type="button"
            size="sm"
            variant={payload.emphasis === 'destructive' ? 'destructive' : 'default'}
            disabled={busy}
            onClick={() => {
              handleResumeAction(payload.confirm_resume_value ?? true)
            }}
          >
            {payload.confirm_label || t('sessionDetail.waitCardConfirm')}
          </Button>
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={busy}
            onClick={() => {
              handleResumeAction(payload.cancel_resume_value ?? false)
            }}
          >
            {payload.cancel_label || t('sessionDetail.waitCardCancel')}
          </Button>
        </div>
      ) : null}

      {payload.kind === 'select' ? (
        <div className="mt-3 flex flex-col gap-2">
          {payload.options.map((option, index) => (
            <button
              key={`${option.label}-${index}`}
              type="button"
              className="rounded-xl border border-amber-200 bg-white/90 px-3 py-2 text-left transition-colors hover:border-amber-300 hover:bg-white disabled:cursor-not-allowed disabled:opacity-50"
              disabled={busy}
              onClick={() => {
                handleResumeAction(option.resume_value)
              }}
            >
              <div className="text-sm font-medium text-gray-900">{option.label}</div>
              {option.description ? (
                <div className="mt-1 text-xs leading-5 text-gray-600">{option.description}</div>
              ) : null}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  )
}
