'use client'

import type {
  WaitChoice,
  WaitConfirmPayload,
  WaitEventData,
  WaitInputTextPayload,
  WaitPayload,
  WaitSelectPayload,
  WaitUserTakeover,
} from './api/types'

export type WaitEventContext = {
  interruptId: string | null
  payload: WaitPayload | null
  title: string | null
  prompt: string | null
  details: string | null
  attachments: string[]
  suggestUserTakeover: boolean
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function toText(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function normalizeAttachments(value: unknown): string[] {
  if (typeof value === 'string') {
    const attachment = value.trim()
    return attachment ? [attachment] : []
  }
  if (!Array.isArray(value)) return []
  return value
    .map((item) => String(item ?? '').trim())
    .filter((item) => item.length > 0)
}

function normalizeWaitChoice(raw: unknown): WaitChoice | null {
  if (!isRecord(raw)) return null
  const label = toText(raw.label)
  if (!label) return null
  return {
    label,
    resume_value: raw.resume_value,
    description: toText(raw.description),
  }
}

function normalizeBasePayload(raw: Record<string, unknown>) {
  const prompt = toText(raw.prompt)
  if (!prompt) return null
  const takeover: WaitUserTakeover = raw.suggest_user_takeover === 'browser' ? 'browser' : 'none'
  return {
    title: toText(raw.title),
    prompt,
    details: toText(raw.details),
    attachments: normalizeAttachments(raw.attachments),
    suggest_user_takeover: takeover,
  }
}

function normalizeInputTextPayload(raw: Record<string, unknown>): WaitInputTextPayload | null {
  const base = normalizeBasePayload(raw)
  if (!base) return null
  return {
    kind: 'input_text',
    ...base,
    placeholder: toText(raw.placeholder),
    submit_label: toText(raw.submit_label) || '继续执行',
    response_key: toText(raw.response_key) || 'message',
    default_value: toText(raw.default_value),
    multiline: raw.multiline !== false,
    allow_empty: raw.allow_empty === true,
  }
}

function normalizeConfirmPayload(raw: Record<string, unknown>): WaitConfirmPayload | null {
  const base = normalizeBasePayload(raw)
  if (!base) return null
  return {
    kind: 'confirm',
    ...base,
    confirm_label: toText(raw.confirm_label) || '继续',
    cancel_label: toText(raw.cancel_label) || '取消',
    confirm_resume_value: raw.confirm_resume_value ?? true,
    cancel_resume_value: raw.cancel_resume_value ?? false,
    emphasis: raw.emphasis === 'destructive' ? 'destructive' : 'default',
  }
}

function normalizeSelectPayload(raw: Record<string, unknown>): WaitSelectPayload | null {
  const base = normalizeBasePayload(raw)
  if (!base) return null
  const options = Array.isArray(raw.options)
    ? raw.options.map(normalizeWaitChoice).filter((option): option is WaitChoice => option !== null)
    : []
  if (options.length === 0) return null
  return {
    kind: 'select',
    ...base,
    options,
    default_resume_value: raw.default_resume_value,
  }
}

export function normalizeWaitPayload(raw: unknown): WaitPayload | null {
  if (!isRecord(raw)) return null
  const kind = toText(raw.kind)
  if (kind === 'confirm') {
    return normalizeConfirmPayload(raw)
  }
  if (kind === 'select') {
    return normalizeSelectPayload(raw)
  }
  if (kind === 'input_text') {
    return normalizeInputTextPayload(raw)
  }
  return null
}

export function parseWaitEventContext(data: WaitEventData): WaitEventContext | null {
  const payload = normalizeWaitPayload(data.payload)
  if (!payload) return null
  return {
    interruptId: toText(data.interrupt_id) || null,
    payload,
    title: payload.title || null,
    prompt: payload.prompt || null,
    details: payload.details || null,
    attachments: payload.attachments ?? [],
    suggestUserTakeover: payload.suggest_user_takeover === 'browser',
  }
}

export function buildResumeValueFromWaitPayload(payload: WaitPayload, inputValue?: string): unknown {
  if (payload.kind === 'input_text') {
    const responseKey = payload.response_key || 'message'
    return {
      [responseKey]: inputValue ?? payload.default_value ?? '',
    }
  }
  return null
}
