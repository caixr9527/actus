export type InitialMessageDraft = {
  message: string
  attachments: string[]
}

type StoredInitialMessageDraft = InitialMessageDraft & {
  createdAt: number
}

const INITIAL_MESSAGE_DRAFT_KEY_PREFIX = 'session-init-draft:'
const INITIAL_MESSAGE_DRAFT_TTL_MS = 15 * 60 * 1000

function getDraftKey(sessionId: string): string {
  return `${INITIAL_MESSAGE_DRAFT_KEY_PREFIX}${sessionId}`
}

function normalizeDraftPayload(payload: unknown): InitialMessageDraft | null {
  if (!payload || typeof payload !== 'object') return null
  const data = payload as { message?: unknown; attachments?: unknown }
  if (typeof data.message !== 'string' || data.message.trim() === '') return null
  const attachments = Array.isArray(data.attachments)
    ? data.attachments.filter((item): item is string => typeof item === 'string' && item.length > 0)
    : []

  return {
    message: data.message,
    attachments,
  }
}

export function saveInitialMessageDraft(
  sessionId: string,
  message: string,
  attachments: string[]
): boolean {
  if (typeof window === 'undefined') return false
  const key = getDraftKey(sessionId)
  const payload: StoredInitialMessageDraft = {
    message,
    attachments,
    createdAt: Date.now(),
  }

  try {
    window.sessionStorage.setItem(key, JSON.stringify(payload))
    return true
  } catch (error) {
    console.warn('[initial-message-draft] save failed:', error)
    return false
  }
}

export function consumeInitialMessageDraft(sessionId: string): InitialMessageDraft | null {
  if (typeof window === 'undefined') return null
  const key = getDraftKey(sessionId)

  let raw: string | null = null
  try {
    raw = window.sessionStorage.getItem(key)
  } catch (error) {
    console.warn('[initial-message-draft] read failed:', error)
    return null
  }
  if (!raw) return null

  try {
    const parsed = JSON.parse(raw) as { createdAt?: unknown; message?: unknown; attachments?: unknown }
    const createdAt = typeof parsed.createdAt === 'number' ? parsed.createdAt : 0
    if (Date.now() - createdAt > INITIAL_MESSAGE_DRAFT_TTL_MS) {
      return null
    }
    return normalizeDraftPayload(parsed)
  } catch (error) {
    console.warn('[initial-message-draft] parse failed:', error)
    return null
  } finally {
    try {
      window.sessionStorage.removeItem(key)
    } catch (error) {
      console.warn('[initial-message-draft] cleanup failed:', error)
    }
  }
}

export function parseLegacyInitQueryParam(initParam: string | null): InitialMessageDraft | null {
  if (!initParam) return null

  try {
    const decoded = decodeURIComponent(window.atob(initParam))
    const parsed = JSON.parse(decoded)
    return normalizeDraftPayload(parsed)
  } catch (error) {
    console.warn('[initial-message-draft] legacy query parse failed:', error)
    return null
  }
}
