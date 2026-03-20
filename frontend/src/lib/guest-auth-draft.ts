export type GuestPendingActionType = 'send' | 'upload' | 'manual_login'

const PENDING_MESSAGE_KEY = 'actus.guest.pending_message'
const PENDING_ACTION_KEY = 'actus.guest.pending_action'

function canUseSessionStorage(): boolean {
  return typeof window !== 'undefined' && typeof window.sessionStorage !== 'undefined'
}

export function saveGuestPendingMessage(message: string): void {
  if (!canUseSessionStorage()) return

  try {
    window.sessionStorage.setItem(PENDING_MESSAGE_KEY, message)
  } catch {
    // ignore storage errors
  }
}

export function loadGuestPendingMessage(): string {
  if (!canUseSessionStorage()) return ''

  try {
    return window.sessionStorage.getItem(PENDING_MESSAGE_KEY) || ''
  } catch {
    return ''
  }
}

export function clearGuestPendingMessage(): void {
  if (!canUseSessionStorage()) return

  try {
    window.sessionStorage.removeItem(PENDING_MESSAGE_KEY)
  } catch {
    // ignore storage errors
  }
}

export function saveGuestPendingAction(action: GuestPendingActionType): void {
  if (!canUseSessionStorage()) return

  try {
    window.sessionStorage.setItem(PENDING_ACTION_KEY, action)
  } catch {
    // ignore storage errors
  }
}

export function loadGuestPendingAction(): GuestPendingActionType | null {
  if (!canUseSessionStorage()) return null

  try {
    const action = window.sessionStorage.getItem(PENDING_ACTION_KEY)
    if (action === 'send' || action === 'upload' || action === 'manual_login') {
      return action
    }
    return null
  } catch {
    return null
  }
}

export function clearGuestPendingAction(): void {
  if (!canUseSessionStorage()) return

  try {
    window.sessionStorage.removeItem(PENDING_ACTION_KEY)
  } catch {
    // ignore storage errors
  }
}
