import type { AuthTokenPair, AuthUser } from "./types"

export type AuthSnapshot = {
  user: AuthUser | null
  accessToken: string | null
  hydrated: boolean
}

const AUTH_SESSION_HINT_STORAGE_KEY = "auth_has_refresh_session"

const INITIAL_SNAPSHOT: AuthSnapshot = {
  user: null,
  accessToken: null,
  hydrated: false,
}

let snapshot: AuthSnapshot = { ...INITIAL_SNAPSHOT }
let hasHydrated = false

const listeners = new Set<() => void>()

function getLocalStorageSafe(): Storage | null {
  if (typeof window === "undefined") {
    return null
  }

  try {
    return window.localStorage ?? null
  } catch {
    return null
  }
}

function setAuthSessionHint(enabled: boolean): void {
  const localStorage = getLocalStorageSafe()
  if (!localStorage) {
    return
  }

  try {
    if (enabled) {
      localStorage.setItem(AUTH_SESSION_HINT_STORAGE_KEY, "1")
      return
    }
    localStorage.removeItem(AUTH_SESSION_HINT_STORAGE_KEY)
  } catch {
    // 忽略浏览器存储异常，避免影响认证主流程。
  }
}

function emitChange(): void {
  listeners.forEach((listener) => listener())
}

function setSnapshot(next: AuthSnapshot): void {
  snapshot = next
  emitChange()
}

export function subscribeAuthStore(listener: () => void): () => void {
  listeners.add(listener)
  return () => {
    listeners.delete(listener)
  }
}

export function getAuthSnapshot(): AuthSnapshot {
  return snapshot
}

export function isAuthenticatedSnapshot(
  value: Pick<AuthSnapshot, "accessToken">,
): boolean {
  return Boolean(value.accessToken)
}

export function hydrateAuthStoreFromStorage(): void {
  if (typeof window === "undefined") {
    return
  }

  if (hasHydrated) {
    return
  }

  hasHydrated = true
  setSnapshot({
    ...snapshot,
    hydrated: true,
  })
}

export function hasAuthSessionHint(): boolean {
  const localStorage = getLocalStorageSafe()
  if (!localStorage) {
    return false
  }

  try {
    return localStorage.getItem(AUTH_SESSION_HINT_STORAGE_KEY) === "1"
  } catch {
    return false
  }
}

export function setAuthenticatedSession(params: {
  tokens: AuthTokenPair
  user?: AuthUser | null
}): void {
  const nextUser = params.user === undefined ? snapshot.user : params.user

  setAuthSessionHint(true)

  setSnapshot({
    ...snapshot,
    user: nextUser,
    accessToken: params.tokens.access_token,
    hydrated: true,
  })
}

export function setCurrentUser(user: AuthUser | null): void {
  setSnapshot({
    ...snapshot,
    user,
    hydrated: true,
  })
}

export function clearAuthenticatedSession(): void {
  setAuthSessionHint(false)

  setSnapshot({
    ...snapshot,
    user: null,
    accessToken: null,
    hydrated: true,
  })
}

export function __resetAuthStoreForTest(): void {
  setAuthSessionHint(false)
  snapshot = { ...INITIAL_SNAPSHOT }
  hasHydrated = false
  listeners.clear()
}
