import type { AuthTokenPair, AuthUser } from "./types"

const REFRESH_TOKEN_STORAGE_KEY = "actus.auth.refresh_token"

export type AuthSnapshot = {
  user: AuthUser | null
  accessToken: string | null
  refreshToken: string | null
  hydrated: boolean
}

const INITIAL_SNAPSHOT: AuthSnapshot = {
  user: null,
  accessToken: null,
  refreshToken: null,
  hydrated: false,
}

let snapshot: AuthSnapshot = { ...INITIAL_SNAPSHOT }
let hasHydratedFromStorage = false

const listeners = new Set<() => void>()

function emitChange(): void {
  listeners.forEach((listener) => listener())
}

function setSnapshot(next: AuthSnapshot): void {
  snapshot = next
  emitChange()
}

function readRefreshTokenFromStorage(): string | null {
  if (typeof window === "undefined") return null

  try {
    return window.localStorage.getItem(REFRESH_TOKEN_STORAGE_KEY)
  } catch {
    return null
  }
}

function writeRefreshTokenToStorage(refreshToken: string | null): void {
  if (typeof window === "undefined") return

  try {
    if (refreshToken) {
      window.localStorage.setItem(REFRESH_TOKEN_STORAGE_KEY, refreshToken)
    } else {
      window.localStorage.removeItem(REFRESH_TOKEN_STORAGE_KEY)
    }
  } catch {
    // 忽略存储异常（例如隐私模式禁用 localStorage）
  }
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

  if (hasHydratedFromStorage) {
    return
  }

  hasHydratedFromStorage = true
  setSnapshot({
    ...snapshot,
    refreshToken: readRefreshTokenFromStorage(),
    hydrated: true,
  })
}

export function setAuthenticatedSession(params: {
  tokens: AuthTokenPair
  user?: AuthUser | null
}): void {
  const nextUser = params.user === undefined ? snapshot.user : params.user

  setSnapshot({
    ...snapshot,
    user: nextUser,
    accessToken: params.tokens.access_token,
    refreshToken: params.tokens.refresh_token,
    hydrated: true,
  })

  writeRefreshTokenToStorage(params.tokens.refresh_token)
}

export function setCurrentUser(user: AuthUser | null): void {
  setSnapshot({
    ...snapshot,
    user,
    hydrated: true,
  })
}

export function clearAuthenticatedSession(): void {
  setSnapshot({
    ...snapshot,
    user: null,
    accessToken: null,
    refreshToken: null,
    hydrated: true,
  })

  writeRefreshTokenToStorage(null)
}

export function __resetAuthStoreForTest(): void {
  snapshot = { ...INITIAL_SNAPSHOT }
  hasHydratedFromStorage = false
  listeners.clear()
}
