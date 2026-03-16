import { registerAuthHooks } from "@/lib/api/fetch"
import { authApi } from "./api"
import {
  clearAuthenticatedSession,
  getAuthSnapshot,
  hydrateAuthStoreFromStorage,
  setCurrentUser,
  setAuthenticatedSession,
} from "./store"

let hooksRegistered = false
let initialized = false
let initializePromise: Promise<void> | null = null
let refreshPromise: Promise<string | null> | null = null
let loadUserPromise: Promise<void> | null = null

function redirectToLogin(): void {
  if (typeof window === "undefined") return

  if (window.location.pathname.startsWith("/auth/")) {
    return
  }

  const redirect = `${window.location.pathname}${window.location.search}`
  const nextUrl = `/?auth=login&redirect=${encodeURIComponent(redirect)}`
  window.location.assign(nextUrl)
}

function ensureHooksRegistered(): void {
  if (hooksRegistered) return

  registerAuthHooks({
    getAccessToken: () => getAuthSnapshot().accessToken,
    refreshAccessToken,
    onAuthFailure: () => {
      clearAuthenticatedSession()
      redirectToLogin()
    },
  })

  hooksRegistered = true
}

async function loadCurrentUserIfNeeded(force = false): Promise<void> {
  const snapshot = getAuthSnapshot()
  if (!snapshot.accessToken) {
    return
  }
  if (!force && snapshot.user) {
    return
  }
  if (loadUserPromise) {
    return loadUserPromise
  }

  loadUserPromise = authApi
    .me()
    .then((user) => {
      setCurrentUser(user)
    })
    .catch(() => {
      // 获取用户信息失败时不阻断主流程，401 会由请求层统一清理登录态。
    })
    .finally(() => {
      loadUserPromise = null
    })

  return loadUserPromise
}

export async function refreshAccessToken(): Promise<string | null> {
  hydrateAuthStoreFromStorage()

  const refreshToken = getAuthSnapshot().refreshToken
  if (!refreshToken) {
    return null
  }

  if (refreshPromise) {
    return refreshPromise
  }

  refreshPromise = authApi
    .refresh(refreshToken)
    .then((result) => {
      setAuthenticatedSession({ tokens: result.tokens })
      void loadCurrentUserIfNeeded()
      return result.tokens.access_token
    })
    .catch(() => {
      clearAuthenticatedSession()
      return null
    })
    .finally(() => {
      refreshPromise = null
    })

  return refreshPromise
}

export async function initializeAuth(): Promise<void> {
  if (typeof window === "undefined") {
    return
  }

  ensureHooksRegistered()

  if (initialized) {
    return
  }

  if (initializePromise) {
    return initializePromise
  }

  initializePromise = (async () => {
    hydrateAuthStoreFromStorage()

    const { accessToken, refreshToken } = getAuthSnapshot()
    if (!accessToken && refreshToken) {
      await refreshAccessToken()
    }

    await loadCurrentUserIfNeeded()

    initialized = true
  })().finally(() => {
    initializePromise = null
  })

  return initializePromise
}

export async function logoutFromServer(): Promise<void> {
  const refreshToken = getAuthSnapshot().refreshToken
  if (!refreshToken) {
    return
  }

  try {
    await authApi.logout(refreshToken)
  } catch {
    // 退出链路采用尽力而为策略，本地仍会清理登录态。
  }
}
