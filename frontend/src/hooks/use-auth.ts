"use client"

import { useCallback, useEffect, useSyncExternalStore } from "react"
import {
  authApi,
  clearAuthenticatedSession,
  getAuthSnapshot,
  initializeAuth,
  isAuthenticatedSnapshot,
  logoutFromServer,
  setAuthenticatedSession,
  subscribeAuthStore,
} from "@/lib/auth"
import type {
  AuthUser,
  LoginRequestPayload,
  RegisterRequestPayload,
  RegisterResponseData,
} from "@/lib/auth"

function redirectToLoginPage(): void {
  if (typeof window === "undefined") return

  if (window.location.pathname.startsWith("/auth/")) {
    return
  }

  window.location.assign("/?auth=login")
}

export type UseAuthResult = {
  user: AuthUser | null
  isLoggedIn: boolean
  isHydrated: boolean
  login: (payload: LoginRequestPayload) => Promise<AuthUser>
  logout: () => Promise<void>
  register: (payload: RegisterRequestPayload) => Promise<RegisterResponseData>
}

export function useAuth(): UseAuthResult {
  const snapshot = useSyncExternalStore(
    subscribeAuthStore,
    getAuthSnapshot,
    getAuthSnapshot,
  )

  useEffect(() => {
    void initializeAuth()
  }, [])

  const login = useCallback(async (payload: LoginRequestPayload): Promise<AuthUser> => {
    const result = await authApi.login(payload)
    setAuthenticatedSession({
      tokens: result.tokens,
      user: result.user,
    })

    return result.user
  }, [])

  const logout = useCallback(async (): Promise<void> => {
    await logoutFromServer()
    clearAuthenticatedSession()
    redirectToLoginPage()
  }, [])

  const register = useCallback(
    async (payload: RegisterRequestPayload): Promise<RegisterResponseData> => {
      return authApi.register(payload)
    },
    [],
  )

  const isLoggedIn = isAuthenticatedSnapshot(snapshot)

  return {
    user: snapshot.user,
    isLoggedIn,
    isHydrated: snapshot.hydrated,
    login,
    logout,
    register,
  }
}
