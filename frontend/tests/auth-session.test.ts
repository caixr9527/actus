import assert from "node:assert/strict"
import test from "node:test"

import {
  __resetAuthSessionForTest,
  __resetAuthStoreForTest,
  clearAuthenticatedSession,
  getAuthSnapshot,
  hasAuthSessionHint,
  initializeAuth,
  setAuthenticatedSession,
} from "../src/lib/auth"
import { authApi } from "../src/lib/auth/api"
import type { AuthTokenPair, AuthUser } from "../src/lib/auth/types"

class MemoryStorage implements Storage {
  private readonly store = new Map<string, string>()

  get length(): number {
    return this.store.size
  }

  clear(): void {
    this.store.clear()
  }

  getItem(key: string): string | null {
    return this.store.get(key) ?? null
  }

  key(index: number): string | null {
    return Array.from(this.store.keys())[index] ?? null
  }

  removeItem(key: string): void {
    this.store.delete(key)
  }

  setItem(key: string, value: string): void {
    this.store.set(key, value)
  }
}

function installWindow(): void {
  ;(globalThis as { window?: unknown }).window = {
    location: {
      pathname: "/",
      search: "",
      assign: () => undefined,
    },
    localStorage: new MemoryStorage(),
  }
}

const demoTokens: AuthTokenPair = {
  access_token: "access-token",
  token_type: "Bearer",
  access_token_expires_in: 1800,
}

const demoUser: AuthUser = {
  user_id: "user-1",
  email: "tester@example.com",
  nickname: "tester",
  avatar_url: null,
  timezone: "Asia/Shanghai",
  locale: "zh-CN",
  auth_provider: "email",
  status: "active",
  created_at: "2026-03-15T00:00:00Z",
  updated_at: "2026-03-15T00:00:00Z",
  last_login_at: null,
  last_login_ip: null,
}

const originalRefresh = authApi.refresh
const originalMe = authApi.me

test.afterEach(() => {
  authApi.refresh = originalRefresh
  authApi.me = originalMe
  __resetAuthSessionForTest()
  __resetAuthStoreForTest()
  delete (globalThis as { window?: unknown }).window
})

test("initializeAuth should skip refresh when there is no session hint", async () => {
  installWindow()

  let refreshCallCount = 0
  authApi.refresh = async () => {
    refreshCallCount += 1
    return {
      tokens: demoTokens,
    }
  }

  await initializeAuth()

  assert.equal(refreshCallCount, 0)
  assert.equal(getAuthSnapshot().accessToken, null)
  assert.equal(getAuthSnapshot().hydrated, true)
})

test("initializeAuth should refresh when session hint exists", async () => {
  installWindow()

  const localStorage = (
    (globalThis as { window?: { localStorage?: Storage } }).window?.localStorage
  )
  assert.ok(localStorage)
  localStorage.setItem("auth_has_refresh_session", "1")

  let refreshCallCount = 0
  authApi.refresh = async () => {
    refreshCallCount += 1
    return {
      tokens: demoTokens,
    }
  }
  authApi.me = async () => demoUser

  await initializeAuth()

  assert.equal(refreshCallCount, 1)
  assert.equal(getAuthSnapshot().accessToken, "access-token")
  assert.equal(getAuthSnapshot().user?.email, "tester@example.com")
})

test("session hint should be written and cleared with auth session state", () => {
  installWindow()

  assert.equal(hasAuthSessionHint(), false)

  setAuthenticatedSession({
    tokens: demoTokens,
  })
  assert.equal(hasAuthSessionHint(), true)

  clearAuthenticatedSession()
  assert.equal(hasAuthSessionHint(), false)
})
