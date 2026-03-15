import assert from "node:assert/strict"
import test from "node:test"

import {
  __resetAuthStoreForTest,
  clearAuthenticatedSession,
  getAuthSnapshot,
  hydrateAuthStoreFromStorage,
  isAuthenticatedSnapshot,
  setAuthenticatedSession,
} from "../src/lib/auth/store"
import type { AuthTokenPair, AuthUser } from "../src/lib/auth/types"

const REFRESH_TOKEN_STORAGE_KEY = "actus.auth.refresh_token"

class MemoryStorage {
  private readonly data = new Map<string, string>()

  getItem(key: string): string | null {
    return this.data.has(key) ? this.data.get(key)! : null
  }

  setItem(key: string, value: string): void {
    this.data.set(key, value)
  }

  removeItem(key: string): void {
    this.data.delete(key)
  }
}

function installWindow(storage: MemoryStorage): void {
  ;(globalThis as { window?: unknown }).window = {
    localStorage: storage,
    location: {
      pathname: "/",
      search: "",
      assign: () => undefined,
    },
  }
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

const demoTokens: AuthTokenPair = {
  access_token: "access-token",
  refresh_token: "refresh-token",
  token_type: "Bearer",
  access_token_expires_in: 1800,
  refresh_token_expires_in: 604800,
}

test.afterEach(() => {
  __resetAuthStoreForTest()
  delete (globalThis as { window?: unknown }).window
})

test("hydrateAuthStoreFromStorage should load refresh token from localStorage", () => {
  __resetAuthStoreForTest()
  const storage = new MemoryStorage()
  storage.setItem(REFRESH_TOKEN_STORAGE_KEY, "persisted-refresh-token")
  installWindow(storage)

  hydrateAuthStoreFromStorage()

  const snapshot = getAuthSnapshot()
  assert.equal(snapshot.hydrated, true)
  assert.equal(snapshot.refreshToken, "persisted-refresh-token")
  assert.equal(snapshot.accessToken, null)
})

test("setAuthenticatedSession should persist refresh token and update in-memory state", () => {
  __resetAuthStoreForTest()
  const storage = new MemoryStorage()
  installWindow(storage)

  setAuthenticatedSession({
    tokens: demoTokens,
    user: demoUser,
  })

  const snapshot = getAuthSnapshot()
  assert.equal(snapshot.accessToken, "access-token")
  assert.equal(snapshot.refreshToken, "refresh-token")
  assert.equal(snapshot.user?.email, "tester@example.com")
  assert.equal(storage.getItem(REFRESH_TOKEN_STORAGE_KEY), "refresh-token")
})

test("clearAuthenticatedSession should clear localStorage and in-memory state", () => {
  __resetAuthStoreForTest()
  const storage = new MemoryStorage()
  installWindow(storage)

  setAuthenticatedSession({
    tokens: demoTokens,
    user: demoUser,
  })
  clearAuthenticatedSession()

  const snapshot = getAuthSnapshot()
  assert.equal(snapshot.accessToken, null)
  assert.equal(snapshot.refreshToken, null)
  assert.equal(snapshot.user, null)
  assert.equal(snapshot.hydrated, true)
  assert.equal(storage.getItem(REFRESH_TOKEN_STORAGE_KEY), null)
})

test("isAuthenticatedSnapshot should treat access token as logged in even without user", () => {
  __resetAuthStoreForTest()
  const storage = new MemoryStorage()
  installWindow(storage)

  setAuthenticatedSession({
    tokens: demoTokens,
  })

  const snapshot = getAuthSnapshot()
  assert.equal(snapshot.user, null)
  assert.equal(isAuthenticatedSnapshot(snapshot), true)
})
