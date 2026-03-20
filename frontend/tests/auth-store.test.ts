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

function installWindow(): void {
  ;(globalThis as { window?: unknown }).window = {
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
  token_type: "Bearer",
  access_token_expires_in: 1800,
}

test.afterEach(() => {
  __resetAuthStoreForTest()
  delete (globalThis as { window?: unknown }).window
})

test("hydrateAuthStoreFromStorage should mark state as hydrated", () => {
  __resetAuthStoreForTest()
  installWindow()

  hydrateAuthStoreFromStorage()

  const snapshot = getAuthSnapshot()
  assert.equal(snapshot.hydrated, true)
  assert.equal(snapshot.accessToken, null)
})

test("setAuthenticatedSession should update in-memory state", () => {
  __resetAuthStoreForTest()
  installWindow()

  setAuthenticatedSession({
    tokens: demoTokens,
    user: demoUser,
  })

  const snapshot = getAuthSnapshot()
  assert.equal(snapshot.accessToken, "access-token")
  assert.equal(snapshot.user?.email, "tester@example.com")
})

test("clearAuthenticatedSession should clear in-memory state", () => {
  __resetAuthStoreForTest()
  installWindow()

  setAuthenticatedSession({
    tokens: demoTokens,
    user: demoUser,
  })
  clearAuthenticatedSession()

  const snapshot = getAuthSnapshot()
  assert.equal(snapshot.accessToken, null)
  assert.equal(snapshot.user, null)
  assert.equal(snapshot.hydrated, true)
})

test("isAuthenticatedSnapshot should treat access token as logged in even without user", () => {
  __resetAuthStoreForTest()
  installWindow()

  setAuthenticatedSession({
    tokens: demoTokens,
  })

  const snapshot = getAuthSnapshot()
  assert.equal(snapshot.user, null)
  assert.equal(isAuthenticatedSnapshot(snapshot), true)
})
