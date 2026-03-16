import assert from "node:assert/strict"
import test from "node:test"

import { getUserDisplayName, maskEmail } from "../src/lib/auth/display"
import type { AuthUser } from "../src/lib/auth/types"

const demoUser: AuthUser = {
  user_id: "user-1",
  email: "tester@example.com",
  nickname: "  ",
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

test("maskEmail should keep domain and hide local part", () => {
  assert.equal(maskEmail("tester@example.com"), "te**er@example.com")
  assert.equal(maskEmail("ab@example.com"), "a*@example.com")
})

test("getUserDisplayName should prefer nickname and fallback to masked email", () => {
  assert.equal(getUserDisplayName(null), "未登录")

  const withNickname: AuthUser = { ...demoUser, nickname: "Alice" }
  assert.equal(getUserDisplayName(withNickname), "Alice")

  assert.equal(getUserDisplayName(demoUser), "te**er@example.com")
})
