import assert from "node:assert/strict"
import test from "node:test"

import { normalizeAuthRedirectTarget } from "../src/lib/auth/redirect"

test("normalizeAuthRedirectTarget should only allow internal paths", () => {
  assert.equal(normalizeAuthRedirectTarget("/sessions/abc"), "/sessions/abc")
  assert.equal(normalizeAuthRedirectTarget(" /sessions/abc?x=1 "), "/sessions/abc?x=1")
  assert.equal(normalizeAuthRedirectTarget("https://evil.com"), null)
  assert.equal(normalizeAuthRedirectTarget("//evil.com"), null)
  assert.equal(normalizeAuthRedirectTarget(""), null)
  assert.equal(normalizeAuthRedirectTarget(null), null)
})
