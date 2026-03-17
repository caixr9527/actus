import assert from "node:assert/strict"
import test from "node:test"

import { validatePasswordStrength } from "../src/lib/auth/validators"

test("validatePasswordStrength should return null for strong password", () => {
  assert.equal(validatePasswordStrength("Password123!"), null)
})

test("validatePasswordStrength should return readable errors", () => {
  assert.equal(validatePasswordStrength("short1!"), "validation.passwordMinLength")
  assert.equal(validatePasswordStrength("12345678"), "validation.passwordRequireLetter")
  assert.equal(validatePasswordStrength("Password"), "validation.passwordRequireNumber")
  assert.equal(
    validatePasswordStrength("Password123~"),
    "validation.passwordAllowedChars",
  )
})
