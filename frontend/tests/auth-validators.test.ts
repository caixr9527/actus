import assert from "node:assert/strict"
import test from "node:test"

import { validatePasswordStrength } from "../src/lib/auth/validators"

test("validatePasswordStrength should return null for strong password", () => {
  assert.equal(validatePasswordStrength("Password123!"), null)
})

test("validatePasswordStrength should return readable errors", () => {
  assert.equal(validatePasswordStrength("short1!"), "密码长度不能少于8位")
  assert.equal(validatePasswordStrength("12345678"), "密码必须包含字母")
  assert.equal(validatePasswordStrength("Password"), "密码必须包含数字")
  assert.equal(
    validatePasswordStrength("Password123~"),
    "密码仅允许字母、数字和常见符号 !@#$%^&*._-",
  )
})
