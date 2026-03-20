import assert from "node:assert/strict"
import test from "node:test"

import { normalizeAppLocale, resolveAppLocale } from "../src/lib/i18n/locale"

test("normalizeAppLocale should normalize zh/en variants", () => {
  assert.equal(normalizeAppLocale("zh"), "zh-CN")
  assert.equal(normalizeAppLocale("zh-HK"), "zh-CN")
  assert.equal(normalizeAppLocale("en"), "en-US")
  assert.equal(normalizeAppLocale("en-GB"), "en-US")
})

test("normalizeAppLocale should fallback when locale unsupported", () => {
  assert.equal(normalizeAppLocale("ja-JP"), "zh-CN")
  assert.equal(normalizeAppLocale("ja-JP", "en-US"), "en-US")
  assert.equal(normalizeAppLocale(""), "zh-CN")
})

test("resolveAppLocale should use user locale as highest priority", () => {
  const locale = resolveAppLocale({
    userLocale: "en",
    persistedLocale: "zh-CN",
    browserLocale: "zh",
  })

  assert.equal(locale, "en-US")
})

test("resolveAppLocale should fallback to persisted then browser locale", () => {
  const fromPersisted = resolveAppLocale({
    persistedLocale: "en-CA",
    browserLocale: "zh-CN",
  })
  assert.equal(fromPersisted, "en-US")

  const fromBrowser = resolveAppLocale({
    browserLocale: "en-GB",
  })
  assert.equal(fromBrowser, "en-US")
})
