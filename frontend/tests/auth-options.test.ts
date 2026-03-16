import assert from "node:assert/strict"
import test from "node:test"

import {
  ensureOptionExists,
  filterSelectOptions,
  getLocaleOptions,
  getTimeZoneOptions,
} from "../src/lib/auth/options"

test("getTimeZoneOptions should return non-empty options", () => {
  const options = getTimeZoneOptions()
  assert.equal(options.length > 0, true)
  assert.equal(options.length <= 40, true)
  assert.equal(typeof options[0]?.value, "string")
  assert.equal(options.some((item) => item.value === "Asia/Shanghai"), true)
  assert.equal(options.some((item) => item.value === "America/New_York"), true)
  assert.equal(options.some((item) => item.value === "UTC"), true)
})

test("getLocaleOptions should return non-empty options", () => {
  const options = getLocaleOptions()
  assert.equal(options.length > 0, true)
  assert.equal(options.length <= 40, true)
  assert.equal(typeof options[0]?.label, "string")
  assert.equal(options.some((item) => item.value === "zh-CN"), true)
  assert.equal(options.some((item) => item.value === "en-US"), true)
})

test("ensureOptionExists should prepend custom option when missing", () => {
  const options = [{ value: "en", label: "English (en)" }]
  const next = ensureOptionExists(options, "zh-CN")
  assert.equal(next[0]?.value, "zh-CN")
})

test("filterSelectOptions should filter by query and keep selected option", () => {
  const options = [
    { value: "Asia/Shanghai", label: "Asia/Shanghai" },
    { value: "Asia/Tokyo", label: "Asia/Tokyo" },
    { value: "UTC", label: "UTC" },
  ]

  const filtered = filterSelectOptions(options, "tokyo")
  assert.deepEqual(filtered.map((item) => item.value), ["Asia/Tokyo"])

  const keepSelected = filterSelectOptions(options, "tokyo", "UTC")
  assert.equal(keepSelected[0]?.value, "UTC")
  assert.equal(keepSelected.some((item) => item.value === "Asia/Tokyo"), true)
})
