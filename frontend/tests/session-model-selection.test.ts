import assert from "node:assert/strict"
import test from "node:test"

import { buildInitialSessionModelParams } from "../src/lib/session-model-selection"

test("buildInitialSessionModelParams should skip auto and empty model values", () => {
  assert.equal(buildInitialSessionModelParams(undefined), null)
  assert.equal(buildInitialSessionModelParams(null), null)
  assert.equal(buildInitialSessionModelParams("auto"), null)
})

test("buildInitialSessionModelParams should return request params for explicit model", () => {
  assert.deepEqual(buildInitialSessionModelParams("gpt-5.4"), {
    model_id: "gpt-5.4",
  })
})
