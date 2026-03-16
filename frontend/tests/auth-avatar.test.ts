import assert from "node:assert/strict"
import test from "node:test"

import { getAvatarFileId, toAvatarFileRef } from "../src/lib/auth/avatar"

test("avatar file ref helpers should parse and build avatar file ref", () => {
  const ref = toAvatarFileRef("file-id-1")
  assert.equal(ref, "file:file-id-1")
  assert.equal(getAvatarFileId(ref), "file-id-1")
  assert.equal(getAvatarFileId("https://example.com/avatar.png"), null)
  assert.equal(getAvatarFileId(""), null)
})
