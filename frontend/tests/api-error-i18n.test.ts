import assert from "node:assert/strict"
import test from "node:test"

import { ApiError } from "../src/lib/api/fetch"
import {
  getApiErrorMessage,
  getApiErrorMessageFromPayload,
  isApiErrorKey,
} from "../src/lib/api/error-i18n"
import { appMessages } from "../src/lib/i18n/messages"
import type { MessageParams } from "../src/lib/i18n/types"

function translate(locale: "zh-CN" | "en-US") {
  return (key: string, params?: MessageParams): string => {
    const template = appMessages[locale][key] ?? key
    if (!params) {
      return template
    }
    return template.replace(/\{(\w+)\}/g, (match, name: string) => {
      if (!(name in params)) {
        return match
      }
      return String(params[name] ?? "")
    })
  }
}

test("getApiErrorMessage should map backend error_key to localized frontend copy", () => {
  const error = new ApiError(
    401,
    "邮箱或密码错误",
    null,
    "error.auth.invalid_credentials",
  )

  assert.equal(
    getApiErrorMessage(error, "authDialog.loginFailed", translate("en-US")),
    "Incorrect email or password",
  )
  assert.equal(isApiErrorKey(error, "error.auth.invalid_credentials"), true)
})

test("getApiErrorMessageFromPayload should fallback to provided key when error_key is unknown", () => {
  assert.equal(
    getApiErrorMessageFromPayload(
      {
        error_key: "error.unknown",
      },
      "profile.updateFailed",
      translate("zh-CN"),
    ),
    "资料更新失败，请稍后重试",
  )
})
