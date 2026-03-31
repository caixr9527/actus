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

test("getApiErrorMessage should map new model-related error_key values", () => {
  assert.equal(
    getApiErrorMessageFromPayload(
      {
        error_key: "error.session.model_id_invalid",
      },
      "chatInput.modelUpdateFailed",
      translate("zh-CN"),
    ),
    "所选模型不存在或未启用",
  )

  assert.equal(
    getApiErrorMessageFromPayload(
      {
        error_key: "error.app_config.default_model_unavailable",
      },
      "chatInput.modelUpdateFailed",
      translate("en-US"),
    ),
    "The default model is temporarily unavailable. Please try again later.",
  )

  assert.equal(
    getApiErrorMessageFromPayload(
      {
        error_key: "error.app_config.mcp_servers_load_failed",
      },
      "api.requestFailed",
      translate("zh-CN"),
    ),
    "加载 MCP 服务器列表失败，请稍后重试",
  )

  assert.equal(
    getApiErrorMessageFromPayload(
      {
        error_key: "error.app_config.a2a_servers_load_failed",
      },
      "api.requestFailed",
      translate("en-US"),
    ),
    "Failed to load A2A agents. Please try again later.",
  )

  assert.equal(
    getApiErrorMessageFromPayload(
      {
        error_key: "error.session.resume_required",
      },
      "sessionDetail.sendFailed",
      translate("zh-CN"),
    ),
    "当前会话正在等待恢复，请继续输入后再执行",
  )

  assert.equal(
    getApiErrorMessageFromPayload(
      {
        error_key: "error.session.not_waiting",
      },
      "sessionDetail.sendFailed",
      translate("en-US"),
    ),
    "This session is not waiting, so it cannot be resumed.",
  )
})
