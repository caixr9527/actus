import { translateRuntime } from "../i18n/runtime"
import type { MessageParams, Translate } from "../i18n/types"
import { ApiError } from "./fetch"

const API_ERROR_MESSAGE_KEY_BY_ERROR_KEY: Record<string, string> = {
  "error.auth.missing_credentials": "apiErrors.auth.missingCredentials",
  "error.auth.invalid_authorization_header": "apiErrors.auth.invalidAuthorizationHeader",
  "error.auth.access_token_required": "apiErrors.auth.accessTokenRequired",
  "error.auth.access_token_expired": "apiErrors.auth.accessTokenExpired",
  "error.auth.access_token_invalid": "apiErrors.auth.accessTokenInvalid",
  "error.auth.token_type_invalid": "apiErrors.auth.tokenTypeInvalid",
  "error.auth.token_user_missing": "apiErrors.auth.tokenUserMissing",
  "error.auth.session_invalidated": "apiErrors.auth.sessionInvalidated",
  "error.auth.service_unavailable": "apiErrors.auth.serviceUnavailable",
  "error.auth.user_not_found": "apiErrors.auth.userNotFound",
  "error.auth.user_status_invalid": "apiErrors.auth.userStatusInvalid",
  "error.auth.refresh_session_missing": "apiErrors.auth.refreshSessionMissing",
  "error.auth.password_mismatch": "authDialog.confirmPasswordMismatch",
  "error.auth.email_already_registered": "apiErrors.auth.emailAlreadyRegistered",
  "error.auth.register_verification_code_required": "authDialog.verificationCodeRequired",
  "error.auth.register_verification_code_invalid": "authDialog.verificationCodeInvalidOrExpired",
  "error.auth.send_code_failed": "apiErrors.auth.sendCodeFailed",
  "error.auth.invalid_credentials": "authDialog.loginCredentialError",
  "error.auth.login_failed": "apiErrors.auth.loginFailed",
  "error.auth.refresh_token_required": "apiErrors.auth.refreshTokenRequired",
  "error.auth.refresh_failed": "apiErrors.auth.refreshFailed",
  "error.auth.refresh_token_invalid": "apiErrors.auth.refreshTokenInvalid",
  "error.auth.refresh_replayed": "apiErrors.auth.refreshReplayed",
  "error.auth.logout_failed": "apiErrors.auth.logoutFailed",
  "error.auth.service_not_configured": "apiErrors.auth.serviceNotConfigured",
  "error.auth.register_code_service_not_configured": "apiErrors.auth.registerCodeServiceNotConfigured",
  "error.auth.email_service_not_configured": "apiErrors.auth.emailServiceNotConfigured",
  "error.auth.login_rate_limited": "apiErrors.auth.loginRateLimited",
  "error.auth.send_code_rate_limited": "apiErrors.auth.sendCodeRateLimited",
  "error.auth.https_required": "apiErrors.auth.httpsRequired",
  "error.user.not_found": "apiErrors.user.notFound",
  "error.user.profile_update_empty": "profile.noChanges",
  "error.user.locale_unsupported": "apiErrors.user.localeUnsupported",
  "error.user.new_password_mismatch": "profile.newPasswordMismatch",
  "error.user.current_password_incorrect": "profile.currentPasswordWrong",
  "error.user.session_cleanup_failed": "apiErrors.user.sessionCleanupFailed",
  "error.session.not_found": "apiErrors.session.notFound",
  "error.session.model_id_invalid": "apiErrors.session.modelIdInvalid",
  "error.session.sandbox_not_bound": "apiErrors.session.sandboxNotBound",
  "error.session.sandbox_unavailable": "apiErrors.session.sandboxUnavailable",
  "error.session.file_read_failed": "apiErrors.session.fileReadFailed",
  "error.session.shell_read_failed": "apiErrors.session.shellReadFailed",
  "error.session.resume_required": "apiErrors.session.resumeRequired",
  "error.session.not_waiting": "apiErrors.session.notWaiting",
  "error.session.resume_checkpoint_invalid": "apiErrors.session.resumeCheckpointInvalid",
  "error.session.resume_value_invalid": "apiErrors.session.resumeValueInvalid",
  "error.session.not_cancelled": "apiErrors.session.notCancelled",
  "error.session.cancelled_continue_unavailable": "apiErrors.session.cancelledContinueUnavailable",
  "error.file.not_found": "apiErrors.file.notFound",
  "error.app_config.load_failed": "apiErrors.appConfig.loadFailed",
  "error.app_config.save_failed": "apiErrors.appConfig.saveFailed",
  "error.app_config.model_invalid": "apiErrors.appConfig.modelInvalid",
  "error.app_config.default_model_unavailable": "apiErrors.appConfig.defaultModelUnavailable",
  "error.app_config.mcp_server_not_found": "apiErrors.appConfig.mcpServerNotFound",
  "error.app_config.a2a_server_not_found": "apiErrors.appConfig.a2aServerNotFound",
  "error.app_config.mcp_servers_load_failed": "apiErrors.appConfig.mcpServersLoadFailed",
  "error.app_config.a2a_servers_load_failed": "apiErrors.appConfig.a2aServersLoadFailed",
  "error.status.unhealthy": "apiErrors.status.unhealthy",
}

type Translator = Translate | ((key: string, params?: MessageParams) => string)

type ApiErrorPayload = {
  error_key?: string | null
  error_params?: Record<string, unknown> | null
  errorKey?: string | null
  errorParams?: Record<string, unknown> | null
}

function normalizeErrorParams(
  params?: Record<string, unknown> | null,
): MessageParams | undefined {
  if (!params) {
    return undefined
  }

  const normalized: MessageParams = {}
  for (const [key, value] of Object.entries(params)) {
    if (typeof value === "number") {
      normalized[key] = value
      continue
    }
    if (value !== undefined && value !== null) {
      normalized[key] = String(value)
    }
  }

  return Object.keys(normalized).length > 0 ? normalized : undefined
}

function resolveTranslator(translate?: Translator): Translator {
  return translate ?? translateRuntime
}

function getPayloadErrorKey(payload?: ApiErrorPayload | null): string | null {
  if (!payload) {
    return null
  }
  return payload.errorKey ?? payload.error_key ?? null
}

function getPayloadErrorParams(
  payload?: ApiErrorPayload | null,
): Record<string, unknown> | null {
  if (!payload) {
    return null
  }
  return payload.errorParams ?? payload.error_params ?? null
}

export function getApiErrorMessageKey(errorKey?: string | null): string | null {
  if (!errorKey) {
    return null
  }
  return API_ERROR_MESSAGE_KEY_BY_ERROR_KEY[errorKey] ?? null
}

export function isApiErrorKey(error: unknown, errorKey: string): error is ApiError {
  return error instanceof ApiError && error.errorKey === errorKey
}

export function getApiErrorMessageFromPayload(
  payload: ApiErrorPayload | null | undefined,
  fallbackKey: string,
  translate?: Translator,
): string {
  const messageKey = getApiErrorMessageKey(getPayloadErrorKey(payload)) ?? fallbackKey
  return resolveTranslator(translate)(
    messageKey,
    normalizeErrorParams(getPayloadErrorParams(payload)),
  )
}

export function getApiErrorMessage(
  error: unknown,
  fallbackKey: string,
  translate?: Translator,
): string {
  if (error instanceof ApiError) {
    return getApiErrorMessageFromPayload(error, fallbackKey, translate)
  }
  if (error instanceof Error) {
    return error.message || resolveTranslator(translate)(fallbackKey)
  }
  return resolveTranslator(translate)(fallbackKey)
}
