import { getAuthSnapshot } from "../auth/store"
import { DEFAULT_APP_LOCALE } from "./constants"
import { normalizeAppLocale, resolveAppLocale, readPersistedAppLocale, getBrowserLocale } from "./locale"
import { appMessages } from "./messages"
import type { AppLocale, MessageParams } from "./types"

function formatMessage(template: string, params?: MessageParams): string {
  if (!params) {
    return template
  }

  return template.replace(/\{(\w+)\}/g, (match, key: string) => {
    if (!(key in params)) {
      return match
    }
    return String(params[key] ?? "")
  })
}

export function resolveRuntimeLocale(): AppLocale {
  const userLocale = getAuthSnapshot().user?.locale
  return resolveAppLocale({
    userLocale: userLocale ? normalizeAppLocale(userLocale, DEFAULT_APP_LOCALE) : null,
    persistedLocale: readPersistedAppLocale(),
    browserLocale: getBrowserLocale(),
    fallback: DEFAULT_APP_LOCALE,
  })
}

export function translateRuntime(
  key: string,
  params?: MessageParams,
  locale?: AppLocale,
): string {
  const targetLocale = locale ?? resolveRuntimeLocale()
  const template =
    appMessages[targetLocale][key] ??
    appMessages[DEFAULT_APP_LOCALE][key] ??
    key
  return formatMessage(template, params)
}
