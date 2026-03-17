import {
  APP_LOCALE_COOKIE_MAX_AGE,
  APP_LOCALE_COOKIE_NAME,
  APP_LOCALE_STORAGE_KEY,
  DEFAULT_APP_LOCALE,
  SUPPORTED_APP_LOCALES,
} from "./constants"
import type { AppLocale } from "./types"

function getWindowSafe(): Window | null {
  if (typeof window === "undefined") {
    return null
  }
  return window
}

export function isSupportedAppLocale(value: string): value is AppLocale {
  return SUPPORTED_APP_LOCALES.includes(value as AppLocale)
}

function tryNormalizeAppLocale(value: string | null | undefined): AppLocale | null {
  if (!value) {
    return null
  }

  const normalized = value.trim()
  if (!normalized) {
    return null
  }

  if (isSupportedAppLocale(normalized)) {
    return normalized
  }

  const lower = normalized.toLowerCase()
  if (lower === "zh" || lower.startsWith("zh-")) {
    return "zh-CN"
  }
  if (lower === "en" || lower.startsWith("en-")) {
    return "en-US"
  }

  return null
}

export function normalizeAppLocale(
  value: string | null | undefined,
  fallback: AppLocale = DEFAULT_APP_LOCALE,
): AppLocale {
  return tryNormalizeAppLocale(value) ?? fallback
}

export function resolveAppLocale(params: {
  userLocale?: string | null
  persistedLocale?: string | null
  browserLocale?: string | null
  fallback?: AppLocale
}): AppLocale {
  const fallback = params.fallback ?? DEFAULT_APP_LOCALE
  return (
    tryNormalizeAppLocale(params.userLocale) ??
    tryNormalizeAppLocale(params.persistedLocale) ??
    tryNormalizeAppLocale(params.browserLocale) ??
    fallback
  )
}

export function getBrowserLocale(): string | null {
  const win = getWindowSafe()
  if (!win) {
    return null
  }

  const primary = win.navigator.languages?.[0]
  if (typeof primary === "string" && primary.trim()) {
    return primary
  }

  const fallback = win.navigator.language
  if (typeof fallback === "string" && fallback.trim()) {
    return fallback
  }

  return null
}

function readLocaleFromCookie(): string | null {
  const win = getWindowSafe()
  if (!win) {
    return null
  }

  const cookieParts = win.document.cookie.split(";")
  for (const part of cookieParts) {
    const [rawKey, ...rawValueParts] = part.trim().split("=")
    if (rawKey !== APP_LOCALE_COOKIE_NAME) {
      continue
    }
    const rawValue = rawValueParts.join("=")
    if (!rawValue) {
      return null
    }
    try {
      return decodeURIComponent(rawValue)
    } catch {
      return rawValue
    }
  }
  return null
}

export function readPersistedAppLocale(): string | null {
  const win = getWindowSafe()
  if (!win) {
    return null
  }

  try {
    const localValue = win.localStorage.getItem(APP_LOCALE_STORAGE_KEY)
    if (localValue && localValue.trim()) {
      return localValue
    }
  } catch {
    // 忽略本地存储不可用场景，继续走 cookie 回退。
  }

  return readLocaleFromCookie()
}

export function persistAppLocale(locale: AppLocale): void {
  const win = getWindowSafe()
  if (!win) {
    return
  }

  try {
    win.localStorage.setItem(APP_LOCALE_STORAGE_KEY, locale)
  } catch {
    // 忽略本地存储不可用场景，保证主流程不受影响。
  }

  win.document.cookie = `${APP_LOCALE_COOKIE_NAME}=${encodeURIComponent(locale)}; path=/; max-age=${APP_LOCALE_COOKIE_MAX_AGE}; samesite=lax`
}
