import type { AppLocale } from "./types"

export const SUPPORTED_APP_LOCALES: readonly AppLocale[] = ["zh-CN", "en-US"]

export const DEFAULT_APP_LOCALE: AppLocale = "zh-CN"

export const APP_LOCALE_STORAGE_KEY = "app_locale"

export const APP_LOCALE_COOKIE_NAME = "app_locale"

export const APP_LOCALE_COOKIE_MAX_AGE = 60 * 60 * 24 * 365
