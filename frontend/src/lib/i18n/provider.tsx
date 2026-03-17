"use client"

import {
  createContext,
  type ReactNode,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react"
import { getAuthSnapshot, subscribeAuthStore } from "@/lib/auth"
import { DEFAULT_APP_LOCALE } from "./constants"
import {
  getBrowserLocale,
  normalizeAppLocale,
  persistAppLocale,
  readPersistedAppLocale,
  resolveAppLocale,
} from "./locale"
import { appMessages } from "./messages"
import type { AppLocale, MessageParams, Translate } from "./types"

type I18nContextValue = {
  locale: AppLocale
  setLocale: (nextLocale: AppLocale) => void
  t: Translate
}

const I18nContext = createContext<I18nContextValue | null>(null)

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

function getCurrentUserLocale(): string | null {
  return getAuthSnapshot().user?.locale ?? null
}

export function I18nProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<AppLocale>(() =>
    resolveAppLocale({
      userLocale: getCurrentUserLocale(),
      persistedLocale: readPersistedAppLocale(),
      browserLocale: getBrowserLocale(),
      fallback: DEFAULT_APP_LOCALE,
    }),
  )

  useEffect(() => {
    return subscribeAuthStore(() => {
      const userLocale = getCurrentUserLocale()
      if (!userLocale) {
        return
      }

      const nextLocale = normalizeAppLocale(userLocale, DEFAULT_APP_LOCALE)
      setLocaleState((prev) => (prev === nextLocale ? prev : nextLocale))
      persistAppLocale(nextLocale)
    })
  }, [])

  useEffect(() => {
    if (typeof document === "undefined") {
      return
    }
    document.documentElement.lang = locale
  }, [locale])

  useEffect(() => {
    persistAppLocale(locale)
  }, [locale])

  const setLocale = useCallback((nextLocale: AppLocale) => {
    setLocaleState((prev) => (prev === nextLocale ? prev : nextLocale))
    persistAppLocale(nextLocale)
  }, [])

  const messages = useMemo(() => appMessages[locale], [locale])

  const t = useCallback<Translate>(
    (key, params) => {
      const template =
        messages[key] ??
        appMessages[DEFAULT_APP_LOCALE][key] ??
        key
      return formatMessage(template, params)
    },
    [messages],
  )

  const contextValue = useMemo(
    () => ({
      locale,
      setLocale,
      t,
    }),
    [locale, setLocale, t],
  )

  return <I18nContext.Provider value={contextValue}>{children}</I18nContext.Provider>
}

export function useI18nContext(): I18nContextValue {
  const context = useContext(I18nContext)
  if (!context) {
    throw new Error("useI18n must be used within I18nProvider")
  }
  return context
}
