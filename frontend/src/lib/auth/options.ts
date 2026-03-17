export type SelectOption = {
  value: string
  label: string
}

const COMMON_TIMEZONES = [
  "Asia/Shanghai",
  "Asia/Tokyo",
  "Asia/Seoul",
  "Asia/Singapore",
  "Asia/Hong_Kong",
  "Asia/Bangkok",
  "Asia/Kolkata",
  "Asia/Dubai",
  "Australia/Sydney",
  "Europe/London",
  "Europe/Paris",
  "Europe/Berlin",
  "Europe/Madrid",
  "Europe/Rome",
  "Europe/Amsterdam",
  "Europe/Moscow",
  "America/New_York",
  "America/Chicago",
  "America/Denver",
  "America/Los_Angeles",
  "America/Toronto",
  "America/Sao_Paulo",
  "America/Mexico_City",
  "Pacific/Auckland",
  "UTC",
] as const

const COMMON_LOCALES = [
  "zh-CN",
  "en-US",
  // "zh-TW",
  // "zh-HK",
  // "en-GB",
  // "en-AU",
  // "en-CA",
  // "ja-JP",
  // "ko-KR",
  // "fr-FR",
  // "de-DE",
  // "es-ES",
  // "es-MX",
  // "it-IT",
  // "nl-NL",
  // "sv-SE",
  // "pl-PL",
  // "ru-RU",
  // "tr-TR",
  // "pt-BR",
  // "pt-PT",
  // "ar-SA",
  // "hi-IN",
  // "th-TH",
  // "vi-VN",
  // "id-ID",
] as const

let timeZoneCache: SelectOption[] | null = null
const localeCacheByDisplayLocale = new Map<string, SelectOption[]>()

export function getTimeZoneOptions(): SelectOption[] {
  if (timeZoneCache) {
    return timeZoneCache
  }

  timeZoneCache = COMMON_TIMEZONES.map((value) => ({
    value,
    label: value,
  }))
  return timeZoneCache
}

function normalizeLocaleCode(localeCode: string): string {
  const locale = new Intl.Locale(localeCode)
  const language = locale.language
  const region = locale.region
  if (!region) {
    return language
  }
  return `${language}-${region}`
}

function buildLocaleLabel(localeCode: string, displayLocale: string): string {
  const locale = new Intl.Locale(localeCode)
  const language = locale.language
  const region = locale.region
  const languageNames = new Intl.DisplayNames([displayLocale], { type: "language" })
  const regionNames = new Intl.DisplayNames([displayLocale], { type: "region" })
  const languageLabel = languageNames.of(language) ?? language
  const fallbackRegionLabel = displayLocale.startsWith("en") ? "Unknown Region" : "未指定地区"
  const regionLabel = region ? (regionNames.of(region) ?? region) : fallbackRegionLabel
  return `${languageLabel}（${regionLabel}） (${localeCode})`
}

export function getLocaleOptions(displayLocale = "zh-CN"): SelectOption[] {
  const cacheKey = normalizeLocaleCode(displayLocale)
  const cached = localeCacheByDisplayLocale.get(cacheKey)
  if (cached) {
    return cached
  }

  const options = COMMON_LOCALES.map((localeCode) => normalizeLocaleCode(localeCode)).map(
    (localeCode) => ({
      value: localeCode,
      label: buildLocaleLabel(localeCode, displayLocale),
    }),
  )
  localeCacheByDisplayLocale.set(cacheKey, options)
  return options
}

export function ensureOptionExists(
  options: SelectOption[],
  value: string,
): SelectOption[] {
  const normalized = value.trim()
  if (!normalized) {
    return options
  }
  if (options.some((item) => item.value === normalized)) {
    return options
  }
  return [{ value: normalized, label: normalized }, ...options]
}

export function filterSelectOptions(
  options: SelectOption[],
  query: string,
  selectedValue?: string,
): SelectOption[] {
  const normalizedQuery = query.trim().toLowerCase()
  const filtered = normalizedQuery
    ? options.filter((option) => {
        const valueText = option.value.toLowerCase()
        const labelText = option.label.toLowerCase()
        return valueText.includes(normalizedQuery) || labelText.includes(normalizedQuery)
      })
    : options

  const normalizedSelectedValue = selectedValue?.trim()
  if (!normalizedSelectedValue) {
    return filtered
  }
  if (filtered.some((item) => item.value === normalizedSelectedValue)) {
    return filtered
  }
  const selectedOption = options.find((item) => item.value === normalizedSelectedValue)
  if (!selectedOption) {
    return filtered
  }
  return [selectedOption, ...filtered]
}
