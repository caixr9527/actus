export type AppLocale = "zh-CN" | "en-US"

export type MessageDictionary = Record<string, string>

export type MessageParams = Record<string, string | number>

export type Translate = (key: string, params?: MessageParams) => string
