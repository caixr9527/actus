import type { AppLocale, MessageDictionary } from "../types"
import { enUSMessages } from "./en-US"
import { zhCNMessages } from "./zh-CN"

export const appMessages: Record<AppLocale, MessageDictionary> = {
  "zh-CN": zhCNMessages,
  "en-US": enUSMessages,
}
