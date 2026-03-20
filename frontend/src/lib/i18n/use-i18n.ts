"use client"

import { useI18nContext } from "./provider"

export function useI18n() {
  return useI18nContext()
}
