import type { AuthUser } from "./types"

type DisplayFallbacks = {
  unknownUserLabel?: string
  guestLabel?: string
}

function maskEmailLocalPart(localPart: string): string {
  if (localPart.length <= 2) {
    return `${localPart[0] ?? ""}*`
  }
  if (localPart.length <= 4) {
    return `${localPart[0]}${"*".repeat(localPart.length - 1)}`
  }
  return `${localPart.slice(0, 2)}${"*".repeat(Math.max(2, localPart.length - 4))}${localPart.slice(-2)}`
}

export function maskEmail(
  email: string,
  unknownUserLabel: string = "未命名用户",
): string {
  const normalized = email.trim().toLowerCase()
  if (!normalized.includes("@")) {
    return normalized || unknownUserLabel
  }
  const [localPart, domain] = normalized.split("@")
  return `${maskEmailLocalPart(localPart)}@${domain}`
}

export function getUserDisplayName(
  user: AuthUser | null,
  fallbacks: DisplayFallbacks = {},
): string {
  const guestLabel = fallbacks.guestLabel ?? "未登录"
  const unknownUserLabel = fallbacks.unknownUserLabel ?? "未命名用户"

  if (!user) {
    return guestLabel
  }
  const nickname = user.nickname?.trim()
  if (nickname) {
    return nickname
  }
  return maskEmail(user.email, unknownUserLabel)
}
