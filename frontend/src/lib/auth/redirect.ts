export function normalizeAuthRedirectTarget(value: string | null | undefined): string | null {
  if (!value) {
    return null
  }

  const normalized = value.trim()
  if (!normalized.startsWith('/')) {
    return null
  }
  if (normalized.startsWith('//')) {
    return null
  }

  return normalized
}
