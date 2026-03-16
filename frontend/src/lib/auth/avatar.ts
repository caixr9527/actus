const AVATAR_FILE_PREFIX = "file:"

export function toAvatarFileRef(fileId: string): string {
  return `${AVATAR_FILE_PREFIX}${fileId}`
}

export function getAvatarFileId(avatarUrl: string | null | undefined): string | null {
  const normalized = avatarUrl?.trim()
  if (!normalized || !normalized.startsWith(AVATAR_FILE_PREFIX)) {
    return null
  }
  const fileId = normalized.slice(AVATAR_FILE_PREFIX.length).trim()
  return fileId || null
}
