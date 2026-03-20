"use client"

import { useEffect, useState } from "react"
import { fileApi } from "@/lib/api"
import { getAvatarFileId } from "@/lib/auth/avatar"

export function useAvatarSrc(avatarUrl: string | null | undefined): string | undefined {
  const [resolvedFileAvatar, setResolvedFileAvatar] = useState<{
    fileId: string
    src: string
  } | null>(null)
  const normalized = avatarUrl?.trim()
  const fileId = getAvatarFileId(normalized)

  useEffect(() => {
    if (!fileId) {
      return
    }

    let canceled = false
    let objectUrl: string | null = null

    void fileApi
      .downloadFile(fileId)
      .then((blob) => {
        if (canceled) {
          return
        }
        objectUrl = URL.createObjectURL(blob)
        setResolvedFileAvatar({
          fileId,
          src: objectUrl,
        })
      })
      .catch(() => undefined)

    return () => {
      canceled = true
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl)
      }
    }
  }, [fileId])

  if (!normalized) {
    return undefined
  }
  if (!fileId) {
    return normalized
  }
  if (resolvedFileAvatar?.fileId === fileId) {
    return resolvedFileAvatar.src
  }
  return undefined
}
