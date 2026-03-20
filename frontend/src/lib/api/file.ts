import { ApiError, get, getApiBaseUrl, post, requestRaw } from "./fetch"
import type { FileInfo, FileUploadParams } from "./types"
import { translateRuntime } from "../i18n/runtime"

type FileDownloadErrorPayload = {
  msg?: string
  message?: string
  data?: unknown
  error_key?: string | null
  error_params?: Record<string, unknown> | null
}

/**
 * 文件模块 API
 */
export const fileApi = {
  /**
   * 上传文件
   * @param params 上传参数，包含文件和可选的会话 ID
   * @returns 文件信息
   */
  uploadFile: async (params: FileUploadParams): Promise<FileInfo> => {
    const formData = new FormData()
    formData.append("file", params.file)

    if (params.session_id) {
      formData.append("session_id", params.session_id)
    }

    return post<FileInfo>("/files", formData)
  },

  /**
   * 获取文件信息
   * @param fileId 文件 ID
   * @returns 文件信息
   */
  getFileInfo: (fileId: string): Promise<FileInfo> => {
    return get<FileInfo>(`/files/${fileId}`)
  },

  /**
   * 下载文件
   * @param fileId 文件 ID
   * @param options 下载选项（可选）
   * @returns Blob 对象
   */
  downloadFile: async (
    fileId: string,
    options?: { signal?: AbortSignal },
  ): Promise<Blob> => {
    const response = await requestRaw(`/files/${fileId}/download`, {
      method: "GET",
      signal: options?.signal,
    })

    if (!response.ok) {
      let errorMessage = response.statusText || translateRuntime("fileApi.downloadFailed")
      let errorData: FileDownloadErrorPayload | null = null

      try {
        const contentType = response.headers.get("content-type")
        if (contentType?.includes("application/json")) {
          errorData = (await response.json()) as FileDownloadErrorPayload
          errorMessage = errorData?.msg || errorData?.message || errorMessage
        } else {
          const text = await response.text()
          errorMessage = text || errorMessage
        }
      } catch {
        // ignore parse error and fallback to statusText
      }
      throw new ApiError(
        response.status,
        errorMessage,
        errorData?.data ?? null,
        errorData?.error_key ?? null,
        errorData?.error_params ?? null,
      )
    }

    return response.blob()
  },

  /**
   * 下载文件并获取 URL（用于直接下载或预览）
   * @param fileId 文件 ID
   * @returns 文件下载 URL
   */
  getFileDownloadUrl: (fileId: string): string => {
    return `${getApiBaseUrl()}/files/${fileId}/download`
  },
}
