import type { ApiResponse } from "./types"

/**
 * API 配置
 */
const API_CONFIG = {
  baseURL: process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:23140/api",
  timeout: 30000, // 30秒
} as const

export type AuthHooks = {
  getAccessToken: () => string | null
  refreshAccessToken: () => Promise<string | null>
  onAuthFailure: () => void
}

let authHooks: AuthHooks | null = null

export function registerAuthHooks(hooks: AuthHooks | null): void {
  authHooks = hooks
}

export type RequestOptions = RequestInit & {
  timeout?: number
  skipErrorHandler?: boolean
  skipAuth?: boolean
  skipAuthRefresh?: boolean
}

function resolveRequestUrl(endpoint: string): string {
  return endpoint.startsWith("http")
    ? endpoint
    : `${API_CONFIG.baseURL}${endpoint}`
}

export function getApiBaseUrl(): string {
  return API_CONFIG.baseURL
}

function getCorsAllowedOriginsHint(requestUrl: string): string | null {
  if (typeof window === "undefined") return null

  try {
    const pageOrigin = window.location.origin
    const targetOrigin = new URL(requestUrl, pageOrigin).origin
    if (targetOrigin === pageOrigin) return null
    return `检测到跨域请求（${pageOrigin} -> ${targetOrigin}），请将 ${pageOrigin} 加入后端 CORS_ALLOWED_ORIGINS`
  } catch {
    return null
  }
}

function toNetworkApiError(requestUrl: string): ApiError {
  const corsHint = getCorsAllowedOriginsHint(requestUrl)
  if (corsHint) {
    return new ApiError(500, `网络请求失败。${corsHint}`)
  }
  return new ApiError(500, "网络连接失败，请检查网络设置")
}

/**
 * 自定义错误类
 */
export class ApiError extends Error {
  constructor(
    public code: number,
    public msg: string,
    public data: unknown = null,
  ) {
    super(msg)
    this.name = "ApiError"
  }
}

/**
 * 解析响应
 */
async function parseResponse<T>(response: Response): Promise<ApiResponse<T>> {
  const contentType = response.headers.get("content-type")

  if (contentType?.includes("application/json")) {
    return await response.json()
  }

  // 处理非 JSON 响应（如文件下载）
  const text = await response.text()
  return {
    code: response.ok ? 0 : response.status,
    msg: response.ok ? "success" : text || response.statusText,
    data: text as unknown as T,
  }
}

/**
 * 处理错误响应
 */
async function handleErrorResponse(response: Response): Promise<never> {
  let errorData: ApiResponse

  try {
    errorData = await parseResponse(response)
  } catch {
    errorData = {
      code: response.status,
      msg: response.statusText || "请求失败",
      data: null,
    }
  }

  throw new ApiError(errorData.code, errorData.msg, errorData.data)
}

/**
 * 带超时的 fetch
 */
function fetchWithTimeout(
  url: string,
  options: RequestInit = {},
  timeout: number = API_CONFIG.timeout,
): Promise<Response> {
  return new Promise((resolve, reject) => {
    const controller = new AbortController()
    const externalSignal = options.signal
    let abortedByTimeout = false

    const onExternalAbort = () => {
      controller.abort()
    }

    if (externalSignal) {
      if (externalSignal.aborted) {
        controller.abort()
      } else {
        externalSignal.addEventListener("abort", onExternalAbort, { once: true })
      }
    }

    const timeoutId = setTimeout(() => {
      abortedByTimeout = true
      controller.abort()
    }, timeout)

    fetch(url, {
      ...options,
      signal: controller.signal,
    })
      .then((response) => {
        resolve(response)
      })
      .catch((error) => {
        if (error instanceof Error && error.name === "AbortError") {
          if (abortedByTimeout) {
            reject(new ApiError(408, "请求超时"))
          } else {
            reject(error)
          }
          return
        }
        reject(error)
      })
      .finally(() => {
        clearTimeout(timeoutId)
        if (externalSignal) {
          externalSignal.removeEventListener("abort", onExternalAbort)
        }
      })
  })
}

function normalizeHeaders(
  headers: HeadersInit | undefined,
  body: BodyInit | null | undefined,
  accessToken: string | null,
  skipAuth: boolean,
): Headers {
  const normalized = new Headers(headers)

  if (
    body !== undefined &&
    body !== null &&
    !(body instanceof FormData) &&
    !normalized.has("Content-Type")
  ) {
    normalized.set("Content-Type", "application/json")
  }

  if (!skipAuth && accessToken && !normalized.has("Authorization")) {
    normalized.set("Authorization", `Bearer ${accessToken}`)
  }

  return normalized
}

async function fetchWithAuthRetry(
  requestUrl: string,
  options: RequestOptions = {},
): Promise<Response> {
  const {
    timeout = API_CONFIG.timeout,
    headers,
    skipAuth = false,
    skipAuthRefresh = false,
    skipErrorHandler,
    ...fetchOptions
  } = options
  void skipErrorHandler

  const execute = async (accessToken: string | null): Promise<Response> => {
    const normalizedHeaders = normalizeHeaders(
      headers,
      fetchOptions.body,
      accessToken,
      skipAuth,
    )
    const credentials = fetchOptions.credentials ?? "include"

    return fetchWithTimeout(
      requestUrl,
      {
        ...fetchOptions,
        credentials,
        headers: normalizedHeaders,
      },
      timeout,
    )
  }

  const initialAccessToken = authHooks?.getAccessToken() ?? null
  let response = await execute(initialAccessToken)

  if (response.status !== 401 || skipAuthRefresh || authHooks === null) {
    return response
  }

  const refreshedAccessToken = await authHooks.refreshAccessToken()
  if (!refreshedAccessToken) {
    authHooks.onAuthFailure()
    return response
  }

  response = await execute(refreshedAccessToken)
  if (response.status === 401) {
    authHooks.onAuthFailure()
  }
  return response
}

/**
 * 发起原始 HTTP 请求（返回原始 Response）
 */
export async function requestRaw(
  endpoint: string,
  options: RequestOptions = {},
): Promise<Response> {
  const requestUrl = resolveRequestUrl(endpoint)

  try {
    return await fetchWithAuthRetry(requestUrl, options)
  } catch (error) {
    if (error instanceof ApiError) {
      throw error
    }

    // 处理网络错误
    if (
      error instanceof TypeError &&
      /failed to fetch|load failed/i.test(error.message)
    ) {
      throw toNetworkApiError(requestUrl)
    }

    throw new ApiError(500, error instanceof Error ? error.message : "未知错误")
  }
}

/**
 * 核心请求函数
 */
export async function request<T = unknown>(
  endpoint: string,
  options: RequestOptions = {},
): Promise<T> {
  const { skipErrorHandler = false, ...rawOptions } = options

  const response = await requestRaw(endpoint, rawOptions)

  // 处理 HTTP 错误状态码
  if (!response.ok) {
    if (skipErrorHandler) {
      return parseResponse<T>(response) as Promise<T>
    }
    await handleErrorResponse(response)
  }

  const result = await parseResponse<T>(response)

  // 处理业务错误（code 不在成功范围内）
  if (result.code !== 0 && result.code !== 200) {
    if (skipErrorHandler) {
      return result.data as T
    }
    throw new ApiError(result.code, result.msg, result.data)
  }

  return result.data as T
}

/**
 * GET 请求
 */
export function get<T = unknown>(
  endpoint: string,
  params?: Record<string, string | number | boolean>,
  options?: RequestOptions,
): Promise<T> {
  let url = endpoint

  if (params) {
    const searchParams = new URLSearchParams()
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        searchParams.append(key, String(value))
      }
    })
    const queryString = searchParams.toString()
    if (queryString) {
      url += `?${queryString}`
    }
  }

  return request<T>(url, {
    ...options,
    method: "GET",
  })
}

/**
 * POST 请求
 */
export function post<T = unknown>(
  endpoint: string,
  data?: unknown,
  options?: RequestOptions,
): Promise<T> {
  return request<T>(endpoint, {
    ...options,
    method: "POST",
    body: data instanceof FormData ? data : JSON.stringify(data),
  })
}

/**
 * PUT 请求
 */
export function put<T = unknown>(
  endpoint: string,
  data?: unknown,
  options?: RequestOptions,
): Promise<T> {
  return request<T>(endpoint, {
    ...options,
    method: "PUT",
    body: JSON.stringify(data),
  })
}

/**
 * DELETE 请求
 */
export function del<T = unknown>(
  endpoint: string,
  options?: RequestOptions,
): Promise<T> {
  return request<T>(endpoint, {
    ...options,
    method: "DELETE",
  })
}

/**
 * 创建 SSE 连接
 */
export function createSSEConnection(
  endpoint: string,
  data?: unknown,
): EventSource {
  const url = resolveRequestUrl(endpoint)

  // 对于 POST 请求，使用 fetch + ReadableStream 方式
  if (data !== undefined) {
    throw new Error("带数据的 SSE 连接请使用 createSSEStream 函数")
  }

  return new EventSource(url)
}

/**
 * 创建流式 SSE 连接（支持 POST 请求）
 *
 * @param endpoint  接口路径
 * @param data      请求体
 * @param options   请求选项，可通过 `signal` 传入外部 AbortSignal 以便调用方随时中止连接
 */
export async function createSSEStream(
  endpoint: string,
  data?: unknown,
  options?: RequestOptions,
): Promise<ReadableStream<Uint8Array>> {
  const { headers = {}, ...fetchOptions } = options || {}

  const response = await requestRaw(endpoint, {
    ...fetchOptions,
    method: "POST",
    headers: {
      Accept: "text/event-stream",
      ...headers,
    },
    body: JSON.stringify(data),
  })

  if (!response.ok) {
    await handleErrorResponse(response)
  }

  if (!response.body) {
    throw new ApiError(500, "响应体为空")
  }

  return response.body
}

/**
 * 解析 SSE 事件流
 */
export async function parseSSEStream(
  stream: ReadableStream<Uint8Array>,
  onEvent: (event: MessageEvent) => void,
  onError?: (error: Error) => void,
): Promise<void> {
  const reader = stream.getReader()
  const decoder = new TextDecoder()
  let buffer = ""

  try {
    while (true) {
      const { done, value } = await reader.read()

      if (done) {
        // 处理缓冲区中剩余的数据
        if (buffer.trim()) {
          processSSEBuffer(buffer, onEvent, onError)
        }
        break
      }

      buffer += decoder.decode(value, { stream: true })

      // 标准化行尾：服务端可能使用 \r\n (CRLF)，统一转为 \n (LF)
      buffer = buffer.replace(/\r\n/g, "\n").replace(/\r/g, "\n")

      const parts = buffer.split("\n\n")

      // 保留最后一个不完整的事件（可能没有以 \n\n 结尾）
      buffer = parts.pop() || ""

      // 处理完整的事件
      for (const part of parts) {
        if (part.trim()) {
          processSSEEvent(part, onEvent, onError)
        }
      }
    }
  } catch (error) {
    // 忽略 AbortError，这是正常的连接中止
    if (error instanceof Error && error.name === "AbortError") {
      return
    }
    if (onError) {
      onError(error instanceof Error ? error : new Error("读取流失败"))
    }
  } finally {
    try {
      reader.releaseLock()
    } catch {
      // 忽略 releaseLock 错误，可能已经被释放
    }
  }
}

/**
 * 处理单个 SSE 事件
 */
function processSSEEvent(
  eventText: string,
  onEvent: (event: MessageEvent) => void,
  onError?: (error: Error) => void,
): void {
  let eventType = "message"
  let eventData = ""
  let eventId = ""

  const lines = eventText.split("\n")

  for (const line of lines) {
    if (line.startsWith("event:")) {
      eventType = line.slice(6).trim()
    } else if (line.startsWith("data:")) {
      // 支持多行数据
      const dataLine = line.slice(5)
      if (eventData) {
        eventData += "\n" + dataLine
      } else {
        eventData = dataLine
      }
    } else if (line.startsWith("id:")) {
      eventId = line.slice(3).trim()
    }
    // 忽略其他行（如 retry:、comment 等）
  }

  if (eventData) {
    try {
      const data = JSON.parse(eventData.trim())
      onEvent(
        new MessageEvent(eventType, {
          data,
          lastEventId: eventId,
        }),
      )
    } catch (error) {
      if (onError) {
        onError(
          error instanceof Error
            ? error
            : new Error(`解析 SSE 数据失败: ${eventData}`),
        )
      }
    }
  }
}

/**
 * 处理 SSE 缓冲区（用于处理流结束时的剩余数据）
 */
function processSSEBuffer(
  buffer: string,
  onEvent: (event: MessageEvent) => void,
  onError?: (error: Error) => void,
): void {
  const events = buffer.split("\n\n").filter((e) => e.trim())
  for (const event of events) {
    processSSEEvent(event, onEvent, onError)
  }
}
