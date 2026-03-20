import type { SessionStatus } from '@/lib/api/types'

export type RetryPolicy = {
  maxRetries: number
  baseDelayMs: number
  maxDelayMs: number
}

export function computeRetryDelayMs(retryCount: number, policy: RetryPolicy): number {
  const safeRetryCount = Math.max(0, retryCount)
  return Math.min(
    policy.baseDelayMs * Math.pow(2, safeRetryCount),
    policy.maxDelayMs,
  )
}

export function canRetry(retryCount: number, policy: RetryPolicy): boolean {
  return retryCount < policy.maxRetries
}

export function shouldStartEmptySessionStream(
  status: SessionStatus | null | undefined,
  isSendingMessage: boolean,
  skipEmptyStream: boolean,
): boolean {
  if (!status) return false
  if (status === 'completed') return false
  if (isSendingMessage) return false
  if (skipEmptyStream) return false
  return true
}
