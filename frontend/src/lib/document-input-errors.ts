import { ApiError } from './api/fetch'
import type { MessageParams, Translate } from './i18n/types'
import { translateRuntime } from './i18n/runtime'

const DOCUMENT_INPUT_ERROR_KEYS = new Set([
  'error.document_input.unsupported_media_image',
  'error.document_input.unsupported_media_audio',
  'error.document_input.unsupported_media_video',
  'error.document_input.unsupported_binary',
])

const DOCUMENT_INPUT_REASON_CODES = new Set([
  'unsupported_media_image',
  'unsupported_media_audio',
  'unsupported_media_video',
  'unsupported_binary',
])

type Translator = Translate | ((key: string, params?: MessageParams) => string)

function resolveTranslator(translate?: Translator): Translator {
  return translate ?? translateRuntime
}

export function isUnsupportedDocumentInputError(error: unknown): error is ApiError {
  if (!(error instanceof ApiError)) {
    return false
  }
  if (error.errorKey && DOCUMENT_INPUT_ERROR_KEYS.has(error.errorKey)) {
    return true
  }
  const reasonCode = error.errorParams?.reason_code
  return typeof reasonCode === 'string' && DOCUMENT_INPUT_REASON_CODES.has(reasonCode)
}

export function getDocumentInputErrorMessage(error: unknown, translate?: Translator): string | null {
  if (!isUnsupportedDocumentInputError(error)) {
    return null
  }
  return resolveTranslator(translate)('apiErrors.documentInput.unsupportedTaskInput')
}
