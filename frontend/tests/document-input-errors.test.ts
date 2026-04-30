import assert from 'node:assert/strict'
import test from 'node:test'

import { ApiError } from '../src/lib/api/fetch'
import {
  getDocumentInputErrorMessage,
  isUnsupportedDocumentInputError,
} from '../src/lib/document-input-errors'
import { getApiErrorMessage } from '../src/lib/api'

const translate = (key: string) => key === 'apiErrors.documentInput.unsupportedTaskInput'
  ? '不支持该类型作为任务输入'
  : key

test('document input media errors should use one stable user-facing message', () => {
  const errors = [
    new ApiError(400, 'bad', null, 'error.document_input.unsupported_media_image', {
      reason_code: 'unsupported_media_image',
    }),
    new ApiError(400, 'bad', null, 'error.document_input.unsupported_media_audio', {
      reason_code: 'unsupported_media_audio',
    }),
    new ApiError(400, 'bad', null, 'error.document_input.unsupported_media_video', {
      reason_code: 'unsupported_media_video',
    }),
    new ApiError(400, 'bad', null, 'error.document_input.unsupported_binary', {
      reason_code: 'unsupported_binary',
    }),
  ]

  for (const error of errors) {
    assert.equal(isUnsupportedDocumentInputError(error), true)
    assert.equal(getDocumentInputErrorMessage(error, translate), '不支持该类型作为任务输入')
    assert.equal(getApiErrorMessage(error, 'fallback', translate), '不支持该类型作为任务输入')
  }
})

test('document input helper should also honor reason_code when error_key is absent', () => {
  const error = new ApiError(400, 'bad', null, null, {
    reason_code: 'unsupported_media_image',
  })

  assert.equal(isUnsupportedDocumentInputError(error), true)
  assert.equal(getDocumentInputErrorMessage(error, translate), '不支持该类型作为任务输入')
})
