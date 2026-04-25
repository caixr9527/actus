import assert from 'node:assert/strict'
import test from 'node:test'

import {
  formatJsonPreview,
  parseDelimitedPreview,
  resolveFilePreviewType,
} from '../src/lib/file-preview'

test('resolveFilePreviewType should classify common previewable formats', () => {
  assert.deepEqual(resolveFilePreviewType('.md'), { kind: 'markdown', binary: false })
  assert.deepEqual(resolveFilePreviewType('json'), { kind: 'json', binary: false })
  assert.deepEqual(resolveFilePreviewType('csv'), { kind: 'csv', binary: false })
  assert.deepEqual(resolveFilePreviewType('pdf'), { kind: 'pdf', binary: true })
  assert.deepEqual(resolveFilePreviewType('mp4'), { kind: 'video', binary: true })
  assert.deepEqual(resolveFilePreviewType('wav'), { kind: 'audio', binary: true })
  assert.deepEqual(resolveFilePreviewType('unknown'), { kind: 'unsupported', binary: true })
})

test('resolveFilePreviewType should use content type fallback', () => {
  assert.deepEqual(resolveFilePreviewType('', 'image/png'), { kind: 'image', binary: true })
  assert.deepEqual(resolveFilePreviewType('', 'application/pdf'), { kind: 'pdf', binary: true })
  assert.deepEqual(resolveFilePreviewType('', 'text/plain'), { kind: 'text', binary: false })
})

test('formatJsonPreview should pretty print valid json and keep invalid text', () => {
  assert.equal(formatJsonPreview('{"a":1}'), '{\n  "a": 1\n}')
  assert.equal(formatJsonPreview('{broken'), '{broken')
})

test('parseDelimitedPreview should parse csv and cap displayed rows', () => {
  assert.deepEqual(parseDelimitedPreview('a,b\n1,2', ','), [['a', 'b'], ['1', '2']])
  assert.equal(parseDelimitedPreview(Array.from({ length: 250 }, (_, i) => `${i},x`).join('\n'), ',').length, 200)
})
