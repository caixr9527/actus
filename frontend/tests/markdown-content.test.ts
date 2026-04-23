import assert from 'node:assert/strict'
import test from 'node:test'

import {
  detectRichTextFormat,
  normalizeAutolinks,
  normalizeMarkdownForRender,
  sanitizeHtmlForRender,
} from '../src/lib/markdown-content-utils'

test('normalizeAutolinks should insert boundary space between URL and CJK chars', () => {
  const input = '访问 https://example.com，查看详情'
  const normalized = normalizeAutolinks(input)
  assert.equal(normalized, '访问 https://example.com ，查看详情')
})

test('normalizeMarkdownForRender should keep raw content when preserveLineBreaks is false', () => {
  const input = 'line1\nline2'
  const normalized = normalizeMarkdownForRender(input, false)
  assert.equal(normalized, 'line1\nline2')
})

test('normalizeMarkdownForRender should convert single line breaks to markdown hard breaks', () => {
  const input = 'line1\nline2'
  const normalized = normalizeMarkdownForRender(input, true)
  assert.equal(normalized, 'line1  \nline2')
})

test('detectRichTextFormat should prefer html for plain html payload', () => {
  const input = '<p>hello <strong>world</strong></p>'
  assert.equal(detectRichTextFormat(input), 'html')
})

test('detectRichTextFormat should keep markdown when markdown syntax exists', () => {
  const input = '# title\n<p>raw</p>'
  assert.equal(detectRichTextFormat(input), 'markdown')
})

test('sanitizeHtmlForRender should strip dangerous tags and attributes', () => {
  const input = '<script>alert(1)</script><p onclick="alert(2)">ok</p><a href="javascript:alert(3)">x</a>'
  const output = sanitizeHtmlForRender(input)
  assert.equal(output.includes('<script'), false)
  assert.equal(output.includes('onclick='), false)
  assert.equal(output.includes('javascript:'), false)
  assert.equal(output.includes('<p>ok</p>'), true)
})
