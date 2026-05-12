import assert from 'node:assert/strict'
import test from 'node:test'

import {
  getUserMessageTextClassName,
  normalizeUserMessageText,
} from '../src/lib/user-message-text'

test('normalizeUserMessageText should preserve user-authored layout', () => {
  const input = '1. 第一项\n\n   子项\n  行首空格\n\t制表符'

  assert.equal(normalizeUserMessageText(input), input)
})

test('normalizeUserMessageText should normalize CRLF without trimming or collapsing spaces', () => {
  const input = '  1. 第一项\r\n\r\n   子项\r  3. 第三项  '

  assert.equal(normalizeUserMessageText(input), '  1. 第一项\n\n   子项\n  3. 第三项  ')
})

test('getUserMessageTextClassName should use plain-text preserving whitespace rules', () => {
  const className = getUserMessageTextClassName()

  assert.equal(className.includes('whitespace-break-spaces'), true)
  assert.equal(className.includes('break-words'), true)
  assert.equal(className.includes('[tab-size:2]'), true)
})
