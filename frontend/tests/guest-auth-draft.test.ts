import assert from 'node:assert/strict'
import test from 'node:test'

import {
  clearGuestPendingAction,
  clearGuestPendingMessage,
  loadGuestPendingAction,
  loadGuestPendingMessage,
  saveGuestPendingAction,
  saveGuestPendingMessage,
} from '../src/lib/guest-auth-draft'

class MemoryStorage {
  private readonly data = new Map<string, string>()

  getItem(key: string): string | null {
    return this.data.has(key) ? this.data.get(key)! : null
  }

  setItem(key: string, value: string): void {
    this.data.set(key, value)
  }

  removeItem(key: string): void {
    this.data.delete(key)
  }
}

function installWindow(storage: MemoryStorage): void {
  ;(globalThis as { window?: unknown }).window = {
    sessionStorage: storage,
  }
}

test.afterEach(() => {
  delete (globalThis as { window?: unknown }).window
})

test('guest draft helpers should save and load pending message', () => {
  const storage = new MemoryStorage()
  installWindow(storage)

  saveGuestPendingMessage('整理本周项目计划')

  assert.equal(loadGuestPendingMessage(), '整理本周项目计划')

  clearGuestPendingMessage()
  assert.equal(loadGuestPendingMessage(), '')
})

test('guest draft helpers should save and load pending action', () => {
  const storage = new MemoryStorage()
  installWindow(storage)

  saveGuestPendingAction('upload')
  assert.equal(loadGuestPendingAction(), 'upload')

  clearGuestPendingAction()
  assert.equal(loadGuestPendingAction(), null)
})
