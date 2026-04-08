import assert from 'node:assert/strict'
import test from 'node:test'

import type { ToolEvent } from '../src/lib/api/types'
import { getBrowserPreviewData } from '../src/lib/browser-tool-preview'
import { getFriendlyToolLabel } from '../src/components/tool-use/utils'

test('getBrowserPreviewData should expose browser degrade and matched target fields', () => {
  const tool = {
    name: 'browser',
    function: 'browser_find_link_by_text',
    args: { text: 'Execution Model' },
    content: {
      url: 'https://example.com/docs/runtime',
      title: 'Runtime Docs',
      page_type: 'listing',
      screenshot: 'https://cdn.example.com/browser.png',
      degrade_reason: 'browser_extract_cards_failed',
      matched_link_text: 'Execution Model',
      matched_link_url: 'https://example.com/docs/execution',
      matched_link_selector: "[data-manus-id='manus-element-0']",
      matched_link_index: 0,
    },
  } as ToolEvent

  const preview = getBrowserPreviewData(tool)

  assert.deepEqual(preview, {
    screenshot: 'https://cdn.example.com/browser.png',
    url: 'https://example.com/docs/runtime',
    title: 'Runtime Docs',
    pageType: 'listing',
    degradeReason: 'browser_extract_cards_failed',
    matchedLinkText: 'Execution Model',
    matchedLinkUrl: 'https://example.com/docs/execution',
    matchedLinkSelector: "[data-manus-id='manus-element-0']",
    matchedLinkIndex: 0,
  })
})

test('getFriendlyToolLabel should describe browser high level routing actions', () => {
  const findLinkTool = {
    name: 'browser',
    function: 'browser_find_link_by_text',
    args: { text: 'Execution Model' },
  } as ToolEvent
  const clickTool = {
    name: 'browser',
    function: 'browser_click',
    args: { index: 0 },
  } as ToolEvent

  assert.equal(
    getFriendlyToolLabel(findLinkTool, 'zh-CN'),
    '正在定位目标链接 Execution Model',
  )
  assert.equal(
    getFriendlyToolLabel(clickTool, 'zh-CN'),
    '正在点击页面元素 #0',
  )
})
