import type { BrowserToolContent, ToolEvent } from './api/types'

export type BrowserPreviewData = {
  screenshot: string
  url: string
  title: string
  pageType: string
  degradeReason: string
  matchedLinkText: string
  matchedLinkUrl: string
  matchedLinkSelector: string
  matchedLinkIndex: number | null
}

function getStringField(record: Record<string, unknown>, key: string): string {
  const value = record[key]
  return typeof value === 'string' ? value : ''
}

function getNumberField(record: Record<string, unknown>, key: string): number | null {
  const value = record[key]
  return typeof value === 'number' ? value : null
}

export function isBrowserToolContent(content: unknown): content is BrowserToolContent {
  return Boolean(content) && typeof content === 'object' && !Array.isArray(content)
}

export function getBrowserPreviewData(tool: ToolEvent): BrowserPreviewData {
  const rawContent = isBrowserToolContent(tool.content) ? tool.content as Record<string, unknown> : null
  const rawArgs = tool.args && typeof tool.args === 'object' ? tool.args : {}

  const contentUrl = rawContent ? getStringField(rawContent, 'url') : ''
  const argUrl = typeof rawArgs.url === 'string' ? rawArgs.url : ''

  return {
    screenshot: rawContent ? getStringField(rawContent, 'screenshot') : '',
    url: contentUrl || argUrl,
    title: rawContent ? getStringField(rawContent, 'title') : '',
    pageType: rawContent ? getStringField(rawContent, 'page_type') : '',
    degradeReason: rawContent ? getStringField(rawContent, 'degrade_reason') : '',
    matchedLinkText: rawContent ? getStringField(rawContent, 'matched_link_text') : '',
    matchedLinkUrl: rawContent ? getStringField(rawContent, 'matched_link_url') : '',
    matchedLinkSelector: rawContent ? getStringField(rawContent, 'matched_link_selector') : '',
    matchedLinkIndex: rawContent ? getNumberField(rawContent, 'matched_link_index') : null,
  }
}
