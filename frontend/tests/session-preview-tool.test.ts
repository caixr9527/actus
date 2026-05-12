import assert from 'node:assert/strict'
import test from 'node:test'

import type { ToolEvent } from '../src/lib/api/types'
import type { TimelineItem } from '../src/lib/session-events'
import { resolvePreviewToolFromTimeline } from '../src/lib/session-preview-tool'

const runtime = {
  session_id: 'session-1',
  run_id: 'run-1',
  status_after_event: null,
  current_step_id: null,
  source_event_id: 'evt-1',
  cursor_event_id: 'evt-1',
  durability: 'persistent',
  visibility: 'timeline',
} as const

function buildToolEvent(toolCallId: string, status: 'calling' | 'called', content?: unknown): ToolEvent {
  return {
    runtime,
    tool_call_id: toolCallId,
    name: 'mcp',
    function: 'mcp_demo',
    args: {},
    status,
    ...(content !== undefined ? { content } : {}),
  } as ToolEvent
}

test('resolvePreviewToolFromTimeline should prefer latest called tool event in step tools', () => {
  const calling = buildToolEvent('call-1', 'calling')
  const called = buildToolEvent('call-1', 'called', { result: 'ok' })
  const timeline: TimelineItem[] = [
    {
      kind: 'step',
      id: 'step-1',
      data: {
        id: 'step-1',
        description: 'demo',
        status: 'running',
      } as unknown as { id: string; description: string; status: string },
      tools: [calling, called],
    } as TimelineItem,
  ]

  const resolved = resolvePreviewToolFromTimeline(calling, timeline)

  assert.equal((resolved as { status?: string } | null)?.status, 'called')
  assert.deepEqual((resolved as { content?: unknown } | null)?.content, { result: 'ok' })
})

test('resolvePreviewToolFromTimeline should fallback to original preview tool when tool_call_id missing', () => {
  const previewTool = {
    runtime,
    name: 'message',
    function: 'message_notify_user',
    args: {},
    status: 'called',
    content: { text: 'done' },
  } as ToolEvent

  const resolved = resolvePreviewToolFromTimeline(previewTool, [])
  assert.equal(resolved, previewTool)
})
