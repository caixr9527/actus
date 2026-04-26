import type { ToolEvent } from '@/lib/api/types'
import type { TimelineItem } from '@/lib/session-events'

/**
 * 从 timeline 中解析 previewTool 的最新版本。
 *
 * 说明：
 * - 同一 tool_call_id 在时间线上可能同时存在 calling/called 两条记录；
 * - 这里按“从后往前”匹配，保证优先返回最新态（通常是 called）。
 */
export function resolvePreviewToolFromTimeline(
  previewTool: ToolEvent | null,
  timeline: TimelineItem[],
): ToolEvent | null {
  if (!previewTool) return null
  const id = (previewTool as { tool_call_id?: string }).tool_call_id
  if (!id) return previewTool

  for (let timelineIndex = timeline.length - 1; timelineIndex >= 0; timelineIndex--) {
    const item = timeline[timelineIndex]
    if (item.kind === 'tool' && (item.data as { tool_call_id?: string }).tool_call_id === id) {
      return item.data
    }
    if (item.kind === 'step') {
      for (let toolIndex = item.tools.length - 1; toolIndex >= 0; toolIndex--) {
        const tool = item.tools[toolIndex]
        if ((tool as { tool_call_id?: string }).tool_call_id === id) return tool
      }
    }
  }
  return previewTool
}
