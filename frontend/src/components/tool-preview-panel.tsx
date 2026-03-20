"use client"

import { useMemo } from "react"
import Image from "next/image"
import type { ToolEvent } from "@/lib/api/types"
import {
  getToolKind,
  getFriendlyToolLabel,
  getArg,
} from "@/components/tool-use/utils"
import type { ToolKind } from "@/components/tool-use/utils"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Button } from "@/components/ui/button"
import { useI18n, type Translate } from "@/lib/i18n"
import {
  Maximize2,
  Monitor,
  Play,
  Terminal,
  Globe,
  Search,
  FileSearch,
  Wrench,
  Bot,
  Sparkles,
} from "lucide-react"

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

export interface ToolPreviewPanelProps {
  tool: ToolEvent
  onClose: () => void
  onJumpToLatest?: () => void
  onOpenVNC?: () => void
}

type ConsoleRecord = { ps1: string; command: string; output: string }

type SearchResultItem = { url: string; title: string; snippet: string }

/* ------------------------------------------------------------------ */
/*  Content extractors                                                 */
/* ------------------------------------------------------------------ */

function getToolContent(tool: ToolEvent): Record<string, unknown> | null {
  const c = tool.content
  if (c && typeof c === "object" && !Array.isArray(c))
    return c as Record<string, unknown>
  return null
}

function getToolDescription(kind: ToolKind, t: Translate): string {
  const map: Record<ToolKind, string> = {
    bash: t("toolPreview.kind.bash"),
    browser: t("toolPreview.kind.browser"),
    search: t("toolPreview.kind.search"),
    file: t("toolPreview.kind.file"),
    mcp: t("toolPreview.kind.mcp"),
    a2a: t("toolPreview.kind.a2a"),
    message: t("toolPreview.kind.message"),
    default: t("toolPreview.kind.default"),
  }
  return map[kind]
}

function renderToolIcon(kind: ToolKind) {
  switch (kind) {
    case "bash":
      return <Terminal size={14} className="flex-shrink-0 text-gray-500" />
    case "browser":
      return <Globe size={14} className="flex-shrink-0 text-gray-500" />
    case "search":
      return <Search size={14} className="flex-shrink-0 text-gray-500" />
    case "file":
      return <FileSearch size={14} className="flex-shrink-0 text-gray-500" />
    case "mcp":
      return <Wrench size={14} className="flex-shrink-0 text-gray-500" />
    case "a2a":
      return <Bot size={14} className="flex-shrink-0 text-gray-500" />
    case "message":
    case "default":
      return <Monitor size={14} className="flex-shrink-0 text-gray-500" />
    default:
      return <Monitor size={14} className="flex-shrink-0 text-gray-500" />
  }
}

/* ------------------------------------------------------------------ */
/*  Jump-to-latest overlay button                                      */
/* ------------------------------------------------------------------ */

function JumpToLatestButton({
  onClick,
  label,
}: {
  onClick: () => void
  label: string
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-white/90 backdrop-blur text-sm text-gray-700 hover:bg-white shadow-md border border-gray-200 transition-colors cursor-pointer"
    >
      <Play size={12} className="fill-current" />
      <span>{label}</span>
    </button>
  )
}

/* ------------------------------------------------------------------ */
/*  Sub-previews                                                       */
/* ------------------------------------------------------------------ */

function ShellPreview({ tool, t }: { tool: ToolEvent; t: Translate }) {
  const content = getToolContent(tool)
  const consoleData = content?.console
  const sessionId = getArg(tool.args, "session_id")

  const records: ConsoleRecord[] = useMemo(() => {
    if (Array.isArray(consoleData)) return consoleData as ConsoleRecord[]
    return []
  }, [consoleData])

  return (
    <div className="flex flex-col gap-3 p-4 h-full">
      <div className="flex-1 rounded-lg overflow-hidden border border-gray-700 bg-[#1e1e1e] flex flex-col min-h-0">
        <div className="text-center text-xs text-gray-400 py-1.5 bg-[#2d2d2d] border-b border-gray-700 flex-shrink-0">
          {sessionId || t("toolPreview.shellSessionDefault")}
        </div>
        <ScrollArea className="flex-1">
          <div className="p-4 font-mono text-sm leading-relaxed">
            {records.length > 0 ? (
              records.map((rec, i) => (
                <div key={i} className="mb-2">
                  <div>
                    <span className="text-green-400">{rec.ps1}</span>{" "}
                    <span className="text-white">{rec.command}</span>
                  </div>
                  {rec.output && (
                    <pre className="text-gray-300 whitespace-pre-wrap break-words mt-0.5">
                      {rec.output}
                    </pre>
                  )}
                </div>
              ))
            ) : (
              <span className="text-gray-500">{t("toolPreview.shellWaitingOutput")}</span>
            )}
          </div>
        </ScrollArea>
      </div>
    </div>
  )
}

function BrowserPreview({
  tool,
  onOpenVNC,
  t,
}: {
  tool: ToolEvent
  onOpenVNC?: () => void
  t: Translate
}) {
  const content = getToolContent(tool)
  const screenshot =
    typeof content?.screenshot === "string" ? content.screenshot : null
  const url = getArg(tool.args, "url", "href", "link")

  return (
    <div className="flex flex-col gap-3 p-4 h-full">
      {url && (
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-100 border text-sm text-gray-600 flex-shrink-0">
          <Globe size={14} className="text-gray-400 flex-shrink-0" />
          <span className="truncate">{url}</span>
        </div>
      )}
      <div className="flex-1 rounded-lg overflow-hidden border min-h-0 relative">
        {screenshot ? (
          <ScrollArea className="h-full">
            <Image
              src={screenshot}
              alt={t("toolPreview.browserScreenshotAlt")}
              width={1280}
              height={720}
              unoptimized
              className="w-full h-auto"
            />
          </ScrollArea>
        ) : (
          <div className="flex items-center justify-center h-full text-sm text-gray-500">
            {t("toolPreview.browserWaitingScreenshot")}
          </div>
        )}
        {onOpenVNC && (
          <button
            type="button"
            onClick={onOpenVNC}
            className="absolute bottom-3 right-3 w-9 h-9 rounded-full bg-gray-800/80 text-white flex items-center justify-center shadow-lg hover:bg-gray-700 transition-colors cursor-pointer z-10"
            aria-label={t("toolPreview.openRemoteDesktopAria")}
          >
            <Sparkles size={16} />
          </button>
        )}
      </div>
    </div>
  )
}

function SearchPreview({ tool, t }: { tool: ToolEvent; t: Translate }) {
  const content = getToolContent(tool)
  const rawResults = content?.results

  const results: SearchResultItem[] = useMemo(() => {
    if (Array.isArray(rawResults)) return rawResults as SearchResultItem[]
    return []
  }, [rawResults])

  const query = getArg(tool.args, "query", "q")

  return (
    <ScrollArea className="h-full">
      <div className="flex flex-col gap-1 p-4">
        {query && (
          <div className="text-sm text-gray-500 mb-3">
            {t("toolPreview.searchSummary", { query, count: results.length })}
          </div>
        )}
        {results.length > 0 ? (
          results.map((item, i) => (
            <a
              key={i}
              href={item.url}
              target="_blank"
              rel="noopener noreferrer"
              className="block p-3 rounded-lg hover:bg-gray-50 transition-colors group"
            >
              <div className="text-xs text-green-700 truncate mb-0.5">
                {item.url}
              </div>
              <div className="text-sm font-medium text-blue-700 group-hover:underline mb-1 line-clamp-1">
                {item.title}
              </div>
              {item.snippet && (
                <div className="text-xs text-gray-600 line-clamp-2">
                  {item.snippet}
                </div>
              )}
            </a>
          ))
        ) : (
          <div className="text-sm text-gray-500 text-center py-8">
            {t("toolPreview.searchNoResults")}
          </div>
        )}
      </div>
    </ScrollArea>
  )
}

function FileToolPreview({ tool, t }: { tool: ToolEvent; t: Translate }) {
  const content = getToolContent(tool)
  const fileContent =
    typeof content?.content === "string" ? content.content : null
  const filepath = getArg(tool.args, "filepath", "path", "pathname")

  return (
    <div className="flex flex-col gap-3 p-4 h-full">
      <div className="flex-1 rounded-lg overflow-hidden border border-gray-700 bg-[#1e1e1e] flex flex-col min-h-0">
        {filepath && (
          <div className="text-center text-xs text-gray-400 py-1.5 bg-[#2d2d2d] border-b border-gray-700 flex-shrink-0 truncate px-4">
            {filepath}
          </div>
        )}
        <ScrollArea className="flex-1">
          <pre className="p-4 font-mono text-sm text-gray-300 whitespace-pre-wrap break-words leading-relaxed">
            {fileContent ?? t("toolPreview.fileWaitingContent")}
          </pre>
        </ScrollArea>
      </div>
    </div>
  )
}

function MCPPreview({ tool, t }: { tool: ToolEvent; t: Translate }) {
  const content = getToolContent(tool)
  const result = content?.result

  return (
    <ScrollArea className="h-full">
      <div className="flex flex-col gap-4 p-4">
        <div className="flex flex-col gap-1">
          <div className="text-xs text-gray-500 uppercase tracking-wide">
            {t("toolPreview.section.toolInfo")}
          </div>
          <div className="rounded-lg border bg-gray-50 p-3 text-sm">
            <div>
              <span className="text-gray-500">{t("toolPreview.label.name")}</span>
              <span className="text-gray-800">{tool.name}</span>
            </div>
            <div>
              <span className="text-gray-500">{t("toolPreview.label.function")}</span>
              <span className="text-gray-800">{tool.function}</span>
            </div>
            {Object.keys(tool.args).length > 0 && (
              <div className="mt-1">
                <span className="text-gray-500">{t("toolPreview.label.args")}</span>
                <pre className="text-xs text-gray-700 mt-1 whitespace-pre-wrap break-words">
                  {JSON.stringify(tool.args, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </div>
        <div className="flex flex-col gap-1">
          <div className="text-xs text-gray-500 uppercase tracking-wide">
            {t("toolPreview.section.executionResult")}
          </div>
          <div className="rounded-lg border border-gray-700 bg-[#1e1e1e] p-4">
            <pre className="font-mono text-sm text-gray-300 whitespace-pre-wrap break-words">
              {result != null
                ? typeof result === "string"
                  ? result
                  : JSON.stringify(result, null, 2)
                : t("toolPreview.waitingExecutionResult")}
            </pre>
          </div>
        </div>
      </div>
    </ScrollArea>
  )
}

function A2APreview({ tool, t }: { tool: ToolEvent; t: Translate }) {
  const content = getToolContent(tool)
  const result = content?.a2a_result

  const query = getArg(tool.args, "query", "message", "input")

  return (
    <ScrollArea className="h-full">
      <div className="flex flex-col gap-4 p-4">
        <div className="flex flex-col gap-1">
          <div className="text-xs text-gray-500 uppercase tracking-wide">
            {t("toolPreview.section.agentCallInfo")}
          </div>
          <div className="rounded-lg border bg-gray-50 p-3 text-sm">
            <div>
              <span className="text-gray-500">{t("toolPreview.label.tool")}</span>
              <span className="text-gray-800">{tool.name}</span>
            </div>
            <div>
              <span className="text-gray-500">{t("toolPreview.label.function")}</span>
              <span className="text-gray-800">{tool.function}</span>
            </div>
            {query && (
              <div>
                <span className="text-gray-500">{t("toolPreview.label.command")}</span>
                <span className="text-gray-800">{query}</span>
              </div>
            )}
          </div>
        </div>
        <div className="flex flex-col gap-1">
          <div className="text-xs text-gray-500 uppercase tracking-wide">
            {t("toolPreview.section.executionResult")}
          </div>
          <div className="rounded-lg border border-gray-700 bg-[#1e1e1e] p-4">
            <pre className="font-mono text-sm text-gray-300 whitespace-pre-wrap break-words">
              {result != null
                ? typeof result === "string"
                  ? result
                  : JSON.stringify(result, null, 2)
                : t("toolPreview.waitingExecutionResult")}
            </pre>
          </div>
        </div>
      </div>
    </ScrollArea>
  )
}

function DefaultPreview({ tool, t }: { tool: ToolEvent; t: Translate }) {
  return (
    <ScrollArea className="h-full">
      <div className="flex flex-col gap-4 p-4">
        <div className="rounded-lg border bg-gray-50 p-3 text-sm">
          <div>
            <span className="text-gray-500">{t("toolPreview.label.name")}</span>
            <span className="text-gray-800">{tool.name}</span>
          </div>
          <div>
            <span className="text-gray-500">{t("toolPreview.label.function")}</span>
            <span className="text-gray-800">{tool.function}</span>
          </div>
        </div>
        {tool.content != null && (
          <div className="rounded-lg border border-gray-700 bg-[#1e1e1e] p-4">
            <pre className="font-mono text-sm text-gray-300 whitespace-pre-wrap break-words">
              {typeof tool.content === "string"
                ? tool.content
                : JSON.stringify(tool.content, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </ScrollArea>
  )
}

/* ------------------------------------------------------------------ */
/*  Main Component                                                     */
/* ------------------------------------------------------------------ */

export function ToolPreviewPanel({
  tool,
  onClose,
  onJumpToLatest,
  onOpenVNC,
}: ToolPreviewPanelProps) {
  const { locale, t } = useI18n()
  const kind = getToolKind(tool)
  const label = getFriendlyToolLabel(tool, locale)
  const toolDesc = getToolDescription(kind, t)

  return (
    <div className="flex flex-col h-full rounded-xl bg-white shadow-xl overflow-hidden">
      {/* Header */}
      <div className="flex flex-col gap-2 px-4 py-3 border-b border-gray-200 bg-gray-50 flex-shrink-0">
        <div className="flex items-center justify-between">
          <h2 className="text-base font-semibold text-gray-900">
            {t("toolPreview.title")}
          </h2>
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={onClose}
            aria-label={t("toolPreview.closePreviewAria")}
            className="cursor-pointer"
          >
            <Maximize2 size={16} />
          </Button>
        </div>
        <div className="flex items-center gap-2 text-sm text-gray-600">
          <Monitor size={14} className="text-gray-500 flex-shrink-0" />
          <span>{t("toolPreview.usingToolPrefix")}</span>
          <span className="font-medium text-gray-800">{toolDesc}</span>
        </div>
        <div className="inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1 border border-gray-200 bg-gray-100 text-gray-700 text-xs w-fit max-w-full">
          {renderToolIcon(kind)}
          <span className="truncate">{label}</span>
        </div>
      </div>

      {/* Content with overlaid jump button */}
      <div className="flex-1 overflow-hidden relative">
        {kind === "bash" && <ShellPreview tool={tool} t={t} />}
        {kind === "browser" && (
          <BrowserPreview tool={tool} onOpenVNC={onOpenVNC} t={t} />
        )}
        {kind === "search" && <SearchPreview tool={tool} t={t} />}
        {kind === "file" && <FileToolPreview tool={tool} t={t} />}
        {kind === "mcp" && <MCPPreview tool={tool} t={t} />}
        {kind === "a2a" && <A2APreview tool={tool} t={t} />}
        {(kind === "default" || kind === "message") && (
          <DefaultPreview tool={tool} t={t} />
        )}

        {/* "跳转实时" overlaid at bottom-center */}
        {onJumpToLatest && (
          <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-10">
            <JumpToLatestButton onClick={onJumpToLatest} label={t("toolPreview.jumpToLatest")} />
          </div>
        )}
      </div>
    </div>
  )
}
