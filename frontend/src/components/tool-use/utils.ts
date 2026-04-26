import type { ToolEvent } from '@/lib/api/types'
import type { AppLocale } from '@/lib/i18n'

export type ToolKind =
  | 'message'
  | 'bash'
  | 'file'
  | 'search'
  | 'browser'
  | 'mcp'
  | 'a2a'
  | 'default'

export function getArg(args: Record<string, unknown>, ...keys: string[]): string {
  if (!args || typeof args !== 'object') return ''
  for (const k of keys) {
    const v = args[k]
    if (typeof v === 'string') return v
  }
  return ''
}

export function truncate(s: string, max: number): string {
  return s.length > max ? s.slice(0, max) + '…' : s
}

export function getToolKind(data: ToolEvent | null | undefined): ToolKind {
  if (!data) return 'default'
  const name = (data.name ?? '').toLowerCase()
  const fn = (data.function ?? '').toLowerCase()

  if (data.function === 'message_notify_user' || data.function === 'message_ask_user') {
    return 'message'
  }
  if (name === 'shell' || name.includes('bash') || fn === 'shell_execute' || fn === 'run' || fn === 'execute' || fn === 'run_command') {
    return 'bash'
  }
  if (name === 'file' || name.includes('file')) {
    return 'file'
  }
  if (name === 'mcp' || name.startsWith('mcp_')) {
    return 'mcp'
  }
  // 剔除原先fn中关于search_web的包含匹配, 避免mcp工具中也存在搜索工具，识别错误
  if (fn === 'search_web' || name === 'search') {
    return 'search'
  }
  if (name === 'browser' || name.includes('browser') || fn.startsWith('browser_')) {
    return 'browser'
  }
  if (name === 'a2a' || name.includes('a2a')) {
    return 'a2a'
  }
  return 'default'
}

/**
 * 根据工具 name/function 和 args 生成事件列表中的人性化提示
 */
export function getFriendlyToolLabel(
  data: ToolEvent | null | undefined,
  locale: AppLocale = 'zh-CN',
): string {
  const isEnglish = locale === 'en-US'
  if (!data) return '—'
  const name = (data.name ?? '').toLowerCase()
  const fn = (data.function ?? '').toLowerCase()
  const args = data.args && typeof data.args === 'object' ? data.args : {}

  if (data.function === 'message_notify_user' || data.function === 'message_ask_user') {
    const text = typeof args.text === 'string' ? args.text : ''
    return text || '—'
  }

  const filepath = getArg(args, 'filepath', 'path', 'pathname')
  const dirPath = getArg(args, 'dir_path', 'directory', 'dir')
  const query = getArg(args, 'query', 'q')
  const command = getArg(args, 'command', 'cmd', 'script')
  const url = getArg(args, 'url', 'href', 'link')
  const key = getArg(args, 'key')
  const text = getArg(args, 'text')
  const index = typeof args.index === 'number' ? String(args.index) : ''

  if (name === 'file') {
    switch (fn) {
      case 'read_file':
        return filepath
          ? isEnglish ? `Reading file ${truncate(filepath, 60)}` : `正在读取文件 ${truncate(filepath, 60)}`
          : isEnglish ? 'Reading file' : '正在读取文件'
      case 'write_file':
        return filepath
          ? isEnglish ? `Writing file ${truncate(filepath, 60)}` : `正在写入文件 ${truncate(filepath, 60)}`
          : isEnglish ? 'Writing file' : '正在写入文件'
      case 'replace_in_file':
        return filepath
          ? isEnglish ? `Replacing in file ${truncate(filepath, 60)}` : `正在替换文件内容 ${truncate(filepath, 60)}`
          : isEnglish ? 'Replacing file content' : '正在替换文件内容'
      case 'search_in_file':
        return filepath
          ? isEnglish ? `Searching in file ${truncate(filepath, 60)}` : `正在在文件中搜索 ${truncate(filepath, 60)}`
          : isEnglish ? 'Searching in file' : '正在在文件中搜索'
      case 'find_files':
        return dirPath
          ? isEnglish ? `Finding files in ${truncate(dirPath, 60)}` : `正在查找文件 ${truncate(dirPath, 60)}`
          : isEnglish ? 'Finding files' : '正在查找文件'
      case 'list_files':
        return dirPath
          ? isEnglish ? `Listing directory ${truncate(dirPath, 60)}` : `正在列出目录 ${truncate(dirPath, 60)}`
          : isEnglish ? 'Listing directory' : '正在列出目录'
      default:
        return filepath
          ? isEnglish ? `Accessing file ${truncate(filepath, 60)}` : `正在访问文件 ${truncate(filepath, 60)}`
          : dirPath
            ? isEnglish ? `Accessing directory ${truncate(dirPath, 60)}` : `正在访问目录 ${truncate(dirPath, 60)}`
            : isEnglish ? 'Accessing files' : '正在访问文件'
    }
  }

  if (name === 'browser' || fn.startsWith('browser_')) {
    switch (fn) {
      case 'browser_view':
        return isEnglish ? 'Viewing current page' : '正在查看当前页面'
      case 'browser_read_current_page_structured':
        return isEnglish ? 'Analyzing current page structure' : '正在分析当前页面结构'
      case 'browser_extract_main_content':
        return isEnglish ? 'Extracting main page content' : '正在提取页面正文'
      case 'browser_extract_cards':
        return isEnglish ? 'Extracting candidate cards' : '正在提取候选卡片'
      case 'browser_find_link_by_text':
        return text
          ? isEnglish ? `Locating target link ${truncate(text, 60)}` : `正在定位目标链接 ${truncate(text, 60)}`
          : isEnglish ? 'Locating target link' : '正在定位目标链接'
      case 'browser_find_actionable_elements':
        return isEnglish ? 'Finding actionable elements' : '正在识别可交互元素'
      case 'browser_navigate':
        return url
          ? isEnglish ? `Opening page ${truncate(url, 80)}` : `正在打开页面 ${truncate(url, 80)}`
          : isEnglish ? 'Opening page' : '正在打开页面'
      case 'browser_restart':
        return url
          ? isEnglish ? `Restarting browser and opening ${truncate(url, 80)}` : `正在重启浏览器并打开 ${truncate(url, 80)}`
          : isEnglish ? 'Restarting browser' : '正在重启浏览器'
      case 'browser_click':
        return index
          ? isEnglish ? `Clicking page element #${index}` : `正在点击页面元素 #${index}`
          : isEnglish ? 'Clicking page element' : '正在点击页面元素'
      case 'browser_input':
        return isEnglish ? 'Typing input' : '正在输入内容'
      case 'browser_move_mouse':
        return isEnglish ? 'Moving mouse' : '正在移动鼠标'
      case 'browser_press_key':
        return key
          ? isEnglish ? `Pressing key ${key}` : `正在按键 ${key}`
          : isEnglish ? 'Pressing key' : '正在按键'
      case 'browser_select_option':
        return isEnglish ? 'Selecting option' : '正在选择下拉选项'
      case 'browser_scroll_up':
        return isEnglish ? 'Scrolling up' : '正在向上滚动页面'
      case 'browser_scroll_down':
        return isEnglish ? 'Scrolling down' : '正在向下滚动页面'
      case 'browser_console_exec':
        return isEnglish ? 'Running console script' : '正在执行控制台脚本'
      case 'browser_console_view':
        return isEnglish ? 'Viewing console output' : '正在查看控制台输出'
      default:
        return url
          ? isEnglish ? `Opening page ${truncate(url, 80)}` : `正在打开页面 ${truncate(url, 80)}`
          : isEnglish ? 'Browsing page' : '正在使用浏览器访问页面'
    }
  }

  if (name === 'search' || fn === 'search_web' || fn.includes('search_web')) {
    if (fn === 'fetch_page') {
      return url
        ? isEnglish ? `Reading page ${truncate(url, 80)}` : `正在读取页面 ${truncate(url, 80)}`
        : isEnglish ? 'Reading page' : '正在读取页面'
    }
    return query
      ? isEnglish ? `Searching ${truncate(query, 60)}` : `正在搜索 ${truncate(query, 60)}`
      : isEnglish ? 'Searching' : '正在搜索'
  }

  if (name === 'shell') {
    switch (fn) {
      case 'shell_execute':
        return command
          ? isEnglish ? `Running command ${truncate(command, 60)}` : `正在执行命令 ${truncate(command, 60)}`
          : isEnglish ? 'Running command' : '正在执行命令'
      case 'shell_read_output':
        return isEnglish ? 'Reading command output' : '正在查看命令输出'
      case 'shell_wait':
        return isEnglish ? 'Waiting for command' : '正在等待命令完成'
      case 'shell_write_input':
        return isEnglish ? 'Sending input to command' : '正在向命令输入内容'
      case 'shell_kill_process':
        return isEnglish ? 'Killing process' : '正在终止进程'
      default:
        return command
          ? isEnglish ? `Running command ${truncate(command, 60)}` : `正在执行命令 ${truncate(command, 60)}`
          : isEnglish ? 'Running command' : '正在执行命令'
    }
  }

  if (name.includes('bash') || fn === 'run' || fn === 'execute' || fn === 'run_command') {
    const cmd = command || (typeof args.input === 'string' ? args.input : '')
    return cmd
      ? isEnglish ? `Running command ${truncate(cmd, 60)}` : `正在执行命令 ${truncate(cmd, 60)}`
      : isEnglish ? 'Running command' : '正在执行命令'
  }

  if (name === 'a2a') {
    switch (fn) {
      case 'get_remote_agent_cards':
        return isEnglish ? 'Fetching remote agents' : '正在获取远程 Agent 列表'
      case 'call_remote_agent':
        return query
          ? isEnglish ? `Calling remote agent: ${truncate(query, 40)}` : `正在调用远程 Agent：${truncate(query, 40)}`
          : isEnglish ? 'Calling remote agent' : '正在调用远程 Agent'
      default:
        return isEnglish ? 'Calling agent' : '正在调用 Agent'
    }
  }

  if (name === 'mcp' || name.startsWith('mcp_')) {
    if (fn.includes('search_web') || fn.includes('search')) {
      return query
        ? isEnglish ? `Searching ${truncate(query, 60)}` : `正在搜索 ${truncate(query, 60)}`
        : isEnglish ? 'Searching' : '正在搜索'
    }
    return isEnglish ? 'Executing via MCP service' : '正在通过 MCP 服务执行操作'
  }

  return isEnglish ? 'Executing operation' : '正在执行操作'
}
