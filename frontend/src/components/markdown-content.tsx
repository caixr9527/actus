'use client'

import { useMemo } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { cn } from '@/lib/utils'
import {
  detectRichTextFormat,
  normalizeMarkdownForRender,
  sanitizeHtmlForRender,
} from '@/lib/markdown-content-utils'

export interface MarkdownContentProps {
  content: string
  className?: string
  format?: 'markdown' | 'html' | 'auto'
  tone?: 'default' | 'muted' | 'inverse' | 'amber'
  compact?: boolean
  preserveLineBreaks?: boolean
}

/**
 * remark-gfm autolink 对紧跟 CJK 字符的 URL 边界检测不准确，
 * 会将 `https://example.com，后续中文` 整段识别为链接。
 * 在 URL 与相邻 CJK 字符/标点之间插入空格修正边界。
 */
const TONE_CLASSES: Record<NonNullable<MarkdownContentProps['tone']>, {
  heading: string
  paragraph: string
  list: string
  strong: string
  codeInline: string
  codeBlock: string
  blockquote: string
  tableHead: string
  tableBody: string
  tableHeaderCell: string
  tableCell: string
  link: string
  htmlWrapper: string
}> = {
  default: {
    heading: 'text-gray-900',
    paragraph: 'text-gray-700',
    list: 'text-gray-700',
    strong: 'text-gray-900',
    codeInline: 'bg-gray-100 text-gray-800',
    codeBlock: 'bg-gray-100 text-gray-800',
    blockquote: 'border-gray-200 text-gray-600',
    tableHead: 'bg-gray-50',
    tableBody: 'divide-gray-200',
    tableHeaderCell: 'border-gray-200 text-gray-900',
    tableCell: 'border-gray-200 text-gray-700',
    link: 'text-blue-600 hover:underline',
    htmlWrapper:
      '[&_h1]:text-gray-900 [&_h2]:text-gray-900 [&_h3]:text-gray-800 [&_h4]:text-gray-800 [&_h5]:text-gray-700 [&_h6]:text-gray-700 [&_p]:text-gray-700 [&_ul]:text-gray-700 [&_ol]:text-gray-700 [&_strong]:text-gray-900 [&_blockquote]:text-gray-600 [&_blockquote]:border-gray-200 [&_a]:text-blue-600 [&_a]:hover:underline [&_code]:bg-gray-100 [&_code]:text-gray-800 [&_pre]:bg-gray-100 [&_pre]:text-gray-800 [&_pre]:rounded-md [&_pre]:p-3 [&_th]:border-gray-200 [&_th]:text-gray-900 [&_td]:border-gray-200 [&_td]:text-gray-700',
  },
  muted: {
    heading: 'text-gray-700',
    paragraph: 'text-gray-500',
    list: 'text-gray-500',
    strong: 'text-gray-700',
    codeInline: 'bg-gray-100 text-gray-700',
    codeBlock: 'bg-gray-100 text-gray-700',
    blockquote: 'border-gray-200 text-gray-500',
    tableHead: 'bg-gray-50',
    tableBody: 'divide-gray-200',
    tableHeaderCell: 'border-gray-200 text-gray-700',
    tableCell: 'border-gray-200 text-gray-500',
    link: 'text-blue-600 hover:underline',
    htmlWrapper:
      '[&_h1]:text-gray-700 [&_h2]:text-gray-700 [&_h3]:text-gray-700 [&_h4]:text-gray-700 [&_h5]:text-gray-700 [&_h6]:text-gray-700 [&_p]:text-gray-500 [&_ul]:text-gray-500 [&_ol]:text-gray-500 [&_strong]:text-gray-700 [&_blockquote]:text-gray-500 [&_blockquote]:border-gray-200 [&_a]:text-blue-600 [&_a]:hover:underline [&_code]:bg-gray-100 [&_code]:text-gray-700 [&_pre]:bg-gray-100 [&_pre]:text-gray-700 [&_pre]:rounded-md [&_pre]:p-3 [&_th]:border-gray-200 [&_th]:text-gray-700 [&_td]:border-gray-200 [&_td]:text-gray-500',
  },
  inverse: {
    heading: 'text-gray-100',
    paragraph: 'text-gray-300',
    list: 'text-gray-300',
    strong: 'text-gray-100',
    codeInline: 'bg-white/10 text-gray-100',
    codeBlock: 'bg-white/10 text-gray-200',
    blockquote: 'border-gray-600 text-gray-300',
    tableHead: 'bg-white/10',
    tableBody: 'divide-gray-700',
    tableHeaderCell: 'border-gray-700 text-gray-100',
    tableCell: 'border-gray-700 text-gray-300',
    link: 'text-sky-300 hover:underline',
    htmlWrapper:
      '[&_h1]:text-gray-100 [&_h2]:text-gray-100 [&_h3]:text-gray-200 [&_h4]:text-gray-200 [&_h5]:text-gray-300 [&_h6]:text-gray-300 [&_p]:text-gray-300 [&_ul]:text-gray-300 [&_ol]:text-gray-300 [&_strong]:text-gray-100 [&_blockquote]:text-gray-300 [&_blockquote]:border-gray-600 [&_a]:text-sky-300 [&_a]:hover:underline [&_code]:bg-white/10 [&_code]:text-gray-100 [&_pre]:bg-white/10 [&_pre]:text-gray-200 [&_pre]:rounded-md [&_pre]:p-3 [&_th]:border-gray-700 [&_th]:text-gray-100 [&_td]:border-gray-700 [&_td]:text-gray-300',
  },
  amber: {
    heading: 'text-amber-900',
    paragraph: 'text-amber-950',
    list: 'text-amber-950',
    strong: 'text-amber-900',
    codeInline: 'bg-white/90 text-amber-900',
    codeBlock: 'bg-white/90 text-amber-900',
    blockquote: 'border-amber-300 text-amber-800',
    tableHead: 'bg-amber-100/60',
    tableBody: 'divide-amber-200',
    tableHeaderCell: 'border-amber-200 text-amber-900',
    tableCell: 'border-amber-200 text-amber-900',
    link: 'text-amber-700 hover:underline',
    htmlWrapper:
      '[&_h1]:text-amber-900 [&_h2]:text-amber-900 [&_h3]:text-amber-900 [&_h4]:text-amber-900 [&_h5]:text-amber-900 [&_h6]:text-amber-900 [&_p]:text-amber-950 [&_ul]:text-amber-950 [&_ol]:text-amber-950 [&_strong]:text-amber-900 [&_blockquote]:text-amber-800 [&_blockquote]:border-amber-300 [&_a]:text-amber-700 [&_a]:hover:underline [&_code]:bg-white/90 [&_code]:text-amber-900 [&_pre]:bg-white/90 [&_pre]:text-amber-900 [&_pre]:rounded-md [&_pre]:p-3 [&_th]:border-amber-200 [&_th]:text-amber-900 [&_td]:border-amber-200 [&_td]:text-amber-900',
  },
}

function getMarkdownComponents(
  tone: NonNullable<MarkdownContentProps['tone']>,
  compact: boolean,
): React.ComponentProps<typeof ReactMarkdown>['components'] {
  const toneClass = TONE_CLASSES[tone]
  const paragraphMarginClass = compact ? 'mb-1.5 last:mb-0' : 'mb-2 last:mb-0'
  const listMarginClass = compact ? 'mb-1.5 space-y-0.5' : 'mb-2 space-y-0.5'
  const blockMarginClass = compact ? 'my-1.5' : 'my-2'
  const headingTopClass = compact ? 'mt-2 first:mt-0' : 'mt-4 first:mt-0'

  return {
    h1: ({ className, ...props }) => (
      <h1 className={cn('text-lg font-semibold mb-2', headingTopClass, toneClass.heading, className)} {...props} />
    ),
    h2: ({ className, ...props }) => (
      <h2 className={cn('text-base font-semibold mb-1.5', compact ? 'mt-1.5 first:mt-0' : 'mt-3 first:mt-0', toneClass.heading, className)} {...props} />
    ),
    h3: ({ className, ...props }) => (
      <h3 className={cn('text-sm font-semibold mb-1', compact ? 'mt-1.5 first:mt-0' : 'mt-2.5 first:mt-0', toneClass.heading, className)} {...props} />
    ),
    h4: ({ className, ...props }) => (
      <h4 className={cn('text-sm font-medium mb-1', compact ? 'mt-1 first:mt-0' : 'mt-2 first:mt-0', toneClass.heading, className)} {...props} />
    ),
    h5: ({ className, ...props }) => (
      <h5 className={cn('text-sm font-medium mb-0.5', compact ? 'mt-1 first:mt-0' : 'mt-1.5 first:mt-0', toneClass.heading, className)} {...props} />
    ),
    h6: ({ className, ...props }) => (
      <h6 className={cn('text-sm font-medium mb-0.5', compact ? 'mt-1 first:mt-0' : 'mt-1 first:mt-0', toneClass.heading, className)} {...props} />
    ),
    p: ({ className, ...props }) => (
      <p className={cn('text-sm leading-relaxed', paragraphMarginClass, toneClass.paragraph, className)} {...props} />
    ),
    ul: ({ className, ...props }) => (
      <ul className={cn('text-sm list-disc pl-5', listMarginClass, toneClass.list, className)} {...props} />
    ),
    ol: ({ className, ...props }) => (
      <ol className={cn('text-sm list-decimal pl-5', listMarginClass, toneClass.list, className)} {...props} />
    ),
    li: ({ className, ...props }) => (
      <li className={cn('leading-relaxed', className)} {...props} />
    ),
    strong: ({ className, ...props }) => (
      <strong className={cn('font-semibold', toneClass.strong, className)} {...props} />
    ),
    code: ({ className, children, ...props }) => {
      const text = typeof children === 'string' ? children : ''
      const isBlock = text.includes('\n')
      return (
        <code
          className={cn(
            isBlock
              ? `block rounded-md text-sm font-mono overflow-x-auto ${blockMarginClass} p-3 ${toneClass.codeBlock}`
              : `inline rounded text-[0.8125em] font-mono px-1.5 py-0.5 ${toneClass.codeInline}`,
            className,
          )}
          {...props}
        >
          {children}
        </code>
      )
    },
    pre: ({ className, ...props }) => (
      <pre className={cn(blockMarginClass, 'overflow-x-auto', className)} {...props} />
    ),
    blockquote: ({ className, ...props }) => (
      <blockquote
        className={cn('border-l-4 pl-3 py-0.5 text-sm italic', blockMarginClass, toneClass.blockquote, className)}
        {...props}
      />
    ),
    table: ({ className, ...props }) => (
      <div className={cn('w-full overflow-x-auto', compact ? 'my-2' : 'my-3')}>
        <table className={cn('w-full border-collapse text-sm', toneClass.list, className)} {...props} />
      </div>
    ),
    thead: ({ className, ...props }) => (
      <thead className={cn(toneClass.tableHead, className)} {...props} />
    ),
    tbody: ({ className, ...props }) => (
      <tbody className={cn('divide-y', toneClass.tableBody, className)} {...props} />
    ),
    tr: ({ className, ...props }) => (
      <tr className={cn('border-b last:border-b-0', compact ? '' : '', className)} {...props} />
    ),
    th: ({ className, ...props }) => (
      <th
        className={cn('border px-3 py-2 text-left font-semibold whitespace-nowrap', toneClass.tableHeaderCell, className)}
        {...props}
      />
    ),
    td: ({ className, ...props }) => (
      <td className={cn('border px-3 py-2 align-top', toneClass.tableCell, className)} {...props} />
    ),
    a: ({ className, href, children, ...props }) => {
      // 安全兜底：如果 href 包含 CJK 字符，说明 autolink 仍然误判，降级为纯文本
      if (href && /[\u4E00-\u9FFF\u3000-\u303F\uFF00-\uFFEF]/.test(href)) {
        return <span className={cn('text-sm', toneClass.paragraph)}>{children}</span>
      }
      return (
        <a
          className={cn('text-sm', toneClass.link, className)}
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          {...props}
        >
          {children}
        </a>
      )
    },
  }
}

export function MarkdownContent({
  content,
  className,
  format = 'markdown',
  tone = 'default',
  compact = false,
  preserveLineBreaks = false,
}: MarkdownContentProps) {
  const effectiveFormat = useMemo(() => {
    if (format !== 'auto') return format
    return detectRichTextFormat(content)
  }, [content, format])

  const markdownInput = useMemo(
    () => normalizeMarkdownForRender(content, preserveLineBreaks),
    [content, preserveLineBreaks],
  )
  const htmlInput = useMemo(
    () => sanitizeHtmlForRender(content),
    [content],
  )
  const components = useMemo(
    () => getMarkdownComponents(tone, compact),
    [compact, tone],
  )

  return (
    <div
      className={cn(
        'markdown-content break-words',
        compact ? 'text-sm leading-relaxed' : '',
        className,
      )}
    >
      {effectiveFormat === 'html' ? (
        <div
          className={cn(
            'text-sm leading-relaxed [&_p]:mb-2 [&_p:last-child]:mb-0 [&_ul]:list-disc [&_ul]:pl-5 [&_ul]:mb-2 [&_ol]:list-decimal [&_ol]:pl-5 [&_ol]:mb-2 [&_li]:leading-relaxed [&_blockquote]:border-l-4 [&_blockquote]:pl-3 [&_blockquote]:py-0.5 [&_blockquote]:my-2 [&_blockquote]:italic [&_table]:w-full [&_table]:border-collapse [&_th]:border [&_th]:px-3 [&_th]:py-2 [&_th]:text-left [&_th]:font-semibold [&_td]:border [&_td]:px-3 [&_td]:py-2 [&_pre]:my-2 [&_pre]:overflow-x-auto',
            TONE_CLASSES[tone].htmlWrapper,
          )}
          // 这里只接受 sanitize 之后的内容，避免直接注入不可信 HTML。
          dangerouslySetInnerHTML={{ __html: htmlInput }}
        />
      ) : (
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
          {markdownInput}
        </ReactMarkdown>
      )}
    </div>
  )
}
