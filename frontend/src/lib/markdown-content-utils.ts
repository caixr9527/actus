const CJK_RANGES = '\u3000-\u303F\u4E00-\u9FFF\uFF01-\uFF60'
const URL_FOLLOWED_BY_CJK = new RegExp(
  `(https?:\\/\\/[^\\s${CJK_RANGES}]+)([${CJK_RANGES}])`,
  'g',
)

const HTML_TAG_PATTERN = /<([a-z][a-z0-9-]*)(\s[^>]*)?>[\s\S]*?<\/\1>|<([a-z][a-z0-9-]*)(\s[^>]*)?\/?>/i
const MARKDOWN_SYNTAX_PATTERN = /(^|\n)\s{0,3}(#{1,6}\s|[-*+]\s|\d+\.\s|>\s)|```|`[^`]+`|\[[^\]]+\]\([^)]+\)/m

const INLINE_EVENT_ATTR_PATTERN = /\son[a-z-]+\s*=\s*(".*?"|'.*?'|[^\s>]+)/gi
const SCRIPT_TAG_PATTERN = /<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi
const STYLE_TAG_PATTERN = /<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>/gi
const IFRAME_TAG_PATTERN = /<iframe\b[^<]*(?:(?!<\/iframe>)<[^<]*)*<\/iframe>/gi
const OBJECT_TAG_PATTERN = /<object\b[^<]*(?:(?!<\/object>)<[^<]*)*<\/object>/gi
const EMBED_TAG_PATTERN = /<embed\b[^>]*>/gi
const SRC_DOC_PATTERN = /\ssrcdoc\s*=\s*(".*?"|'.*?'|[^\s>]+)/gi
const JAVASCRIPT_URL_ATTR_PATTERN = /\s(href|src|xlink:href)\s*=\s*(['"])\s*javascript:[\s\S]*?\2/gi

export function normalizeAutolinks(text: string): string {
  return text.replace(URL_FOLLOWED_BY_CJK, '$1 $2')
}

function stripDangerousTagsAndAttrs(raw: string): string {
  return raw
    .replace(SCRIPT_TAG_PATTERN, '')
    .replace(STYLE_TAG_PATTERN, '')
    .replace(IFRAME_TAG_PATTERN, '')
    .replace(OBJECT_TAG_PATTERN, '')
    .replace(EMBED_TAG_PATTERN, '')
    .replace(INLINE_EVENT_ATTR_PATTERN, '')
    .replace(SRC_DOC_PATTERN, '')
    .replace(JAVASCRIPT_URL_ATTR_PATTERN, '')
}

export function detectRichTextFormat(content: string): 'markdown' | 'html' {
  const trimmed = content.trim()
  if (!trimmed) return 'markdown'
  if (HTML_TAG_PATTERN.test(trimmed) && !MARKDOWN_SYNTAX_PATTERN.test(trimmed)) {
    return 'html'
  }
  return 'markdown'
}

export function sanitizeHtmlForRender(rawHtml: string): string {
  const preSanitized = stripDangerousTagsAndAttrs(rawHtml)

  if (typeof window === 'undefined' || typeof DOMParser === 'undefined') {
    return preSanitized
  }

  const parser = new DOMParser()
  const doc = parser.parseFromString(preSanitized, 'text/html')
  const blockedTags = ['script', 'style', 'iframe', 'object', 'embed', 'link', 'meta', 'base']

  for (const tag of blockedTags) {
    doc.querySelectorAll(tag).forEach((node) => node.remove())
  }

  doc.querySelectorAll('*').forEach((element) => {
    for (const attr of Array.from(element.attributes)) {
      const name = attr.name.toLowerCase()
      const value = attr.value.trim().toLowerCase()
      if (name.startsWith('on') || name === 'srcdoc') {
        element.removeAttribute(attr.name)
        continue
      }
      if (
        (name === 'href' || name === 'src' || name === 'xlink:href')
        && (value.startsWith('javascript:') || value.startsWith('data:text/html'))
      ) {
        element.removeAttribute(attr.name)
      }
    }
  })

  return doc.body.innerHTML
}

export function normalizeMarkdownForRender(content: string, preserveLineBreaks: boolean): string {
  const normalized = normalizeAutolinks(content)
  if (!preserveLineBreaks) return normalized
  return normalized.replace(/\n/g, '  \n')
}
