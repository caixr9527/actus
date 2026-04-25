export type FilePreviewKind =
  | 'text'
  | 'markdown'
  | 'json'
  | 'csv'
  | 'image'
  | 'pdf'
  | 'audio'
  | 'video'
  | 'unsupported'

export type FilePreviewType = {
  kind: FilePreviewKind
  binary: boolean
}

const markdownExtensions = new Set(['md', 'markdown', 'mdx'])
const jsonExtensions = new Set(['json', 'jsonl', 'geojson', 'ipynb'])
const csvExtensions = new Set(['csv', 'tsv'])
const imageExtensions = new Set(['jpg', 'jpeg', 'png', 'gif', 'svg', 'webp', 'bmp', 'ico', 'avif'])
const pdfExtensions = new Set(['pdf'])
const audioExtensions = new Set(['mp3', 'wav', 'ogg', 'm4a', 'aac', 'flac', 'webm'])
const videoExtensions = new Set(['mp4', 'webm', 'mov', 'm4v', 'ogv'])
const textExtensions = new Set([
  'txt', 'xml', 'html', 'htm', 'css', 'scss', 'sass', 'less',
  'js', 'jsx', 'ts', 'tsx', 'vue', 'py', 'java', 'go', 'rs', 'c', 'cpp', 'h', 'hpp',
  'cs', 'php', 'rb', 'swift', 'kt', 'scala', 'sh', 'bash', 'zsh', 'yml', 'yaml',
  'toml', 'ini', 'conf', 'config', 'log', 'sql', 'r', 'dart', 'lua', 'pl', 'perl',
  'dockerfile', 'gitignore', 'env', 'properties', 'gradle', 'lock',
])

function normalizeExtension(extension: string): string {
  return extension.toLowerCase().replace(/^\./, '').trim()
}

export function resolveFilePreviewType(extension: string, contentType?: string | null): FilePreviewType {
  const ext = normalizeExtension(extension)
  const mime = String(contentType || '').toLowerCase()

  if (markdownExtensions.has(ext)) return { kind: 'markdown', binary: false }
  if (jsonExtensions.has(ext) || mime.includes('application/json')) return { kind: 'json', binary: false }
  if (csvExtensions.has(ext) || mime.includes('text/csv') || mime.includes('text/tab-separated-values')) {
    return { kind: 'csv', binary: false }
  }
  if (imageExtensions.has(ext) || mime.startsWith('image/')) return { kind: 'image', binary: true }
  if (pdfExtensions.has(ext) || mime.includes('application/pdf')) return { kind: 'pdf', binary: true }
  if (audioExtensions.has(ext) || mime.startsWith('audio/')) return { kind: 'audio', binary: true }
  if (videoExtensions.has(ext) || mime.startsWith('video/')) return { kind: 'video', binary: true }
  if (textExtensions.has(ext) || mime.startsWith('text/')) return { kind: 'text', binary: false }
  return { kind: 'unsupported', binary: true }
}

export function formatJsonPreview(raw: string): string {
  try {
    return JSON.stringify(JSON.parse(raw), null, 2)
  } catch {
    return raw
  }
}

export function parseDelimitedPreview(raw: string, delimiter: ',' | '\t'): string[][] {
  const rows = raw
    .split(/\r?\n/)
    .map((line) => line.split(delimiter).map((cell) => cell.trim()))
    .filter((row) => row.some((cell) => cell.length > 0))
  return rows.slice(0, 200)
}
