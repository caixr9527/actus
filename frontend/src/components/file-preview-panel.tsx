'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import Image from 'next/image'
import { fileApi, getApiErrorMessage } from '@/lib/api'
import { Button } from '@/components/ui/button'
import { Download, FileText, X } from 'lucide-react'
import { ScrollArea } from '@/components/ui/scroll-area'
import { formatFileSize } from '@/lib/utils'
import { toast } from 'sonner'
import type { AttachmentFile } from '@/lib/session-events'
import { useI18n } from '@/lib/i18n'
import { MarkdownContent } from '@/components/markdown-content'
import {
  formatJsonPreview,
  parseDelimitedPreview,
  resolveFilePreviewType,
} from '@/lib/file-preview'

export interface FilePreviewPanelProps {
  /** 要预览的文件信息 */
  file: AttachmentFile | null
  /** 关闭回调 */
  onClose: () => void
}

export function FilePreviewPanel({ file, onClose }: FilePreviewPanelProps) {
  const { t } = useI18n()
  const [content, setContent] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const previewRequestIdRef = useRef(0)
  const previewAbortControllerRef = useRef<AbortController | null>(null)
  const previewObjectUrlRef = useRef<string | null>(null)

  const fileType = file
    ? resolveFilePreviewType(file.extension, (file as { content_type?: string | null }).content_type)
    : { kind: 'unsupported' as const, binary: true }

  const cleanupPreviewObjectUrl = useCallback(() => {
    if (!previewObjectUrlRef.current) return
    URL.revokeObjectURL(previewObjectUrlRef.current)
    previewObjectUrlRef.current = null
  }, [])

  // 加载文件内容
  const loadFileContent = useCallback(async (fileId: string, binary: boolean, unsupported: boolean) => {
    previewRequestIdRef.current += 1
    const requestId = previewRequestIdRef.current
    previewAbortControllerRef.current?.abort()
    const controller = new AbortController()
    previewAbortControllerRef.current = controller

    setLoading(true)
    setError(null)
    setContent(null)
    cleanupPreviewObjectUrl()
    setImageUrl(null)

    if (unsupported) {
      setLoading(false)
      return
    }

    try {
      if (binary) {
        // 浏览器原生可预览的二进制格式统一走 object URL。
        const blob = await fileApi.downloadFile(fileId, { signal: controller.signal })
        if (controller.signal.aborted || requestId !== previewRequestIdRef.current) return
        const url = URL.createObjectURL(blob)
        previewObjectUrlRef.current = url
        setImageUrl(url)
      } else {
        // 文本类型：读取内容
        const blob = await fileApi.downloadFile(fileId, { signal: controller.signal })
        if (controller.signal.aborted || requestId !== previewRequestIdRef.current) return
        const text = await blob.text()
        if (controller.signal.aborted || requestId !== previewRequestIdRef.current) return
        setContent(text)
      }
    } catch (err) {
      if (controller.signal.aborted || requestId !== previewRequestIdRef.current) return
      const msg = getApiErrorMessage(err, 'filePreview.loadContentFailed', t)
      setError(msg)
      toast.error(msg)
    } finally {
      if (requestId === previewRequestIdRef.current) {
        setLoading(false)
      }
    }
  }, [cleanupPreviewObjectUrl, t])

  // 下载文件
  const handleDownload = useCallback(async () => {
    if (!file) return
    
    try {
      const blob = await fileApi.downloadFile(file.id)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = file.filename
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
      toast.success(t('filePreview.downloadSuccess', { filename: file.filename }))
    } catch (err) {
      const msg = getApiErrorMessage(err, 'filePreview.downloadFailedDefault', t)
      toast.error(t('filePreview.downloadFailed', { error: msg }))
    }
  }, [file, t])

  // 当文件改变时加载内容
  useEffect(() => {
    if (file && file.id) {
      loadFileContent(file.id, fileType.binary, fileType.kind === 'unsupported')
    }
  }, [file, fileType.binary, fileType.kind, loadFileContent])

  // 清理函数：关闭时释放资源
  useEffect(() => {
    return () => {
      previewRequestIdRef.current += 1
      previewAbortControllerRef.current?.abort()
      previewAbortControllerRef.current = null
      cleanupPreviewObjectUrl()
    }
  }, [cleanupPreviewObjectUrl])

  if (!file) {
    return null
  }

  return (
    <div className="flex flex-col h-full bg-white border-l border-gray-200">
      {/* 头部：文件名 + 操作按钮 - 添加背景色区分 */}
      <div className="flex items-center justify-between gap-3 px-4 py-3 border-b border-gray-200 bg-gray-50 flex-shrink-0">
        <div className="flex items-center gap-3 min-w-0 flex-1">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-blue-100 text-blue-600">
            <FileText size={16} />
          </div>
          <div className="min-w-0 flex-1">
            <p className="text-sm font-medium text-gray-900 truncate">{file.filename}</p>
            <p className="text-xs text-gray-500">
              {file.extension.replace(/^\./, '')} · {formatFileSize(file.size)}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-1 flex-shrink-0">
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={handleDownload}
            aria-label={t('filePreview.downloadAria')}
            className="cursor-pointer"
          >
            <Download size={16} />
          </Button>
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={onClose}
            aria-label={t('filePreview.closeAria')}
            className="cursor-pointer"
          >
            <X size={16} />
          </Button>
        </div>
      </div>

      {/* 内容区域 */}
      <div className="flex-1 overflow-hidden">
        {loading && (
          <div className="flex items-center justify-center h-full">
            <p className="text-sm text-gray-500">{t('common.loading')}</p>
          </div>
        )}

        {error && !loading && (
          <div className="flex items-center justify-center h-full px-6">
            <p className="text-sm text-red-600">{error}</p>
          </div>
        )}

        {!loading && !error && fileType.kind === 'unsupported' && (
          <div className="flex flex-col items-center justify-center h-full px-6 gap-4">
            <div className="flex h-16 w-16 items-center justify-center rounded-full bg-gray-100 text-gray-400">
              <FileText size={32} />
            </div>
            <div className="text-center">
              <p className="text-sm text-gray-700 font-medium">{t('filePreview.unsupportedTitle')}</p>
              <p className="text-xs text-gray-500 mt-1">{t('filePreview.unsupportedHint')}</p>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={handleDownload}
              className="gap-2"
            >
              <Download size={16} />
              {t('filePreview.downloadAction')}
            </Button>
          </div>
        )}

        {!loading && !error && fileType.kind === 'image' && imageUrl && (
          <ScrollArea className="h-full">
            <div className="p-4">
              <Image
                src={imageUrl}
                alt={file.filename}
                width={1200}
                height={800}
                unoptimized
                className="max-w-full h-auto rounded-lg border"
              />
            </div>
          </ScrollArea>
        )}

        {!loading && !error && fileType.kind === 'pdf' && imageUrl && (
          <iframe
            src={imageUrl}
            title={file.filename}
            className="h-full w-full border-0 bg-white"
          />
        )}

        {!loading && !error && fileType.kind === 'audio' && imageUrl && (
          <div className="flex h-full items-center justify-center p-6">
            <audio controls src={imageUrl} className="w-full max-w-xl">
              {t('filePreview.unsupportedHint')}
            </audio>
          </div>
        )}

        {!loading && !error && fileType.kind === 'video' && imageUrl && (
          <div className="flex h-full items-center justify-center bg-black p-4">
            <video controls src={imageUrl} className="max-h-full max-w-full rounded-lg">
              {t('filePreview.unsupportedHint')}
            </video>
          </div>
        )}

        {!loading && !error && fileType.kind === 'markdown' && content !== null && (
          <ScrollArea className="h-full">
            <div className="p-4">
              <MarkdownContent content={content} className="text-gray-700" />
            </div>
          </ScrollArea>
        )}

        {!loading && !error && fileType.kind === 'json' && content !== null && (
          <ScrollArea className="h-full">
            <pre className="p-4 text-xs font-mono whitespace-pre-wrap break-words text-gray-700">
              {formatJsonPreview(content)}
            </pre>
          </ScrollArea>
        )}

        {!loading && !error && fileType.kind === 'csv' && content !== null && (
          <ScrollArea className="h-full">
            <div className="p-4">
              <div className="overflow-auto rounded-lg border border-gray-200">
                <table className="min-w-full border-collapse text-xs">
                  <tbody>
                    {parseDelimitedPreview(content, file.extension.toLowerCase().replace(/^\./, '') === 'tsv' ? '\t' : ',').map((row, rowIndex) => (
                      <tr key={rowIndex} className={rowIndex === 0 ? 'bg-gray-50 font-medium text-gray-800' : 'text-gray-700'}>
                        {row.map((cell, cellIndex) => (
                          <td key={cellIndex} className="max-w-[240px] border border-gray-200 px-2 py-1 align-top break-words">
                            {cell}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </ScrollArea>
        )}

        {!loading && !error && fileType.kind === 'text' && content !== null && (
          <ScrollArea className="h-full">
            <pre className="p-4 text-xs font-mono whitespace-pre-wrap break-words text-gray-700">
              {content}
            </pre>
          </ScrollArea>
        )}
      </div>
    </div>
  )
}
