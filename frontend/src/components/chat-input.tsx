'use client'

import {useState, useRef, forwardRef, useImperativeHandle} from 'react'
import {cn, formatFileSize} from '@/lib/utils'
import {ScrollArea, ScrollBar} from '@/components/ui/scroll-area'
import {Item, ItemActions, ItemContent, ItemDescription, ItemMedia, ItemTitle} from '@/components/ui/item'
import {Avatar, AvatarGroupCount} from '@/components/ui/avatar'
import {ArrowUp, Check, ChevronsUpDown, FileText, Paperclip, XCircle, Loader2, Pause, Sparkles} from 'lucide-react'
import {Button} from '@/components/ui/button'
import {DropdownMenu, DropdownMenuContent, DropdownMenuTrigger} from '@/components/ui/dropdown-menu'
import { getApiErrorMessage } from '@/lib/api'
import {fileApi} from '@/lib/api/file'
import type {FileInfo, ListModelItem} from '@/lib/api/types'
import { performModelSelection, resolveModelSelectorState } from '@/lib/chat-model-selector'
import { resolveChatInputInteractionState } from '@/lib/chat-input-interaction'
import {toast} from 'sonner'
import { useI18n } from '@/lib/i18n'

interface ChatInputProps {
  className?: string
  onInputValueChange?: (value: string) => void
  onSend?: (message: string, files: FileInfo[]) => Promise<void>
  onRequireAuth?: (action: 'send' | 'upload', message: string) => boolean | Promise<boolean>
  disabled?: boolean
  /** 当前会话 ID，上传附件时会关联到该会话 */
  sessionId?: string | null
  /** 任务是否正在运行中 */
  isRunning?: boolean
  /** 点击暂停按钮的回调 */
  onStop?: () => void
  modelOptions?: ListModelItem[]
  currentModelId?: string | null
  defaultModelId?: string | null
  modelsLoading?: boolean
  modelUpdating?: boolean
  onModelChange?: (modelId: string) => Promise<void>
}

export interface ChatInputRef {
  setInputText: (text: string) => void
  getInputValue: () => string
  getFiles: () => FileInfo[]
}

export const ChatInput = forwardRef<ChatInputRef, ChatInputProps>(
  ({
    className,
    onInputValueChange,
    onSend,
    onRequireAuth,
    disabled = false,
    sessionId,
    isRunning = false,
    onStop,
    modelOptions = [],
    currentModelId,
    defaultModelId,
    modelsLoading = false,
    modelUpdating = false,
    onModelChange,
  }, ref) => {
    const { t } = useI18n()
    const [files, setFiles] = useState<FileInfo[]>([])
    const [uploading, setUploading] = useState(false)
    const [sending, setSending] = useState(false)
    const [modelMenuOpen, setModelMenuOpen] = useState(false)
    const [inputValue, setInputValue] = useState('')
    const fileInputRef = useRef<HTMLInputElement>(null)
    const textareaRef = useRef<HTMLTextAreaElement>(null)
    const { selectedModelId, selectedModel, defaultModel: effectiveDefaultModel } = resolveModelSelectorState(
      modelOptions,
      currentModelId,
      defaultModelId,
    )
    const showModelSelector = onModelChange !== undefined
    const modelButtonLabel = selectedModel?.display_name ?? t('chatInput.modelAuto')
    const modelButtonBadge = selectedModel?.config?.badge ?? (selectedModelId === 'auto' ? t('chatInput.modelAutoBadge') : null)
    const interactionState = resolveChatInputInteractionState({
      disabled,
      isRunning,
      sending,
      inputValue,
      modelsLoading,
      modelUpdating,
    })

    const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      const value = e.target.value
      setInputValue(value)
      onInputValueChange?.(value)
    }

    useImperativeHandle(ref, () => ({
      setInputText: (text: string) => {
        setInputValue(text)
        onInputValueChange?.(text)
        // 聚焦到输入框
        textareaRef.current?.focus()
      },
      getInputValue: () => inputValue,
      getFiles: () => files,
    }))

    const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
      const selectedFiles = event.target.files
      if (!selectedFiles || selectedFiles.length === 0) {
        return
      }

      setUploading(true)

      try {
        const uploadPromises = Array.from(selectedFiles).map(async (file) => {
          try {
            const fileInfo = await fileApi.uploadFile({
              file,
              ...(sessionId && { session_id: sessionId }),
            })
            return fileInfo
          } catch (error) {
            const errorMessage = getApiErrorMessage(error, 'chatInput.uploadFailed', t)
            toast.error(
              t('chatInput.uploadSingleFailed', {
                name: file.name,
                error: errorMessage,
              }),
            )
            return null
          }
        })

        const uploadedFiles = (await Promise.all(uploadPromises)).filter(
          (file): file is FileInfo => file !== null
        )

        if (uploadedFiles.length > 0) {
          setFiles((prev) => [...prev, ...uploadedFiles])
          toast.success(t('chatInput.uploadSuccess', { count: uploadedFiles.length }))
        }
      } catch {
        toast.error(t('chatInput.uploadProcessError'))
      } finally {
        setUploading(false)
        // 重置input，以便可以重复选择同一文件
        if (fileInputRef.current) {
          fileInputRef.current.value = ''
        }
      }
    }

    const checkAuthGate = async (action: 'send' | 'upload', message: string): Promise<boolean> => {
      if (!onRequireAuth) {
        return true
      }
      try {
        return await onRequireAuth(action, message)
      } catch {
        return false
      }
    }

    const handleUploadClick = async () => {
      const allowed = await checkAuthGate('upload', inputValue)
      if (!allowed) {
        return
      }
      fileInputRef.current?.click()
    }

    const handleRemoveFile = (fileId: string) => {
      setFiles((prev) => prev.filter((file) => file.id !== fileId))
    }

    const handleModelSelect = async (modelId: string) => {
      if (!interactionState.canSwitchModel) {
        return
      }
      try {
        await performModelSelection({
          nextModelId: modelId,
          selectedModelId,
          modelsLoading,
          modelUpdating,
          onModelChange,
          closeMenu: () => setModelMenuOpen(false),
        })
      } catch (error) {
        toast.error(getApiErrorMessage(error, 'chatInput.modelUpdateFailed', t))
      }
    }

    const handleSend = async () => {
      if (!interactionState.canSend) {
        return
      }
      const trimmedMessage = inputValue.trim()
      
      // 验证消息不为空
      if (!trimmedMessage) {
        toast.error(t('chatInput.messageRequired'))
        textareaRef.current?.focus()
        return
      }

      const allowed = await checkAuthGate('send', inputValue)
      if (!allowed) {
        return
      }

      // 如果提供了 onSend 回调，使用它
      if (onSend) {
        setSending(true)
        try {
          await onSend(trimmedMessage, files)
          // 发送成功后清空输入框和文件列表
          setInputValue('')
          setFiles([])
          onInputValueChange?.('')
        } catch (error) {
          // 错误处理由 onSend 内部处理
          console.error('发送消息失败:', error)
        } finally {
          setSending(false)
        }
      }
    }

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      // 支持 Ctrl/Cmd + Enter 发送
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault()
        handleSend()
      }
    }

    return (
    <div className={cn('flex flex-col bg-white w-full rounded-2xl py-3 border', className)}>
      {/* 顶部的文件列表 */}
      {files.length > 0 && (
        <div className="w-full px-4 mb-1">
          <ScrollArea className="w-full whitespace-nowrap">
            <div className="flex w-max space-x-4 pb-4">
              {files.map((file) => (
                <Item
                  key={file.id}
                  variant="muted"
                  className="p-2 flex-shrink-0 gap-2"
                >
                  {/* 左侧文件图标 */}
                  <ItemMedia>
                    <Avatar className="size-8">
                      <AvatarGroupCount>
                        <FileText/>
                      </AvatarGroupCount>
                    </Avatar>
                  </ItemMedia>
                  {/* 文件信息 */}
                  <ItemContent className="gap-0">
                    <ItemTitle className="text-sm text-gray-700">{file.filename}</ItemTitle>
                    <ItemDescription className="text-xs">
                      {file.extension} · {formatFileSize(file.size)}
                    </ItemDescription>
                  </ItemContent>
                  <ItemActions>
                    <Button
                      variant="ghost"
                      size="icon-xs"
                      className="cursor-pointer"
                      onClick={() => handleRemoveFile(file.id)}
                      disabled={uploading}
                    >
                      <XCircle/>
                    </Button>
                  </ItemActions>
                </Item>
              ))}
            </div>
            <ScrollBar orientation="horizontal"/>
          </ScrollArea>
        </div>
      )}
      {/* 中间输入框 */}
      <div className="px-4 mb-3">
        <textarea
          ref={textareaRef}
          rows={2}
          value={inputValue}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          placeholder={t('chatInput.placeholder')}
          className="scrollbar-hide outline-none w-full text-sm resize-none h-[46px] min-h-[40px]"
          disabled={!interactionState.canEditInput}
        />
      </div>
      {/* 底部上传&发送按钮 */}
      <footer className="flex flex-row justify-between w-full px-3">
        {/* 上传按钮 */}
        <div className="flex gap-2">
          <input
            ref={fileInputRef}
            type="file"
            multiple
            className="hidden"
            onChange={handleFileSelect}
            disabled={uploading}
          />
          <Button
            variant="outline"
            className="rounded-full w-8 h-8 cursor-pointer"
            onClick={() => {
              void handleUploadClick()
            }}
            disabled={uploading}
          >
            {uploading ? (
              <Loader2 className="size-4 animate-spin"/>
            ) : (
              <Paperclip/>
            )}
          </Button>
          {showModelSelector ? (
            <DropdownMenu open={modelMenuOpen} onOpenChange={setModelMenuOpen}>
              <DropdownMenuTrigger asChild>
                <Button
                  type="button"
                  variant="outline"
                  className="h-8 max-w-[220px] rounded-full px-3 text-xs font-medium text-gray-700 shadow-none"
                  disabled={!interactionState.canSwitchModel}
                >
                  <span className="inline-flex min-w-0 items-center gap-2">
                    <Sparkles className="size-3.5 text-gray-500" />
                    <span className="truncate">
                      {modelsLoading ? t('chatInput.modelLoading') : modelButtonLabel}
                    </span>
                    {modelButtonBadge ? (
                      <span className="rounded-full bg-gray-100 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-[0.08em] text-gray-500">
                        {modelButtonBadge}
                      </span>
                    ) : null}
                  </span>
                  <ChevronsUpDown className="size-3.5 shrink-0 text-gray-400" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="start" className="w-[320px] p-2">
                <div className="mb-2 px-2 pt-1">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-gray-400">
                    {t('chatInput.modelSelectTitle')}
                  </p>
                </div>
                <div className="flex flex-col gap-1">
                  <button
                    type="button"
                    className={cn(
                      'flex w-full cursor-pointer items-start gap-3 rounded-xl border px-3 py-2.5 text-left transition-colors',
                      selectedModelId === 'auto'
                        ? 'border-gray-300 bg-gray-50'
                        : 'border-transparent hover:border-gray-200 hover:bg-gray-50',
                    )}
                    onClick={() => {
                      void handleModelSelect('auto')
                    }}
                  >
                    <Check className={cn('mt-0.5 size-4 shrink-0', selectedModelId === 'auto' ? 'opacity-100 text-gray-700' : 'opacity-0')} />
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <span className="truncate text-sm font-medium text-gray-800">{t('chatInput.modelAuto')}</span>
                        <span className="rounded-full bg-gray-100 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-[0.08em] text-gray-500">
                          {t('chatInput.modelAutoBadge')}
                        </span>
                      </div>
                      <p className="mt-1 text-xs leading-5 text-gray-500">
                        {effectiveDefaultModel
                          ? t('chatInput.modelAutoDescriptionWithDefault', { model: effectiveDefaultModel.display_name })
                          : t('chatInput.modelAutoDescription')}
                      </p>
                    </div>
                  </button>
                  {modelOptions.map((model) => {
                    const isSelected = model.id === selectedModelId
                    return (
                      <button
                        key={model.id}
                        type="button"
                        className={cn(
                          'flex w-full cursor-pointer items-start gap-3 rounded-xl border px-3 py-2.5 text-left transition-colors',
                          isSelected
                            ? 'border-gray-300 bg-gray-50'
                            : 'border-transparent hover:border-gray-200 hover:bg-gray-50',
                        )}
                        onClick={() => {
                          void handleModelSelect(model.id)
                        }}
                      >
                        <Check className={cn('mt-0.5 size-4 shrink-0', isSelected ? 'opacity-100 text-gray-700' : 'opacity-0')} />
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center gap-2">
                            <span className="truncate text-sm font-medium text-gray-800">{model.display_name}</span>
                            {model.config?.badge ? (
                              <span className="rounded-full bg-gray-100 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-[0.08em] text-gray-500">
                                {model.config.badge}
                              </span>
                            ) : null}
                          </div>
                          <p className="mt-1 text-xs leading-5 text-gray-500">
                            {model.config?.description || model.provider}
                          </p>
                        </div>
                      </button>
                    )
                  })}
                </div>
              </DropdownMenuContent>
            </DropdownMenu>
          ) : null}
        </div>
        {/* 发送/暂停按钮 */}
        <div className="flex gap-2">
          {isRunning ? (
            // 任务运行中时显示暂停按钮
            <Button
              variant="outline"
              className="rounded-full w-8 h-8 cursor-pointer"
              onClick={onStop}
              disabled={!onStop}
            >
              <Pause className="size-4" />
            </Button>
          ) : (
            // 任务未运行时显示发送按钮
            <Button
              variant="outline"
              className="rounded-full w-8 h-8 cursor-pointer"
              onClick={handleSend}
              disabled={!interactionState.canSend}
            >
              {sending ? (
                <Loader2 className="size-4 animate-spin"/>
              ) : (
                <ArrowUp/>
              )}
            </Button>
          )}
        </div>
      </footer>
    </div>
    )
  }
)

ChatInput.displayName = 'ChatInput'
