"use client"

import { useEffect, useRef, useState } from "react"
import { useRouter } from "next/navigation"
import { ChatHeader } from "@/components/chat-header"
import { ChatInput, type ChatInputRef } from "@/components/chat-input"
import { SuggestedQuestions } from "@/components/suggested-questions"
import { AuthLoginDialog } from "@/components/auth-login-dialog"
import { sessionApi } from "@/lib/api/session"
import { saveInitialMessageDraft } from "@/lib/initial-message-draft"
import {
  clearGuestPendingAction,
  clearGuestPendingMessage,
  loadGuestPendingAction,
  loadGuestPendingMessage,
  saveGuestPendingAction,
  saveGuestPendingMessage,
  type GuestPendingActionType,
} from "@/lib/guest-auth-draft"
import { useAuth } from "@/hooks/use-auth"
import type { FileInfo } from "@/lib/api/types"
import { toast } from "sonner"

export default function Page() {
  const router = useRouter()
  const { isLoggedIn } = useAuth()
  const chatInputRef = useRef<ChatInputRef>(null)
  const [sending, setSending] = useState(false)
  const [authDialogOpen, setAuthDialogOpen] = useState(false)

  useEffect(() => {
    if (!isLoggedIn) {
      return
    }

    const pendingMessage = loadGuestPendingMessage()
    const pendingAction = loadGuestPendingAction()

    if (pendingMessage) {
      requestAnimationFrame(() => {
        chatInputRef.current?.setInputText(pendingMessage)
      })
    }

    if (pendingAction === "upload") {
      toast.info("请重新选择需要上传的附件")
    }

    clearGuestPendingMessage()
    clearGuestPendingAction()
  }, [isLoggedIn])

  const handleQuestionClick = (question: string) => {
    chatInputRef.current?.setInputText(question)
  }

  const openLoginDialog = (action: GuestPendingActionType, message: string) => {
    saveGuestPendingAction(action)
    saveGuestPendingMessage(message)
    setAuthDialogOpen(true)
  }

  const handleRequireAuth = async (
    action: "send" | "upload",
    message: string,
  ): Promise<boolean> => {
    if (isLoggedIn) {
      return true
    }

    openLoginDialog(action, message)
    return false
  }

  const handleLoginClick = () => {
    const currentMessage = chatInputRef.current?.getInputValue() || ""
    openLoginDialog("manual_login", currentMessage)
  }

  const handleAuthDialogOpenChange = (open: boolean) => {
    setAuthDialogOpen(open)
    if (!open && !isLoggedIn) {
      clearGuestPendingAction()
      clearGuestPendingMessage()
    }
  }

  const handleSend = async (message: string, files: FileInfo[]) => {
    if (sending) return

    setSending(true)

    try {
      // 1. 创建新会话
      const session = await sessionApi.createSession()
      const sessionId = session.session_id

      // 2. 优先写入 sessionStorage，避免 URL 长度与可见性问题
      const attachments = files.map((file) => file.id)
      const stored = saveInitialMessageDraft(sessionId, message, attachments)

      // 3. 跳转到详情页。若 sessionStorage 不可用，回退到旧 query 参数链路
      if (stored) {
        router.push(`/sessions/${sessionId}`)
      } else {
        const payload = JSON.stringify({ message, attachments })
        const encoded = btoa(encodeURIComponent(payload))
        router.push(`/sessions/${sessionId}?init=${encoded}`)
      }
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "创建会话失败"
      toast.error(errorMessage)
      setSending(false)
      throw error
    }
  }

  return (
    <div className="h-full flex flex-col">
      {/* 顶部header */}
      <ChatHeader onLoginClick={handleLoginClick} />
      {/* 中间对话框 - 垂直居中，视觉上移一个导航栏高度 */}
      <div className="flex-1 flex items-center justify-center px-4 py-6 sm:py-8 -mt-12 sm:-mt-16">
        <div className="w-full max-w-full sm:max-w-[768px] sm:min-w-[390px] mx-auto">
          {/* 对话提示内容 */}
          <div className="text-[24px] sm:text-[32px] font-bold mb-4 sm:mb-6 text-center sm:text-left">
            <div className="text-gray-700">您好！</div>
            <div className="text-gray-500">我能为您做什么?</div>
          </div>
          {/* 对话框 */}
          <ChatInput
            ref={chatInputRef}
            className="mb-4 sm:mb-6"
            onSend={handleSend}
            onRequireAuth={handleRequireAuth}
            disabled={sending}
          />
          {/* 推荐对话内容 */}
          <SuggestedQuestions onQuestionClick={handleQuestionClick} />
        </div>
      </div>

      <AuthLoginDialog
        open={authDialogOpen}
        onOpenChange={handleAuthDialogOpenChange}
      />
    </div>
  )
}
