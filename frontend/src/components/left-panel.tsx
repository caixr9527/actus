'use client'

import {useCallback, useEffect} from 'react'
import {useRouter} from 'next/navigation'
import {Sidebar, SidebarContent, SidebarHeader, SidebarTrigger} from '@/components/ui/sidebar'
import {Button} from '@/components/ui/button'
import {Plus} from 'lucide-react'
import {Kbd, KbdGroup} from '@/components/ui/kbd'
import {SessionList} from '@/components/session-list'

const NEW_TASK_SHORTCUT_KEY = 'k'

function isEditableElement(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  if (target.isContentEditable) return true
  if (target.closest('[contenteditable="true"]')) return true
  if (target instanceof HTMLInputElement) return true
  if (target instanceof HTMLTextAreaElement) return true
  if (target instanceof HTMLSelectElement) return true
  return false
}

export function LeftPanel() {
  const router = useRouter()
  const handleCreateTask = useCallback(() => {
    router.push('/')
  }, [router])

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const isShortcut = (event.metaKey || event.ctrlKey)
        && !event.altKey
        && !event.shiftKey
        && event.key.toLowerCase() === NEW_TASK_SHORTCUT_KEY

      if (!isShortcut) return
      if (isEditableElement(event.target)) return

      event.preventDefault()
      handleCreateTask()
    }

    window.addEventListener('keydown', onKeyDown)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
    }
  }, [handleCreateTask])

  return (
    <Sidebar>
      {/* 顶部的切换按钮 */}
      <SidebarHeader>
        <SidebarTrigger className="cursor-pointer"/>
      </SidebarHeader>
      {/* 中间内容 */}
      <SidebarContent className="p-2">
        {/* 新建任务 */}
        <Button
          variant="outline"
          className="cursor-pointer mb-3"
          onClick={handleCreateTask}
        >
          <Plus/>
          新建任务
          <KbdGroup>
            <Kbd>⌘/Ctrl</Kbd>
            <Kbd>K</Kbd>
          </KbdGroup>
        </Button>
        {/* 会话列表 */}
        <SessionList/>
      </SidebarContent>
    </Sidebar>
  )
}
