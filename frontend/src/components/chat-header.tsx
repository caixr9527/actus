'use client'

import Link from 'next/link'
import {useRouter} from 'next/navigation'
import {SidebarTrigger, useSidebar} from '@/components/ui/sidebar'
import {ManusSettings} from '@/components/manus-settings'
import {Button} from '@/components/ui/button'
import {useAuth} from '@/hooks/use-auth'

interface ChatHeaderProps {
  onLoginClick?: () => void
}

export function ChatHeader({onLoginClick}: ChatHeaderProps) {
  const router = useRouter()
  const {isLoggedIn} = useAuth()
  const {open, isMobile} = useSidebar()
  const handleLoginClick = () => {
    if (onLoginClick) {
      onLoginClick()
      return
    }
    router.push('/auth/login')
  }

  return (
    <header className="flex justify-between items-center w-full py-2 px-4 z-50">
      {/* 左侧操作&logo */}
      <div className="flex items-center gap-2">
        {/* 面板操作按钮: 关闭面板&移动端下会显示 */}
        {isLoggedIn && (!open || isMobile) && <SidebarTrigger className="cursor-pointer"/>}
        {/* Logo占位符 */}
        <Link href="/" className="block bg-white w-[80px] h-9 rounded-md"/>
      </div>
      {/* 右侧：未登录显示登录按钮，已登录显示设置入口 */}
      {isLoggedIn ? (
        <ManusSettings/>
      ) : (
        <Button
          variant="outline"
          size="sm"
          className="cursor-pointer"
          onClick={handleLoginClick}
        >
          登录
        </Button>
      )}
    </header>
  )
}
