'use client'

import Link from 'next/link'
import {useRouter} from 'next/navigation'
import {SidebarTrigger, useSidebar} from '@/components/ui/sidebar'
import {Button} from '@/components/ui/button'
import {useAuth} from '@/hooks/use-auth'
import {useI18n} from '@/lib/i18n'

interface ChatHeaderProps {
  onLoginClick?: () => void
}

export function ChatHeader({onLoginClick}: ChatHeaderProps) {
  const router = useRouter()
  const {t} = useI18n()
  const {isLoggedIn} = useAuth()
  const {open, isMobile} = useSidebar()
  const handleLoginClick = () => {
    if (onLoginClick) {
      onLoginClick()
      return
    }
    router.push('/?auth=login')
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
      {/* 右侧：未登录显示登录按钮 */}
      {!isLoggedIn ? (
        <Button
          size="default"
          className="cursor-pointer h-9 px-4 font-semibold rounded-full shadow-[0_4px_12px_rgba(0,0,0,0.15)] transition-none hover:bg-primary active:bg-primary hover:shadow-[0_4px_12px_rgba(0,0,0,0.15)] active:shadow-[0_4px_12px_rgba(0,0,0,0.15)]"
          onClick={handleLoginClick}
        >
          {t('chatHeader.login')}
        </Button>
      ) : null}
    </header>
  )
}
