"use client"

import { useMemo, useState } from "react"
import { ChevronsUpDown, LogOut, Settings, UserRound } from "lucide-react"
import { toast } from "sonner"
import { useAuth } from "@/hooks/use-auth"
import { getUserDisplayName, maskEmail } from "@/lib/auth/display"
import { useAvatarSrc } from "@/hooks/use-avatar-src"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { ManusSettings } from "@/components/manus-settings"
import { ProfileSettingsDialog } from "@/components/profile-settings-dialog"

export function LeftPanelAccountMenu() {
  const { user, logout, isLoggedIn } = useAuth()

  const [settingsOpen, setSettingsOpen] = useState(false)
  const [profileOpen, setProfileOpen] = useState(false)
  const [loggingOut, setLoggingOut] = useState(false)
  const avatarSrc = useAvatarSrc(user?.avatar_url)

  const displayName = useMemo(() => {
    if (!user && isLoggedIn) {
      return "加载中..."
    }
    return getUserDisplayName(user)
  }, [isLoggedIn, user])
  const secondaryLabel = useMemo(() => {
    if (!user && isLoggedIn) {
      return "正在加载资料..."
    }
    if (!user) {
      return ""
    }
    if (user.nickname?.trim()) {
      return maskEmail(user.email)
    }
    return ""
  }, [isLoggedIn, user])

  const openSettings = () => {
    setProfileOpen(false)
    setSettingsOpen(true)
  }

  const openProfile = () => {
    setSettingsOpen(false)
    setProfileOpen(true)
  }

  const handleLogout = async () => {
    if (loggingOut) {
      return
    }
    setLoggingOut(true)
    try {
      await logout()
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "退出登录失败，请稍后重试")
    } finally {
      setLoggingOut(false)
    }
  }

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            type="button"
            variant="ghost"
            className="h-14 w-full cursor-pointer items-center justify-start gap-3 rounded-lg border border-transparent bg-transparent px-2.5 py-2 shadow-none hover:bg-gray-100/70 data-[state=open]:bg-gray-100/80"
          >
            <Avatar className="size-9 shrink-0 border border-gray-200/60">
              <AvatarImage src={avatarSrc} alt={displayName} />
              <AvatarFallback className="bg-gray-100/80 text-sm font-medium text-gray-600">
                {displayName.slice(0, 1)}
              </AvatarFallback>
            </Avatar>
            <div className="min-w-0 flex-1 text-left">
              <p className="truncate text-sm font-medium text-gray-700">{displayName}</p>
              {secondaryLabel ? (
                <p className="truncate text-xs text-gray-500/90">{secondaryLabel}</p>
              ) : null}
            </div>
            <ChevronsUpDown className="size-4 shrink-0 text-gray-300" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent
          side="top"
          align="start"
          className="mb-2 w-[var(--radix-dropdown-menu-trigger-width)] min-w-[230px]"
        >
          <DropdownMenuItem className="cursor-pointer" onClick={openSettings}>
            <Settings />
            系统设置
          </DropdownMenuItem>
          <DropdownMenuItem className="cursor-pointer" onClick={openProfile}>
            <UserRound />
            个人中心
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem
            variant="destructive"
            className="cursor-pointer"
            onClick={handleLogout}
            disabled={loggingOut}
          >
            <LogOut />
            {loggingOut ? "退出中..." : "退出登录"}
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      <ManusSettings
        open={settingsOpen}
        onOpenChange={setSettingsOpen}
        showTrigger={false}
      />
      <ProfileSettingsDialog open={profileOpen} onOpenChange={setProfileOpen} />
    </>
  )
}
