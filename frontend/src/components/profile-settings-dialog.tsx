"use client"

import { type ChangeEvent, useCallback, useEffect, useMemo, useRef, useState } from "react"
import { ImagePlus, Loader2, LockKeyhole, UserRound } from "lucide-react"
import { toast } from "sonner"
import { ApiError } from "@/lib/api"
import { fileApi } from "@/lib/api/file"
import {
  authApi,
  clearAuthenticatedSession,
  setCurrentUser,
  toAvatarFileRef,
} from "@/lib/auth"
import {
  ensureOptionExists,
  getLocaleOptions,
  getTimeZoneOptions,
} from "@/lib/auth/options"
import type { AuthUser } from "@/lib/auth"
import { validatePasswordStrength } from "@/lib/auth/validators"
import { useAvatarSrc } from "@/hooks/use-avatar-src"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { SearchableSelect } from "@/components/ui/searchable-select"
import { Separator } from "@/components/ui/separator"

type ProfileSettingsDialogProps = {
  open: boolean
  onOpenChange: (open: boolean) => void
}

type ProfileTab = "profile" | "password"

type ProfileFormState = {
  email: string
  nickname: string
  avatar_url: string
  timezone: string
  locale: string
}

const EMPTY_PROFILE: ProfileFormState = {
  email: "",
  nickname: "",
  avatar_url: "",
  timezone: "",
  locale: "",
}

function buildProfileForm(user: AuthUser): ProfileFormState {
  return {
    email: user.email,
    nickname: user.nickname ?? "",
    avatar_url: user.avatar_url ?? "",
    timezone: user.timezone,
    locale: user.locale,
  }
}

function normalizeTextField(value: string): string | null {
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : null
}

function getUserInitial(user: ProfileFormState): string {
  const nickname = user.nickname.trim()
  if (nickname) {
    return nickname.slice(0, 1)
  }
  const email = user.email.trim()
  if (!email) {
    return "U"
  }
  return email.slice(0, 1).toUpperCase()
}

export function ProfileSettingsDialog({
  open,
  onOpenChange,
}: ProfileSettingsDialogProps) {
  const [activeTab, setActiveTab] = useState<ProfileTab>("profile")

  const [loadingProfile, setLoadingProfile] = useState(false)
  const [savingProfile, setSavingProfile] = useState(false)
  const [savingPassword, setSavingPassword] = useState(false)
  const [uploadingAvatar, setUploadingAvatar] = useState(false)

  const [profileForm, setProfileForm] = useState<ProfileFormState>(EMPTY_PROFILE)
  const [profileSnapshot, setProfileSnapshot] = useState<ProfileFormState>(EMPTY_PROFILE)

  const [oldPassword, setOldPassword] = useState("")
  const [newPassword, setNewPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")

  const [profileError, setProfileError] = useState<string | null>(null)
  const [timezoneError, setTimezoneError] = useState<string | null>(null)
  const [localeError, setLocaleError] = useState<string | null>(null)
  const [oldPasswordError, setOldPasswordError] = useState<string | null>(null)
  const [newPasswordError, setNewPasswordError] = useState<string | null>(null)
  const [confirmPasswordError, setConfirmPasswordError] = useState<string | null>(null)

  const avatarFileInputRef = useRef<HTMLInputElement | null>(null)
  const avatarSrc = useAvatarSrc(profileForm.avatar_url)

  const timeZoneOptions = useMemo(
    () => ensureOptionExists(getTimeZoneOptions(), profileForm.timezone),
    [profileForm.timezone],
  )
  const localeOptions = useMemo(
    () => ensureOptionExists(getLocaleOptions(), profileForm.locale),
    [profileForm.locale],
  )

  const resetPasswordForm = useCallback(() => {
    setOldPassword("")
    setNewPassword("")
    setConfirmPassword("")
    setOldPasswordError(null)
    setNewPasswordError(null)
    setConfirmPasswordError(null)
  }, [])

  const syncProfileFromUser = useCallback((user: AuthUser) => {
    const next = buildProfileForm(user)
    setProfileForm(next)
    setProfileSnapshot(next)
  }, [])

  const loadProfile = useCallback(async () => {
    setLoadingProfile(true)
    setProfileError(null)
    try {
      const user = await authApi.me()
      setCurrentUser(user)
      syncProfileFromUser(user)
    } catch (error) {
      const message =
        error instanceof ApiError ? error.msg : "获取个人资料失败，请稍后重试"
      setProfileError(message)
      toast.error(message)
    } finally {
      setLoadingProfile(false)
    }
  }, [syncProfileFromUser])

  useEffect(() => {
    if (!open) {
      resetPasswordForm()
      setActiveTab("profile")
      return
    }
    void loadProfile()
  }, [open, loadProfile, resetPasswordForm])

  const canSubmitPassword = useMemo(() => {
    return (
      oldPassword.trim().length > 0 &&
      newPassword.trim().length > 0 &&
      confirmPassword.trim().length > 0 &&
      !savingPassword
    )
  }, [confirmPassword, newPassword, oldPassword, savingPassword])

  const hasProfileChanges = useMemo(() => {
    return (
      normalizeTextField(profileForm.nickname) !== normalizeTextField(profileSnapshot.nickname) ||
      normalizeTextField(profileForm.avatar_url) !== normalizeTextField(profileSnapshot.avatar_url) ||
      profileForm.timezone.trim() !== profileSnapshot.timezone.trim() ||
      profileForm.locale.trim() !== profileSnapshot.locale.trim()
    )
  }, [profileForm, profileSnapshot])

  const handleProfileInputChange = (
    field: keyof ProfileFormState,
    value: string,
  ) => {
    setProfileForm((prev) => ({ ...prev, [field]: value }))
  }

  const handleAvatarUploadClick = () => {
    avatarFileInputRef.current?.click()
  }

  const handleAvatarFileChange = async (
    event: ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0]
    event.target.value = ""
    if (!file) {
      return
    }
    if (!file.type.startsWith("image/")) {
      toast.error("头像仅支持图片格式")
      return
    }
    if (file.size > 5 * 1024 * 1024) {
      toast.error("头像大小不能超过 5MB")
      return
    }

    setUploadingAvatar(true)
    try {
      const uploaded = await fileApi.uploadFile({ file })
      setProfileForm((prev) => ({
        ...prev,
        avatar_url: toAvatarFileRef(uploaded.id),
      }))
      toast.success("头像上传成功")
    } catch (error) {
      const message = error instanceof ApiError ? error.msg : "头像上传失败，请稍后重试"
      toast.error(message)
    } finally {
      setUploadingAvatar(false)
    }
  }

  const handleSaveProfile = async () => {
    setProfileError(null)
    setTimezoneError(null)
    setLocaleError(null)

    const timezone = profileForm.timezone.trim()
    const locale = profileForm.locale.trim()

    let hasError = false
    if (!timezone) {
      setTimezoneError("请选择时区")
      hasError = true
    }
    if (!locale) {
      setLocaleError("请选择语言")
      hasError = true
    }
    if (hasError) {
      return
    }

    const payload: {
      nickname?: string | null
      avatar_url?: string | null
      timezone?: string
      locale?: string
    } = {}

    const nextNickname = normalizeTextField(profileForm.nickname)
    const prevNickname = normalizeTextField(profileSnapshot.nickname)
    if (nextNickname !== prevNickname) {
      payload.nickname = nextNickname
    }

    const nextAvatar = normalizeTextField(profileForm.avatar_url)
    const prevAvatar = normalizeTextField(profileSnapshot.avatar_url)
    if (nextAvatar !== prevAvatar) {
      payload.avatar_url = nextAvatar
    }

    if (timezone !== profileSnapshot.timezone.trim()) {
      payload.timezone = timezone
    }
    if (locale !== profileSnapshot.locale.trim()) {
      payload.locale = locale
    }

    if (Object.keys(payload).length === 0) {
      toast.info("当前没有可保存的变更")
      return
    }

    setSavingProfile(true)
    try {
      const result = await authApi.updateCurrentUser(payload)
      setCurrentUser(result.user)
      syncProfileFromUser(result.user)
      toast.success("资料更新成功")
    } catch (error) {
      const message = error instanceof ApiError ? error.msg : "资料更新失败，请稍后重试"
      setProfileError(message)
      toast.error(message)
    } finally {
      setSavingProfile(false)
    }
  }

  const handleUpdatePassword = async () => {
    setOldPasswordError(null)
    setNewPasswordError(null)
    setConfirmPasswordError(null)

    let hasError = false
    if (!oldPassword) {
      setOldPasswordError("请输入当前密码")
      hasError = true
    }

    const nextPasswordError = validatePasswordStrength(newPassword)
    if (nextPasswordError) {
      setNewPasswordError(nextPasswordError)
      hasError = true
    }

    if (!confirmPassword) {
      setConfirmPasswordError("请再次输入新密码")
      hasError = true
    } else if (confirmPassword !== newPassword) {
      setConfirmPasswordError("两次输入的新密码不一致")
      hasError = true
    }

    if (hasError) {
      return
    }

    setSavingPassword(true)
    try {
      await authApi.updatePassword({
        old_password: oldPassword,
        new_password: newPassword,
        confirm_password: confirmPassword,
      })
      resetPasswordForm()
      toast.success("密码更新成功，请重新登录")
      onOpenChange(false)
      clearAuthenticatedSession()
      window.location.assign("/?auth=login")
    } catch (error) {
      const message = error instanceof ApiError ? error.msg : "密码更新失败，请稍后重试"
      if (message.includes("旧密码")) {
        setOldPasswordError("当前密码错误")
      } else if (message.includes("密码不一致")) {
        setConfirmPasswordError("两次输入的新密码不一致")
      } else {
        setConfirmPasswordError(message)
      }
      toast.error(message)
    } finally {
      setSavingPassword(false)
    }
  }

  const renderProfileContent = () => (
    <div className="space-y-5">
      <section className="rounded-lg border border-gray-200 p-4">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-3">
            <Avatar className="size-14">
              <AvatarImage src={avatarSrc} alt="头像" />
              <AvatarFallback className="text-base">
                {getUserInitial(profileForm)}
              </AvatarFallback>
            </Avatar>
            <div className="space-y-1">
              <p className="text-sm font-medium text-gray-700">头像</p>
              <p className="text-xs text-gray-500">支持 JPG/PNG/WebP，大小不超过 5MB</p>
            </div>
          </div>
          <div className="flex gap-2">
            <Button
              type="button"
              variant="outline"
              className="cursor-pointer"
              onClick={handleAvatarUploadClick}
              disabled={uploadingAvatar}
            >
              {uploadingAvatar ? <Loader2 className="animate-spin" /> : <ImagePlus />}
              上传头像
            </Button>
            <input
              ref={avatarFileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleAvatarFileChange}
            />
          </div>
        </div>
      </section>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div className="space-y-2 sm:col-span-2">
          <Label htmlFor="profile-email">邮箱</Label>
          <Input id="profile-email" value={profileForm.email} disabled />
        </div>
        <div className="space-y-2 sm:col-span-2">
          <Label htmlFor="profile-nickname">昵称</Label>
          <Input
            id="profile-nickname"
            placeholder="请输入昵称"
            value={profileForm.nickname}
            onChange={(event) => handleProfileInputChange("nickname", event.target.value)}
          />
        </div>
        <div className="space-y-2">
          <Label>时区</Label>
          <SearchableSelect
            value={profileForm.timezone}
            options={timeZoneOptions}
            placeholder="请选择时区"
            searchPlaceholder="搜索时区，例如 Shanghai / Tokyo / UTC"
            emptyText="无匹配时区"
            onValueChange={(value) => {
              handleProfileInputChange("timezone", value)
              if (timezoneError) {
                setTimezoneError(null)
              }
            }}
          />
          {timezoneError ? <p className="text-xs text-red-500">{timezoneError}</p> : null}
        </div>
        <div className="space-y-2">
          <Label>语言地区</Label>
          <SearchableSelect
            value={profileForm.locale}
            options={localeOptions}
            placeholder="请选择语言地区"
            searchPlaceholder="搜索语言或地区，例如 中文 / English / 日本语"
            emptyText="无匹配语言地区"
            onValueChange={(value) => {
              handleProfileInputChange("locale", value)
              if (localeError) {
                setLocaleError(null)
              }
            }}
          />
          {localeError ? <p className="text-xs text-red-500">{localeError}</p> : null}
        </div>
      </div>

      {profileError ? <p className="text-xs text-red-500">{profileError}</p> : null}
      <div className="flex justify-end">
        <Button
          type="button"
          onClick={handleSaveProfile}
          disabled={savingProfile || uploadingAvatar || !hasProfileChanges}
          className="cursor-pointer"
        >
          {savingProfile ? <Loader2 className="animate-spin" /> : null}
          保存资料
        </Button>
      </div>
    </div>
  )

  const renderPasswordContent = () => (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label htmlFor="profile-old-password">当前密码</Label>
        <Input
          id="profile-old-password"
          type="password"
          placeholder="请输入当前密码"
          value={oldPassword}
          onChange={(event) => setOldPassword(event.target.value)}
        />
        {oldPasswordError ? <p className="text-xs text-red-500">{oldPasswordError}</p> : null}
      </div>
      <div className="space-y-2">
        <Label htmlFor="profile-new-password">新密码</Label>
        <Input
          id="profile-new-password"
          type="password"
          placeholder="请输入新密码"
          value={newPassword}
          onChange={(event) => setNewPassword(event.target.value)}
        />
        {newPasswordError ? <p className="text-xs text-red-500">{newPasswordError}</p> : null}
      </div>
      <div className="space-y-2">
        <Label htmlFor="profile-confirm-password">确认新密码</Label>
        <Input
          id="profile-confirm-password"
          type="password"
          placeholder="请再次输入新密码"
          value={confirmPassword}
          onChange={(event) => setConfirmPassword(event.target.value)}
        />
        {confirmPasswordError ? (
          <p className="text-xs text-red-500">{confirmPasswordError}</p>
        ) : null}
      </div>
      <div className="flex justify-end">
        <Button
          type="button"
          onClick={handleUpdatePassword}
          disabled={!canSubmitPassword}
          className="cursor-pointer"
        >
          {savingPassword ? <Loader2 className="animate-spin" /> : null}
          更新密码
        </Button>
      </div>
    </div>
  )

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="!max-w-[850px]">
        <DialogHeader className="border-b pb-4">
          <DialogTitle className="text-gray-700">个人中心</DialogTitle>
          <DialogDescription className="text-gray-500">
            管理个人资料与密码安全设置
          </DialogDescription>
        </DialogHeader>

        {loadingProfile ? (
          <div className="flex min-h-[360px] items-center justify-center text-gray-500">
            <Loader2 className="size-5 animate-spin" />
            <span className="ml-2 text-sm">正在加载个人资料...</span>
          </div>
        ) : (
          <div className="flex flex-row gap-4">
            <div className="max-w-[180px]">
              <div className="flex flex-col gap-0">
                <Button
                  variant={activeTab === "profile" ? "default" : "ghost"}
                  className="cursor-pointer justify-start"
                  onClick={() => setActiveTab("profile")}
                >
                  <UserRound />
                  基本资料
                </Button>
                <Button
                  variant={activeTab === "password" ? "default" : "ghost"}
                  className="cursor-pointer justify-start"
                  onClick={() => setActiveTab("password")}
                >
                  <LockKeyhole />
                  修改密码
                </Button>
              </div>
            </div>

            <Separator orientation="vertical" />

            <div className="h-[500px] flex-1 overflow-y-auto pr-1">
              {activeTab === "profile" ? renderProfileContent() : renderPasswordContent()}
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}
