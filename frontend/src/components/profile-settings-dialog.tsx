"use client"

import { type ChangeEvent, useCallback, useEffect, useMemo, useRef, useState } from "react"
import { ImagePlus, Loader2, LockKeyhole, UserRound } from "lucide-react"
import { toast } from "sonner"
import { getApiErrorMessage, isApiErrorKey } from "@/lib/api"
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
import { useI18n } from "@/lib/i18n"
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
  const { locale, t } = useI18n()
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
    () => ensureOptionExists(getLocaleOptions(locale), profileForm.locale),
    [locale, profileForm.locale],
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
      const message = getApiErrorMessage(error, "profile.loadFailed", t)
      setProfileError(message)
      toast.error(message)
    } finally {
      setLoadingProfile(false)
    }
  }, [syncProfileFromUser, t])

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
      toast.error(t("profile.avatarImageOnly"))
      return
    }
    if (file.size > 5 * 1024 * 1024) {
      toast.error(t("profile.avatarTooLarge"))
      return
    }

    setUploadingAvatar(true)
    try {
      const uploaded = await fileApi.uploadFile({ file })
      setProfileForm((prev) => ({
        ...prev,
        avatar_url: toAvatarFileRef(uploaded.id),
      }))
      toast.success(t("profile.avatarUploadSuccess"))
    } catch (error) {
      const message = getApiErrorMessage(error, "profile.avatarUploadFailed", t)
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
      setTimezoneError(t("profile.timezoneRequired"))
      hasError = true
    }
    if (!locale) {
      setLocaleError(t("profile.localeRequired"))
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
      toast.info(t("profile.noChanges"))
      return
    }

    setSavingProfile(true)
    try {
      const result = await authApi.updateCurrentUser(payload)
      setCurrentUser(result.user)
      syncProfileFromUser(result.user)
      toast.success(t("profile.updateSuccess"))
    } catch (error) {
      const message = getApiErrorMessage(error, "profile.updateFailed", t)
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
      setOldPasswordError(t("profile.oldPasswordRequired"))
      hasError = true
    }

    const nextPasswordError = validatePasswordStrength(newPassword)
    if (nextPasswordError) {
      setNewPasswordError(t(nextPasswordError))
      hasError = true
    }

    if (!confirmPassword) {
      setConfirmPasswordError(t("profile.confirmNewPasswordRequired"))
      hasError = true
    } else if (confirmPassword !== newPassword) {
      setConfirmPasswordError(t("profile.newPasswordMismatch"))
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
      toast.success(t("profile.passwordUpdateSuccess"))
      onOpenChange(false)
      clearAuthenticatedSession()
      window.location.assign("/?auth=login")
    } catch (error) {
      const message = getApiErrorMessage(error, "profile.passwordUpdateFailed", t)
      if (isApiErrorKey(error, "error.user.current_password_incorrect")) {
        setOldPasswordError(getApiErrorMessage(error, "profile.currentPasswordWrong", t))
      } else if (isApiErrorKey(error, "error.user.new_password_mismatch")) {
        setConfirmPasswordError(getApiErrorMessage(error, "profile.newPasswordMismatch", t))
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
              <AvatarImage src={avatarSrc} alt={t("profile.avatarAlt")} />
              <AvatarFallback className="text-base">
                {getUserInitial(profileForm)}
              </AvatarFallback>
            </Avatar>
            <div className="space-y-1">
              <p className="text-sm font-medium text-gray-700">{t("profile.avatarLabel")}</p>
              <p className="text-xs text-gray-500">{t("profile.avatarHint")}</p>
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
              {t("profile.uploadAvatar")}
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
          <Label htmlFor="profile-email">{t("profile.emailLabel")}</Label>
          <Input id="profile-email" value={profileForm.email} disabled />
        </div>
        <div className="space-y-2 sm:col-span-2">
          <Label htmlFor="profile-nickname">{t("profile.nicknameLabel")}</Label>
          <Input
            id="profile-nickname"
            placeholder={t("profile.nicknamePlaceholder")}
            value={profileForm.nickname}
            onChange={(event) => handleProfileInputChange("nickname", event.target.value)}
          />
        </div>
        <div className="space-y-2">
          <Label>{t("profile.timezoneLabel")}</Label>
          <SearchableSelect
            value={profileForm.timezone}
            options={timeZoneOptions}
            placeholder={t("profile.timezonePlaceholder")}
            searchPlaceholder={t("profile.timezoneSearchPlaceholder")}
            emptyText={t("profile.timezoneEmpty")}
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
          <Label>{t("profile.localeLabel")}</Label>
          <SearchableSelect
            value={profileForm.locale}
            options={localeOptions}
            placeholder={t("profile.localePlaceholder")}
            searchPlaceholder={t("profile.localeSearchPlaceholder")}
            emptyText={t("profile.localeEmpty")}
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
          {t("profile.saveProfile")}
        </Button>
      </div>
    </div>
  )

  const renderPasswordContent = () => (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label htmlFor="profile-old-password">{t("profile.oldPasswordLabel")}</Label>
        <Input
          id="profile-old-password"
          type="password"
          placeholder={t("profile.oldPasswordPlaceholder")}
          value={oldPassword}
          onChange={(event) => setOldPassword(event.target.value)}
        />
        {oldPasswordError ? <p className="text-xs text-red-500">{oldPasswordError}</p> : null}
      </div>
      <div className="space-y-2">
        <Label htmlFor="profile-new-password">{t("profile.newPasswordLabel")}</Label>
        <Input
          id="profile-new-password"
          type="password"
          placeholder={t("profile.newPasswordPlaceholder")}
          value={newPassword}
          onChange={(event) => setNewPassword(event.target.value)}
        />
        {newPasswordError ? <p className="text-xs text-red-500">{newPasswordError}</p> : null}
      </div>
      <div className="space-y-2">
        <Label htmlFor="profile-confirm-password">{t("profile.confirmPasswordLabel")}</Label>
        <Input
          id="profile-confirm-password"
          type="password"
          placeholder={t("profile.confirmPasswordPlaceholder")}
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
          {t("profile.updatePassword")}
        </Button>
      </div>
    </div>
  )

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="!max-w-[850px]">
        <DialogHeader className="border-b pb-4">
          <DialogTitle className="text-gray-700">{t("profile.dialogTitle")}</DialogTitle>
          <DialogDescription className="text-gray-500">
            {t("profile.dialogDescription")}
          </DialogDescription>
        </DialogHeader>

        {loadingProfile ? (
          <div className="flex min-h-[360px] items-center justify-center text-gray-500">
            <Loader2 className="size-5 animate-spin" />
            <span className="ml-2 text-sm">{t("profile.loading")}</span>
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
                  {t("profile.tabProfile")}
                </Button>
                <Button
                  variant={activeTab === "password" ? "default" : "ghost"}
                  className="cursor-pointer justify-start"
                  onClick={() => setActiveTab("password")}
                >
                  <LockKeyhole />
                  {t("profile.tabPassword")}
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
