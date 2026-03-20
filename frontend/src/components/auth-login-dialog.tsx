'use client'

import Image from 'next/image'
import {type FormEvent, useEffect, useMemo, useState} from 'react'
import {Button} from '@/components/ui/button'
import {Input} from '@/components/ui/input'
import {Label} from '@/components/ui/label'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import {useAuth} from '@/hooks/use-auth'
import {authApi} from '@/lib/auth'
import {useI18n} from '@/lib/i18n'
import {validatePasswordStrength} from '@/lib/auth/validators'
import {getApiErrorMessage, isApiErrorKey} from '@/lib/api'
import {toast} from 'sonner'

const EMAIL_REGEX = /^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$/

export type AuthDialogMode = 'login' | 'register'

type AuthLoginDialogProps = {
  open: boolean
  onOpenChange: (open: boolean) => void
  initialEmail?: string
  initialMode?: AuthDialogMode
  onModeChange?: (mode: AuthDialogMode) => void
  onSuccess?: () => void
}

export function AuthLoginDialog({
  open,
  onOpenChange,
  initialEmail,
  initialMode = 'login',
  onModeChange,
  onSuccess,
}: AuthLoginDialogProps) {
  const {login, register} = useAuth()
  const {t} = useI18n()

  const [mode, setMode] = useState<AuthDialogMode>(initialMode)
  const [email, setEmail] = useState('')
  const [loginPassword, setLoginPassword] = useState('')
  const [registerPassword, setRegisterPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [verificationCode, setVerificationCode] = useState('')

  const [submitting, setSubmitting] = useState(false)
  const [sendingCode, setSendingCode] = useState(false)
  const [verificationRequired, setVerificationRequired] = useState(false)
  const [verificationCooldown, setVerificationCooldown] = useState(0)

  const [emailError, setEmailError] = useState<string | null>(null)
  const [passwordError, setPasswordError] = useState<string | null>(null)
  const [confirmPasswordError, setConfirmPasswordError] = useState<string | null>(null)
  const [verificationCodeError, setVerificationCodeError] = useState<string | null>(null)
  const [submitError, setSubmitError] = useState<string | null>(null)

  useEffect(() => {
    if (open) {
      setMode(initialMode)
      if (typeof initialEmail === 'string' && initialEmail.trim().length > 0) {
        setEmail(initialEmail.trim())
      }
      return
    }

    setLoginPassword('')
    setRegisterPassword('')
    setConfirmPassword('')
    setVerificationCode('')
    setSubmitting(false)
    setSendingCode(false)
    setVerificationRequired(false)
    setVerificationCooldown(0)
    setEmailError(null)
    setPasswordError(null)
    setConfirmPasswordError(null)
    setVerificationCodeError(null)
    setSubmitError(null)
  }, [open, initialEmail, initialMode])

  useEffect(() => {
    if (verificationCooldown <= 0) {
      return
    }

    const timer = window.setInterval(() => {
      setVerificationCooldown((prev) => (prev > 0 ? prev - 1 : 0))
    }, 1000)

    return () => {
      window.clearInterval(timer)
    }
  }, [verificationCooldown])

  const canLoginSubmit = useMemo(() => {
    return email.trim().length > 0 && loginPassword.length > 0 && !submitting
  }, [email, loginPassword, submitting])

  const canRegisterSubmit = useMemo(() => {
    if (submitting || sendingCode) {
      return false
    }
    if (!email.trim() || !registerPassword || !confirmPassword) {
      return false
    }
    if (verificationRequired && !verificationCode.trim()) {
      return false
    }
    return true
  }, [
    submitting,
    sendingCode,
    email,
    registerPassword,
    confirmPassword,
    verificationRequired,
    verificationCode,
  ])

  const clearFormErrors = () => {
    setEmailError(null)
    setPasswordError(null)
    setConfirmPasswordError(null)
    setVerificationCodeError(null)
    setSubmitError(null)
  }

  const switchMode = (nextMode: AuthDialogMode) => {
    if (nextMode === mode) {
      return
    }
    setMode(nextMode)
    onModeChange?.(nextMode)
    clearFormErrors()
    setLoginPassword('')
    setRegisterPassword('')
    setConfirmPassword('')
    setVerificationCode('')
    setVerificationRequired(false)
    setVerificationCooldown(0)
  }

  const validateEmail = (): string | null => {
    const normalizedEmail = email.trim()
    if (!normalizedEmail) {
      return t('authDialog.emailRequired')
    }
    if (!EMAIL_REGEX.test(normalizedEmail)) {
      return t('authDialog.emailInvalid')
    }
    return null
  }

  const validateLoginForm = (): boolean => {
    let hasError = false

    const nextEmailError = validateEmail()
    if (nextEmailError) {
      setEmailError(nextEmailError)
      hasError = true
    } else {
      setEmailError(null)
    }

    if (!loginPassword) {
      setPasswordError(t('authDialog.passwordRequired'))
      hasError = true
    } else {
      setPasswordError(null)
    }

    return !hasError
  }

  const validateRegisterForm = (): boolean => {
    let hasError = false

    const nextEmailError = validateEmail()
    if (nextEmailError) {
      setEmailError(nextEmailError)
      hasError = true
    } else {
      setEmailError(null)
    }

    const nextPasswordError = validatePasswordStrength(registerPassword)
    if (nextPasswordError) {
      setPasswordError(t(nextPasswordError))
      hasError = true
    } else {
      setPasswordError(null)
    }

    if (!confirmPassword) {
      setConfirmPasswordError(t('authDialog.confirmPasswordRequired'))
      hasError = true
    } else if (confirmPassword !== registerPassword) {
      setConfirmPasswordError(t('authDialog.confirmPasswordMismatch'))
      hasError = true
    } else {
      setConfirmPasswordError(null)
    }

    if (verificationRequired && verificationCode.trim().length === 0) {
      setVerificationCodeError(t('authDialog.verificationCodeRequired'))
      hasError = true
    } else {
      setVerificationCodeError(null)
    }

    return !hasError
  }

  const handleSendCode = async () => {
    setSubmitError(null)
    setVerificationCodeError(null)

    const nextEmailError = validateEmail()
    if (nextEmailError) {
      setEmailError(nextEmailError)
      return
    }

    if (sendingCode || verificationCooldown > 0) {
      return
    }

    setSendingCode(true)
    try {
      const result = await authApi.sendRegisterCode({
        email: email.trim().toLowerCase(),
      })

      setVerificationRequired(result.verification_required)
      if (result.verification_required) {
        setVerificationCooldown(60)
        toast.success(t('authDialog.sendCodeSuccess'))
      } else {
        toast.info(t('authDialog.sendCodeNotRequired'))
      }
    } catch (error) {
      setSubmitError(getApiErrorMessage(error, 'authDialog.sendCodeFailed', t))
    } finally {
      setSendingCode(false)
    }
  }

  const handleLoginSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setSubmitError(null)

    if (!validateLoginForm()) {
      return
    }

    setSubmitting(true)
    try {
      await login({
        email: email.trim(),
        password: loginPassword,
      })
      onOpenChange(false)
      onSuccess?.()
    } catch (error) {
      setSubmitError(getApiErrorMessage(error, 'authDialog.loginFailed', t))
      setLoginPassword('')
    } finally {
      setSubmitting(false)
    }
  }

  const handleRegisterSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setSubmitError(null)

    if (!validateRegisterForm()) {
      return
    }

    setSubmitting(true)
    try {
      await register({
        email: email.trim().toLowerCase(),
        password: registerPassword,
        confirm_password: confirmPassword,
        verification_code: verificationCode.trim() || undefined,
      })

      toast.success(t('authDialog.registerSuccess'))
      setLoginPassword('')
      setRegisterPassword('')
      setConfirmPassword('')
      setVerificationCode('')
      setVerificationRequired(false)
      setVerificationCooldown(0)
      switchMode('login')
    } catch (error) {
      if (isApiErrorKey(error, 'error.auth.email_already_registered')) {
        setEmailError(getApiErrorMessage(error, 'authDialog.registerFailed', t))
      } else if (isApiErrorKey(error, 'error.auth.password_mismatch')) {
        setConfirmPasswordError(getApiErrorMessage(error, 'authDialog.confirmPasswordMismatch', t))
      } else if (isApiErrorKey(error, 'error.auth.register_verification_code_required')) {
        setVerificationRequired(true)
        setVerificationCodeError(getApiErrorMessage(error, 'authDialog.verificationCodeRequired', t))
      } else if (isApiErrorKey(error, 'error.auth.register_verification_code_invalid')) {
        setVerificationRequired(true)
        setVerificationCodeError(
          getApiErrorMessage(error, 'authDialog.verificationCodeInvalidOrExpired', t),
        )
      } else {
        setSubmitError(getApiErrorMessage(error, 'authDialog.registerFailed', t))
      }
    } finally {
      setSubmitting(false)
    }
  }

  const title = mode === 'login' ? t('authDialog.login.title') : t('authDialog.register.title')
  const description =
    mode === 'login'
      ? t('authDialog.login.description')
      : t('authDialog.register.description')
  const modeSwitchBaseClassName =
    'h-10 rounded-xl text-sm font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-900/20'
  const formInputClassName =
    'h-11 rounded-xl border-zinc-200 bg-white/90 text-[15px] placeholder:text-zinc-400 focus-visible:border-zinc-900 focus-visible:ring-zinc-900/15'
  const submitErrorClassName =
    'rounded-xl border border-destructive/25 bg-destructive/10 px-3 py-2 text-sm text-destructive'

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="max-w-[calc(100%-1rem)] overflow-hidden border-0 bg-transparent p-0 shadow-none sm:max-w-[560px]"
        showCloseButton
      >
        <div className="relative overflow-hidden rounded-3xl border border-zinc-200/80 bg-white/90 shadow-[0_36px_80px_-44px_rgba(24,24,27,0.55)] backdrop-blur-xl">
          <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(120%_95%_at_50%_-20%,rgba(17,24,39,0.14),transparent_58%),radial-gradient(70%_70%_at_100%_100%,rgba(15,23,42,0.08),transparent_68%)]" />

          <div className="relative p-5 sm:p-8">
            <div className="mb-6 flex items-center gap-3">
              <div className="relative size-12 overflow-hidden rounded-2xl border border-zinc-200 bg-white shadow-sm">
                <Image
                  src="/logo.svg"
                  alt="Actus"
                  fill
                  sizes="48px"
                  className="object-contain p-1"
                />
              </div>
              <div className="space-y-0.5">
                <p className="text-xs font-semibold tracking-[0.25em] text-zinc-500">ACTUS</p>
                <p className="text-sm text-zinc-600">Agent Workspace</p>
              </div>
            </div>

            <DialogHeader className="mb-5 space-y-4 text-left">
              <div className="space-y-1">
                <DialogTitle className="text-2xl font-semibold text-zinc-900">{title}</DialogTitle>
                <DialogDescription className="text-sm leading-6 text-zinc-600">
                  {description}
                </DialogDescription>
              </div>

              <div className="grid grid-cols-2 rounded-2xl border border-zinc-200 bg-zinc-100/70 p-1">
                <button
                  type="button"
                  className={`${modeSwitchBaseClassName} ${
                    mode === 'login'
                      ? 'bg-white text-zinc-900 shadow-sm'
                      : 'text-zinc-500 hover:text-zinc-700'
                  }`}
                  onClick={() => switchMode('login')}
                >
                  {t('authDialog.loginAction')}
                </button>
                <button
                  type="button"
                  className={`${modeSwitchBaseClassName} ${
                    mode === 'register'
                      ? 'bg-white text-zinc-900 shadow-sm'
                      : 'text-zinc-500 hover:text-zinc-700'
                  }`}
                  onClick={() => switchMode('register')}
                >
                  {t('authDialog.registerAction')}
                </button>
              </div>
            </DialogHeader>

            {mode === 'login' ? (
              <form className="space-y-4" onSubmit={handleLoginSubmit}>
                <div className="space-y-2">
                  <Label htmlFor="login-email" className="text-xs font-medium tracking-wide text-zinc-600">
                    {t('authDialog.emailLabel')}
                  </Label>
                  <Input
                    id="login-email"
                    type="email"
                    autoComplete="email"
                    className={formInputClassName}
                    placeholder={t('authDialog.placeholder.email')}
                    value={email}
                    onChange={(event) => {
                      setEmail(event.target.value)
                      if (emailError) {
                        setEmailError(null)
                      }
                    }}
                    aria-invalid={emailError ? 'true' : 'false'}
                    disabled={submitting}
                  />
                  {emailError && <p className="text-sm text-destructive">{emailError}</p>}
                </div>

                <div className="space-y-2">
                  <Label htmlFor="login-password" className="text-xs font-medium tracking-wide text-zinc-600">
                    {t('authDialog.passwordLabel')}
                  </Label>
                  <Input
                    id="login-password"
                    type="password"
                    autoComplete="current-password"
                    className={formInputClassName}
                    placeholder={t('authDialog.placeholder.password')}
                    value={loginPassword}
                    onChange={(event) => {
                      setLoginPassword(event.target.value)
                      if (passwordError) {
                        setPasswordError(null)
                      }
                    }}
                    aria-invalid={passwordError ? 'true' : 'false'}
                    disabled={submitting}
                  />
                  {passwordError && <p className="text-sm text-destructive">{passwordError}</p>}
                </div>

                {submitError && <p className={submitErrorClassName}>{submitError}</p>}

                <Button
                  type="submit"
                  className="h-11 w-full rounded-xl bg-zinc-900 text-white shadow-[0_16px_30px_-20px_rgba(24,24,27,0.8)] hover:bg-zinc-800"
                  disabled={!canLoginSubmit}
                >
                  {submitting ? t('authDialog.loginSubmitting') : t('authDialog.loginAction')}
                </Button>

                <p className="text-center text-sm text-zinc-500">
                  {t('authDialog.noAccount')}
                  <button
                    type="button"
                    className="ml-1 font-medium text-zinc-900 underline-offset-4 hover:underline"
                    onClick={() => switchMode('register')}
                  >
                    {t('authDialog.goRegister')}
                  </button>
                </p>
              </form>
            ) : (
              <form className="space-y-4" onSubmit={handleRegisterSubmit}>
                <div className="space-y-2">
                  <Label htmlFor="register-email" className="text-xs font-medium tracking-wide text-zinc-600">
                    {t('authDialog.emailLabel')}
                  </Label>
                  <Input
                    id="register-email"
                    type="email"
                    autoComplete="email"
                    className={formInputClassName}
                    placeholder={t('authDialog.placeholder.email')}
                    value={email}
                    onChange={(event) => {
                      setEmail(event.target.value)
                      if (emailError) {
                        setEmailError(null)
                      }
                    }}
                    disabled={submitting}
                    aria-invalid={emailError ? 'true' : 'false'}
                  />
                  {emailError ? (
                    <p className="text-sm text-destructive">{emailError}</p>
                  ) : null}
                </div>

                <div className="space-y-2">
                  <Button
                    type="button"
                    variant="outline"
                    className="h-11 w-full rounded-xl border-zinc-200 bg-white/80 text-zinc-800 shadow-none hover:bg-zinc-50"
                    onClick={() => {
                      void handleSendCode()
                    }}
                    disabled={submitting || sendingCode || verificationCooldown > 0}
                  >
                    {sendingCode
                      ? t('authDialog.sendCodeSending')
                      : verificationCooldown > 0
                        ? t('authDialog.sendCodeResendAfter', {seconds: verificationCooldown})
                        : t('authDialog.sendCodeAction')}
                  </Button>
                </div>

                {verificationRequired && (
                  <div className="space-y-2">
                    <Label
                      htmlFor="register-verification-code"
                      className="text-xs font-medium tracking-wide text-zinc-600"
                    >
                      {t('authDialog.verificationCodeLabel')}
                    </Label>
                    <Input
                      id="register-verification-code"
                      type="text"
                      inputMode="numeric"
                      autoComplete="one-time-code"
                      className={formInputClassName}
                      placeholder={t('authDialog.placeholder.verificationCode')}
                      value={verificationCode}
                      onChange={(event) => {
                        setVerificationCode(event.target.value)
                        if (verificationCodeError) {
                          setVerificationCodeError(null)
                        }
                      }}
                      disabled={submitting}
                      aria-invalid={verificationCodeError ? 'true' : 'false'}
                    />
                    {verificationCodeError && (
                      <p className="text-sm text-destructive">{verificationCodeError}</p>
                    )}
                  </div>
                )}

                <div className="space-y-2">
                  <Label htmlFor="register-password" className="text-xs font-medium tracking-wide text-zinc-600">
                    {t('authDialog.passwordLabel')}
                  </Label>
                  <Input
                    id="register-password"
                    type="password"
                    autoComplete="new-password"
                    className={formInputClassName}
                    placeholder={t('authDialog.placeholder.password')}
                    value={registerPassword}
                    onChange={(event) => {
                      setRegisterPassword(event.target.value)
                      if (passwordError) {
                        setPasswordError(null)
                      }
                    }}
                    disabled={submitting}
                    aria-invalid={passwordError ? 'true' : 'false'}
                  />
                  {passwordError ? (
                    <p className="text-sm text-destructive">{passwordError}</p>
                  ) : (
                    <p className="text-xs leading-5 text-zinc-500">{t('authDialog.passwordHint')}</p>
                  )}
                </div>

                <div className="space-y-2">
                  <Label
                    htmlFor="register-confirm-password"
                    className="text-xs font-medium tracking-wide text-zinc-600"
                  >
                    {t('authDialog.confirmPasswordLabel')}
                  </Label>
                  <Input
                    id="register-confirm-password"
                    type="password"
                    autoComplete="new-password"
                    className={formInputClassName}
                    placeholder={t('authDialog.confirmPasswordRequired')}
                    value={confirmPassword}
                    onChange={(event) => {
                      setConfirmPassword(event.target.value)
                      if (confirmPasswordError) {
                        setConfirmPasswordError(null)
                      }
                    }}
                    disabled={submitting}
                    aria-invalid={confirmPasswordError ? 'true' : 'false'}
                  />
                  {confirmPasswordError && (
                    <p className="text-sm text-destructive">{confirmPasswordError}</p>
                  )}
                </div>

                {submitError && <p className={submitErrorClassName}>{submitError}</p>}

                <Button
                  type="submit"
                  className="h-11 w-full rounded-xl bg-zinc-900 text-white shadow-[0_16px_30px_-20px_rgba(24,24,27,0.8)] hover:bg-zinc-800"
                  disabled={!canRegisterSubmit}
                >
                  {submitting ? t('authDialog.registerSubmitting') : t('authDialog.registerAction')}
                </Button>

                <p className="text-center text-sm text-zinc-500">
                  {t('authDialog.hasAccount')}
                  <button
                    type="button"
                    className="ml-1 font-medium text-zinc-900 underline-offset-4 hover:underline"
                    onClick={() => switchMode('login')}
                  >
                    {t('authDialog.goLogin')}
                  </button>
                </p>
              </form>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
