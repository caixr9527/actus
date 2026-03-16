'use client'

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
import {validatePasswordStrength} from '@/lib/auth/validators'
import {ApiError} from '@/lib/api'
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
      return '请输入邮箱地址'
    }
    if (!EMAIL_REGEX.test(normalizedEmail)) {
      return '请输入有效邮箱地址'
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
      setPasswordError('请输入密码')
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
      setPasswordError(nextPasswordError)
      hasError = true
    } else {
      setPasswordError(null)
    }

    if (!confirmPassword) {
      setConfirmPasswordError('请再次输入密码')
      hasError = true
    } else if (confirmPassword !== registerPassword) {
      setConfirmPasswordError('两次输入的密码不一致')
      hasError = true
    } else {
      setConfirmPasswordError(null)
    }

    if (verificationRequired && verificationCode.trim().length === 0) {
      setVerificationCodeError('请输入验证码')
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
        toast.success('验证码已发送，请查收邮箱')
      } else {
        toast.info('当前环境无需验证码校验')
      }
    } catch (error) {
      if (error instanceof ApiError) {
        setSubmitError(error.msg || '验证码发送失败，请稍后重试')
      } else if (error instanceof Error) {
        setSubmitError(error.message)
      } else {
        setSubmitError('验证码发送失败，请稍后重试')
      }
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
      if (error instanceof ApiError) {
        setSubmitError(error.msg || '邮箱或密码错误')
      } else if (error instanceof Error) {
        setSubmitError(error.message)
      } else {
        setSubmitError('登录失败，请稍后重试')
      }
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

      toast.success('注册成功，请登录继续')
      setLoginPassword('')
      setRegisterPassword('')
      setConfirmPassword('')
      setVerificationCode('')
      setVerificationRequired(false)
      setVerificationCooldown(0)
      switchMode('login')
    } catch (error) {
      if (error instanceof ApiError) {
        if (error.msg.includes('该邮箱已注册')) {
          setEmailError(error.msg)
        } else if (error.msg.includes('两次输入的密码不一致')) {
          setConfirmPasswordError('两次输入的密码不一致')
        } else if (error.msg.includes('请输入邮箱验证码')) {
          setVerificationRequired(true)
          setVerificationCodeError('请输入验证码')
        } else if (
          error.msg.includes('验证码错误') ||
          error.msg.includes('已过期')
        ) {
          setVerificationRequired(true)
          setVerificationCodeError('验证码错误或已过期，请重新获取')
        } else {
          setSubmitError(error.msg || '注册失败，请稍后重试')
        }
      } else if (error instanceof Error) {
        setSubmitError(error.message)
      } else {
        setSubmitError('注册失败，请稍后重试')
      }
    } finally {
      setSubmitting(false)
    }
  }

  const title = mode === 'login' ? '登录后继续' : '创建账号'
  const description =
    mode === 'login'
      ? '登录后可创建会话并保存你的任务历史'
      : '注册后可保存会话、文件与任务历史'

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[420px] p-5 sm:p-6" showCloseButton>
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription>{description}</DialogDescription>
        </DialogHeader>

        {mode === 'login' ? (
          <form className="space-y-4" onSubmit={handleLoginSubmit}>
            <div className="space-y-2">
              <Label htmlFor="login-email">邮箱</Label>
              <Input
                id="login-email"
                type="email"
                autoComplete="email"
                placeholder="you@example.com"
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
              <Label htmlFor="login-password">密码</Label>
              <Input
                id="login-password"
                type="password"
                autoComplete="current-password"
                placeholder="请输入密码"
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

            {submitError && <p className="text-sm text-destructive">{submitError}</p>}

            <Button type="submit" className="w-full" disabled={!canLoginSubmit}>
              {submitting ? '登录中...' : '登录'}
            </Button>

            <p className="text-sm text-muted-foreground text-center">
              没有账号？
              <button
                type="button"
                className="ml-1 text-foreground underline-offset-4 hover:underline"
                onClick={() => switchMode('register')}
              >
                去注册
              </button>
            </p>
          </form>
        ) : (
          <form className="space-y-4" onSubmit={handleRegisterSubmit}>
            <div className="space-y-2">
              <Label htmlFor="register-email">邮箱</Label>
              <Input
                id="register-email"
                type="email"
                autoComplete="email"
                placeholder="you@example.com"
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
                className="w-full"
                onClick={() => {
                  void handleSendCode()
                }}
                disabled={submitting || sendingCode || verificationCooldown > 0}
              >
                {sendingCode
                  ? '发送中...'
                  : verificationCooldown > 0
                    ? `${verificationCooldown}s 后可重发`
                    : '发送验证码'}
              </Button>
            </div>

            {verificationRequired && (
              <div className="space-y-2">
                <Label htmlFor="register-verification-code">验证码</Label>
                <Input
                  id="register-verification-code"
                  type="text"
                  inputMode="numeric"
                  autoComplete="one-time-code"
                  placeholder="请输入邮箱验证码"
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
              <Label htmlFor="register-password">密码</Label>
              <Input
                id="register-password"
                type="password"
                autoComplete="new-password"
                placeholder="请输入密码"
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
                <p className="text-xs text-muted-foreground">
                  至少 8 位，包含字母和数字，可使用 !@#$%^&*._- 符号。
                </p>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="register-confirm-password">确认密码</Label>
              <Input
                id="register-confirm-password"
                type="password"
                autoComplete="new-password"
                placeholder="请再次输入密码"
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

            {submitError && <p className="text-sm text-destructive">{submitError}</p>}

            <Button type="submit" className="w-full" disabled={!canRegisterSubmit}>
              {submitting ? '注册中...' : '注册'}
            </Button>

            <p className="text-sm text-muted-foreground text-center">
              已有账号？
              <button
                type="button"
                className="ml-1 text-foreground underline-offset-4 hover:underline"
                onClick={() => switchMode('login')}
              >
                去登录
              </button>
            </p>
          </form>
        )}
      </DialogContent>
    </Dialog>
  )
}
