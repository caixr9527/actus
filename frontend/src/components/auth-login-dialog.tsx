'use client'

import {type FormEvent, useEffect, useMemo, useState} from 'react'
import Link from 'next/link'
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
import {ApiError} from '@/lib/api'

const EMAIL_REGEX = /^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$/

type AuthLoginDialogProps = {
  open: boolean
  onOpenChange: (open: boolean) => void
  onSuccess?: () => void
}

export function AuthLoginDialog({
  open,
  onOpenChange,
  onSuccess,
}: AuthLoginDialogProps) {
  const {login} = useAuth()

  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [emailError, setEmailError] = useState<string | null>(null)
  const [passwordError, setPasswordError] = useState<string | null>(null)
  const [submitError, setSubmitError] = useState<string | null>(null)

  useEffect(() => {
    if (!open) {
      setPassword('')
      setPasswordError(null)
      setSubmitError(null)
    }
  }, [open])

  const canSubmit = useMemo(() => {
    return email.trim().length > 0 && password.length > 0 && !submitting
  }, [email, password, submitting])

  const validate = (): boolean => {
    let hasError = false

    const normalizedEmail = email.trim()
    if (!normalizedEmail) {
      setEmailError('请输入邮箱地址')
      hasError = true
    } else if (!EMAIL_REGEX.test(normalizedEmail)) {
      setEmailError('请输入有效邮箱地址')
      hasError = true
    } else {
      setEmailError(null)
    }

    if (!password) {
      setPasswordError('请输入密码')
      hasError = true
    } else {
      setPasswordError(null)
    }

    return !hasError
  }

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setSubmitError(null)

    if (!validate()) {
      return
    }

    setSubmitting(true)
    try {
      await login({
        email: email.trim(),
        password,
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
      setPassword('')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[420px] p-5 sm:p-6" showCloseButton>
        <DialogHeader>
          <DialogTitle>登录后继续</DialogTitle>
          <DialogDescription>
            登录后可创建会话并保存你的任务历史
          </DialogDescription>
        </DialogHeader>

        <form className="space-y-4" onSubmit={handleSubmit}>
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
            {emailError && (
              <p className="text-sm text-destructive">{emailError}</p>
            )}
          </div>

          <div className="space-y-2">
            <Label htmlFor="login-password">密码</Label>
            <Input
              id="login-password"
              type="password"
              autoComplete="current-password"
              placeholder="请输入密码"
              value={password}
              onChange={(event) => {
                setPassword(event.target.value)
                if (passwordError) {
                  setPasswordError(null)
                }
              }}
              aria-invalid={passwordError ? 'true' : 'false'}
              disabled={submitting}
            />
            {passwordError && (
              <p className="text-sm text-destructive">{passwordError}</p>
            )}
          </div>

          {submitError && (
            <p className="text-sm text-destructive">{submitError}</p>
          )}

          <Button type="submit" className="w-full" disabled={!canSubmit}>
            {submitting ? '登录中...' : '登录'}
          </Button>

          <p className="text-sm text-muted-foreground text-center">
            没有账号？
            <Link href="/auth/register" className="ml-1 text-foreground underline-offset-4 hover:underline">
              去注册
            </Link>
          </p>
        </form>
      </DialogContent>
    </Dialog>
  )
}
