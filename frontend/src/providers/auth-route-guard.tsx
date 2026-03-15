"use client"

import { useEffect } from "react"
import { usePathname, useRouter, useSearchParams } from "next/navigation"
import { useAuth } from "@/hooks/use-auth"
import { resolveAuthRouteGuardRedirect } from "@/lib/auth/route-guard"

export function AuthRouteGuard() {
  const router = useRouter()
  const pathname = usePathname()
  const searchParams = useSearchParams()
  const { isHydrated, isLoggedIn } = useAuth()

  const search = searchParams.toString()

  useEffect(() => {
    if (!isHydrated) {
      return
    }

    const redirectTo = resolveAuthRouteGuardRedirect({
      pathname,
      search: search ? `?${search}` : "",
      isLoggedIn,
    })

    if (redirectTo && redirectTo !== `${pathname}${search ? `?${search}` : ""}`) {
      router.replace(redirectTo)
    }
  }, [isHydrated, isLoggedIn, pathname, router, search])

  return null
}
