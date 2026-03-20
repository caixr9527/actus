"use client"

import { type ReactNode, useEffect, useState } from "react"
import { initializeAuth } from "@/lib/auth"

export function AuthProvider({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(false)

  useEffect(() => {
    let unmounted = false

    void initializeAuth().finally(() => {
      if (!unmounted) {
        setReady(true)
      }
    })

    return () => {
      unmounted = true
    }
  }, [])

  if (!ready) {
    return null
  }

  return <>{children}</>
}
