"use client"

import { type CSSProperties, type ReactNode } from "react"
import { SidebarProvider } from "@/components/ui/sidebar"
import { LeftPanel } from "@/components/left-panel"
import { useAuth } from "@/hooks/use-auth"
import { SessionsProvider } from "@/providers/sessions-provider"

const sidebarLayoutStyle = {
  "--sidebar-width": "300px",
  "--sidebar-width-icon": "300px",
} as CSSProperties

export function AppShell({ children }: { children: ReactNode }) {
  const { isLoggedIn } = useAuth()

  const content = (
    <div className="flex-1 bg-[#f8f8f7] h-screen overflow-hidden">
      {children}
    </div>
  )

  if (!isLoggedIn) {
    return <SidebarProvider style={sidebarLayoutStyle}>{content}</SidebarProvider>
  }

  return (
    <SessionsProvider>
      <SidebarProvider style={sidebarLayoutStyle}>
        <LeftPanel />
        {content}
      </SidebarProvider>
    </SessionsProvider>
  )
}
