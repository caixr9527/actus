import React from "react"
import type { Metadata } from "next"
import { AuthProvider } from "@/providers/auth-provider"
import { AppShell } from "@/providers/app-shell"
import { DEFAULT_APP_LOCALE } from "@/lib/i18n/constants"
import { I18nProvider } from "@/lib/i18n/provider"
import { Toaster } from "@/components/ui/sonner"
import "./globals.css"

export const metadata: Metadata = {
  title: "Actus",
  description:
    "Actus 是一个行动引擎，它超越了答案的范畴，可以执行任务、自动化工作流程，并扩展您的能力。",
  icons: {
    icon: "/logo.svg",
    shortcut: "/logo.svg",
  },
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang={DEFAULT_APP_LOCALE} suppressHydrationWarning>
      <body className="h-screen overflow-hidden">
        <I18nProvider>
          <AuthProvider>
            <AppShell>{children}</AppShell>
          </AuthProvider>
        </I18nProvider>
        <Toaster position="top-center" richColors />
      </body>
    </html>
  )
}
