import { redirect } from "next/navigation"

interface LoginPageProps {
  searchParams: Promise<{
    redirect?: string | string[]
    email?: string | string[]
  }>
}

export default async function LoginPage({ searchParams }: LoginPageProps) {
  const params = await searchParams
  const nextParams = new URLSearchParams({ auth: "login" })

  const redirectParam = params.redirect
  if (typeof redirectParam === "string" && redirectParam.trim().length > 0) {
    nextParams.set("redirect", redirectParam)
  }

  const emailParam = params.email
  if (typeof emailParam === "string" && emailParam.trim().length > 0) {
    nextParams.set("email", emailParam)
  }

  redirect(`/?${nextParams.toString()}`)
}
