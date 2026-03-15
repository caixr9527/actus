import { redirect } from "next/navigation"

interface RegisterPageProps {
  searchParams: Promise<{
    redirect?: string | string[]
    email?: string | string[]
  }>
}

export default async function RegisterPage({ searchParams }: RegisterPageProps) {
  const params = await searchParams
  const nextParams = new URLSearchParams({ auth: "register" })

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
