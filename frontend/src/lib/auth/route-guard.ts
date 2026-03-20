import { normalizeAuthRedirectTarget } from "./redirect"

type ResolveAuthRouteGuardRedirectParams = {
  pathname: string
  search: string
  isLoggedIn: boolean
}

function parseSearchParams(search: string): URLSearchParams {
  if (!search) {
    return new URLSearchParams()
  }

  const normalized = search.startsWith("?") ? search.slice(1) : search
  return new URLSearchParams(normalized)
}

function buildPathWithSearch(pathname: string, searchParams: URLSearchParams): string {
  const query = searchParams.toString()
  if (!query) {
    return pathname
  }
  return `${pathname}?${query}`
}

function buildLoginRedirectHref(target: string): string {
  const query = new URLSearchParams({
    auth: "login",
    redirect: target,
  })
  return `/?${query.toString()}`
}

function hasAuthFlowQuery(searchParams: URLSearchParams): boolean {
  const auth = searchParams.get("auth")
  return auth === "login" || auth === "register"
}

export function resolveAuthRouteGuardRedirect({
  pathname,
  search,
  isLoggedIn,
}: ResolveAuthRouteGuardRedirectParams): string | null {
  const searchParams = parseSearchParams(search)
  const currentPath = buildPathWithSearch(pathname, searchParams)

  if (isLoggedIn) {
    if (pathname.startsWith("/auth/")) {
      return "/"
    }

    if (pathname === "/" && hasAuthFlowQuery(searchParams)) {
      const redirectTarget = normalizeAuthRedirectTarget(searchParams.get("redirect"))
      if (redirectTarget && redirectTarget !== "/") {
        return redirectTarget
      }
      return "/"
    }

    return null
  }

  if (pathname === "/" || pathname.startsWith("/auth/")) {
    return null
  }

  return buildLoginRedirectHref(currentPath)
}
