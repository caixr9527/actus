import assert from "node:assert/strict"
import test from "node:test"

import { resolveAuthRouteGuardRedirect } from "../src/lib/auth/route-guard"

test("unauthenticated user should be redirected to login popup on protected route", () => {
  const redirect = resolveAuthRouteGuardRedirect({
    pathname: "/sessions/abc",
    search: "?from=share",
    isLoggedIn: false,
  })

  assert.equal(
    redirect,
    "/?auth=login&redirect=%2Fsessions%2Fabc%3Ffrom%3Dshare",
  )
})

test("unauthenticated user should stay on home and auth compatibility routes", () => {
  assert.equal(
    resolveAuthRouteGuardRedirect({
      pathname: "/",
      search: "?auth=login",
      isLoggedIn: false,
    }),
    null,
  )

  assert.equal(
    resolveAuthRouteGuardRedirect({
      pathname: "/auth/login",
      search: "",
      isLoggedIn: false,
    }),
    null,
  )
})

test("authenticated user should leave auth routes", () => {
  assert.equal(
    resolveAuthRouteGuardRedirect({
      pathname: "/auth/register",
      search: "",
      isLoggedIn: true,
    }),
    "/",
  )
})

test("authenticated user should respect redirect query on auth popup route", () => {
  assert.equal(
    resolveAuthRouteGuardRedirect({
      pathname: "/",
      search: "?auth=login&redirect=%2Fsessions%2Fabc",
      isLoggedIn: true,
    }),
    "/sessions/abc",
  )

  assert.equal(
    resolveAuthRouteGuardRedirect({
      pathname: "/",
      search: "?auth=register&redirect=https%3A%2F%2Fevil.com",
      isLoggedIn: true,
    }),
    "/",
  )
})
