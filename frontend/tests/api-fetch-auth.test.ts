import assert from "node:assert/strict"
import test from "node:test"

import { ApiError, registerAuthHooks, request } from "../src/lib/api/fetch"

function jsonResponse(status: number, payload: unknown): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: {
      "Content-Type": "application/json",
    },
  })
}

test("request should refresh token and retry once when first response is 401", async () => {
  const originalFetch = globalThis.fetch
  const authorizationHeaders: Array<string | null> = []

  globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
    const authHeader = new Headers(init?.headers).get("Authorization")
    authorizationHeaders.push(authHeader)

    if (authorizationHeaders.length === 1) {
      return jsonResponse(401, {
        code: 401,
        msg: "未授权",
        data: null,
      })
    }

    return jsonResponse(200, {
      code: 200,
      msg: "success",
      data: {
        ok: true,
      },
    })
  }) as typeof fetch

  let refreshCount = 0
  let authFailureCount = 0

  registerAuthHooks({
    getAccessToken: () => "expired-token",
    refreshAccessToken: async () => {
      refreshCount += 1
      return "fresh-token"
    },
    onAuthFailure: () => {
      authFailureCount += 1
    },
  })

  try {
    const result = await request<{ ok: boolean }>("/sessions", {
      method: "GET",
    })

    assert.deepEqual(result, { ok: true })
    assert.equal(refreshCount, 1)
    assert.equal(authFailureCount, 0)
    assert.equal(authorizationHeaders[0], "Bearer expired-token")
    assert.equal(authorizationHeaders[1], "Bearer fresh-token")
  } finally {
    registerAuthHooks(null)
    globalThis.fetch = originalFetch
  }
})

test("request should invoke onAuthFailure and throw 401 when refresh fails", async () => {
  const originalFetch = globalThis.fetch

  globalThis.fetch = (async () => {
    return jsonResponse(401, {
      code: 401,
      msg: "未授权",
      data: null,
    })
  }) as typeof fetch

  let refreshCount = 0
  let authFailureCount = 0

  registerAuthHooks({
    getAccessToken: () => "expired-token",
    refreshAccessToken: async () => {
      refreshCount += 1
      return null
    },
    onAuthFailure: () => {
      authFailureCount += 1
    },
  })

  try {
    await assert.rejects(
      async () => {
        await request("/sessions", {
          method: "GET",
        })
      },
      (error: unknown) => {
        assert.ok(error instanceof ApiError)
        assert.equal(error.code, 401)
        return true
      },
    )

    assert.equal(refreshCount, 1)
    assert.equal(authFailureCount, 1)
  } finally {
    registerAuthHooks(null)
    globalThis.fetch = originalFetch
  }
})

test("request should invoke onAuthFailure when retry after refresh is still 401", async () => {
  const originalFetch = globalThis.fetch

  globalThis.fetch = (async () => {
    return jsonResponse(401, {
      code: 401,
      msg: "未授权",
      data: null,
    })
  }) as typeof fetch

  let refreshCount = 0
  let authFailureCount = 0

  registerAuthHooks({
    getAccessToken: () => "expired-token",
    refreshAccessToken: async () => {
      refreshCount += 1
      return "fresh-token"
    },
    onAuthFailure: () => {
      authFailureCount += 1
    },
  })

  try {
    await assert.rejects(
      async () => {
        await request("/sessions", {
          method: "GET",
        })
      },
      (error: unknown) => {
        assert.ok(error instanceof ApiError)
        assert.equal(error.code, 401)
        return true
      },
    )

    assert.equal(refreshCount, 1)
    assert.equal(authFailureCount, 1)
  } finally {
    registerAuthHooks(null)
    globalThis.fetch = originalFetch
  }
})

test("request should preserve backend error_key and error_params on ApiError", async () => {
  const originalFetch = globalThis.fetch

  globalThis.fetch = (async () => {
    return jsonResponse(400, {
      code: 400,
      msg: "邮箱或密码错误",
      error_key: "error.auth.invalid_credentials",
      error_params: {
        attempt: 3,
      },
      data: null,
    })
  }) as typeof fetch

  try {
    await assert.rejects(
      async () => {
        await request("/auth/login", {
          method: "POST",
          body: JSON.stringify({ email: "user@example.com", password: "wrong" }),
          skipAuth: true,
        })
      },
      (error: unknown) => {
        assert.ok(error instanceof ApiError)
        assert.equal(error.code, 400)
        assert.equal(error.errorKey, "error.auth.invalid_credentials")
        assert.deepEqual(error.errorParams, { attempt: 3 })
        return true
      },
    )
  } finally {
    globalThis.fetch = originalFetch
  }
})

test("request should fallback to generic message when backend msg is omitted", async () => {
  const originalFetch = globalThis.fetch

  globalThis.fetch = (async () => {
    return jsonResponse(400, {
      code: 400,
      data: null,
      error_key: "error.auth.invalid_credentials",
    })
  }) as typeof fetch

  try {
    await assert.rejects(
      async () => {
        await request("/auth/login", {
          method: "POST",
          body: JSON.stringify({ email: "user@example.com", password: "wrong" }),
          skipAuth: true,
        })
      },
      (error: unknown) => {
        assert.ok(error instanceof ApiError)
        assert.equal(error.code, 400)
        assert.equal(error.msg, "请求失败")
        assert.equal(error.errorKey, "error.auth.invalid_credentials")
        return true
      },
    )
  } finally {
    globalThis.fetch = originalFetch
  }
})
