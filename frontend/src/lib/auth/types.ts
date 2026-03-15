export type AuthTokenType = "Bearer"

export type AuthTokenPair = {
  access_token: string
  refresh_token: string
  token_type: AuthTokenType
  access_token_expires_in: number
  refresh_token_expires_in: number
}

export type AuthUser = {
  user_id: string
  email: string
  nickname: string | null
  avatar_url: string | null
  timezone: string
  locale: string
  auth_provider: string
  status: string
  created_at: string
  updated_at: string
  last_login_at: string | null
  last_login_ip: string | null
}

export type LoginRequestPayload = {
  email: string
  password: string
}

export type LoginResponseData = {
  tokens: AuthTokenPair
  user: AuthUser
}

export type RegisterRequestPayload = {
  email: string
  password: string
  verification_code?: string
}

export type RegisterResponseData = {
  user_id: string
  email: string
  auth_provider: string
  status: string
  created_at: string
}

export type RefreshTokenResponseData = {
  tokens: AuthTokenPair
}

export type LogoutResponseData = {
  success: boolean
}
