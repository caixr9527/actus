import { post } from "@/lib/api/fetch"
import type {
  LoginRequestPayload,
  LoginResponseData,
  LogoutResponseData,
  RefreshTokenResponseData,
  RegisterRequestPayload,
  RegisterResponseData,
  SendRegisterCodeRequestPayload,
  SendRegisterCodeResponseData,
} from "./types"

export const authApi = {
  register: (payload: RegisterRequestPayload): Promise<RegisterResponseData> => {
    return post<RegisterResponseData>("/auth/register", payload, {
      skipAuth: true,
      skipAuthRefresh: true,
    })
  },

  sendRegisterCode: (
    payload: SendRegisterCodeRequestPayload,
  ): Promise<SendRegisterCodeResponseData> => {
    return post<SendRegisterCodeResponseData>("/auth/register/send-code", payload, {
      skipAuth: true,
      skipAuthRefresh: true,
    })
  },

  login: (payload: LoginRequestPayload): Promise<LoginResponseData> => {
    return post<LoginResponseData>("/auth/login", payload, {
      skipAuth: true,
      skipAuthRefresh: true,
    })
  },

  refresh: (refreshToken: string): Promise<RefreshTokenResponseData> => {
    return post<RefreshTokenResponseData>(
      "/auth/refresh",
      { refresh_token: refreshToken },
      {
        skipAuth: true,
        skipAuthRefresh: true,
      },
    )
  },

  logout: (refreshToken: string): Promise<LogoutResponseData> => {
    return post<LogoutResponseData>("/auth/logout", {
      refresh_token: refreshToken,
    })
  },
}
