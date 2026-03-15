export { authApi } from "./api"
export {
  normalizeAuthRedirectTarget,
} from "./redirect"
export {
  clearAuthenticatedSession,
  getAuthSnapshot,
  hydrateAuthStoreFromStorage,
  isAuthenticatedSnapshot,
  setAuthenticatedSession,
  setCurrentUser,
  subscribeAuthStore,
  __resetAuthStoreForTest,
} from "./store"
export { initializeAuth, logoutFromServer, refreshAccessToken } from "./session"
export type {
  AuthSnapshot,
} from "./store"
export type {
  AuthTokenPair,
  AuthUser,
  LoginRequestPayload,
  LoginResponseData,
  LogoutResponseData,
  RefreshTokenResponseData,
  RegisterRequestPayload,
  RegisterResponseData,
  SendRegisterCodeRequestPayload,
  SendRegisterCodeResponseData,
} from "./types"
