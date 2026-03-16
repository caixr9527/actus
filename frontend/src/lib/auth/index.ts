export { authApi } from "./api"
export { getAvatarFileId, toAvatarFileRef } from "./avatar"
export { getUserDisplayName, maskEmail } from "./display"
export {
  ensureOptionExists,
  filterSelectOptions,
  getLocaleOptions,
  getTimeZoneOptions,
} from "./options"
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
  GetCurrentUserResponseData,
  RefreshTokenResponseData,
  RegisterRequestPayload,
  RegisterResponseData,
  SendRegisterCodeRequestPayload,
  SendRegisterCodeResponseData,
  UpdateCurrentUserPayload,
  UpdateCurrentUserResponseData,
  UpdatePasswordPayload,
  UpdatePasswordResponseData,
} from "./types"
