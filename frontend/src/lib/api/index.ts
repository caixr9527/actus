/**
 * API 模块统一导出
 */

// 核心 fetch 封装
export {
  request,
  requestRaw,
  get,
  post,
  put,
  del,
  registerAuthHooks,
  getApiBaseUrl,
  createSSEConnection,
  createSSEStream,
  parseSSEStream,
  ApiError,
} from "./fetch";
export type { RequestOptions, AuthHooks } from "./fetch";

// 类型定义
export type {
  ApiResponse,
  SessionStatus,
  ExecutionStatus,
  ToolEventStatus,
  MCPTransport,
  AgentConfig,
  ListMCPServerItem,
  MCPServerConfig,
  MCPConfig,
  MCPServersData,
  PublicModelConfig,
  ListModelItem,
  ModelsData,
  ListA2AServerItem,
  A2AServersData,
  CreateA2AServerParams,
  FileInfo,
  FileUploadParams,
  Session,
  SessionDetail,
  SessionsData,
  CreateSessionParams,
  ChatMessage,
  ChatParams,
  UpdateSessionModelParams,
  UpdateSessionModelResponse,
  PlanStep,
  PlanEvent,
  StepEvent,
  ToolEvent,
  SSEEventType,
  SSEEventData,
  SSEEventHandler,
  SessionFile,
  ViewFileParams,
  ViewShellParams,
} from "./types";

// 模块 API
export { configApi } from "./config";
export {
  getApiErrorMessage,
  getApiErrorMessageFromPayload,
  getApiErrorMessageKey,
  isApiErrorKey,
} from "./error-i18n";
export { fileApi } from "./file";
export { sessionApi } from "./session";
