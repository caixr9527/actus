/**
 * API 统一响应格式
 */
export type ApiResponse<T = unknown> = {
  code: number;
  msg?: string | null;
  data: T | null;
  error_key?: string | null;
  error_params?: Record<string, unknown> | null;
};

/**
 * 会话状态
 */
export type SessionStatus = "pending" | "running" | "waiting" | "completed" | "failed" | "cancelled";

/**
 * 执行状态
 */
export type ExecutionStatus = "pending" | "running" | "completed" | "failed" | "cancelled";

/**
 * 工具事件状态
 */
export type ToolEventStatus = "calling" | "called";

/**
 * MCP 传输类型
 */
export type MCPTransport = "stdio" | "sse" | "streamable_http";

// ==================== 配置模块类型 ====================

export type PublicModelConfig = {
  temperature?: number;
  max_tokens?: number;
  description?: string;
  badge?: string;
  [key: string]: unknown;
};

export type ListModelItem = {
  id: string;
  display_name: string;
  provider: string;
  enabled: boolean;
  sort_order: number;
  config?: PublicModelConfig;
};

export type ModelsData = {
  default_model_id: string;
  models: ListModelItem[];
};

/**
 * Agent 通用配置
 */
export type AgentConfig = {
  max_iterations?: number;
  max_retries?: number;
  max_search_results?: number;
  [key: string]: unknown;
};

/**
 * MCP 服务器列表项（GET 响应）
 */
export type ListMCPServerItem = {
  server_name: string;
  enabled: boolean;
  transport: MCPTransport;
  tools: string[];
};

/**
 * MCP 服务器列表响应
 */
export type MCPServersData = {
  mcp_servers: ListMCPServerItem[];
};

/**
 * MCP 服务器配置（POST 请求体中单个服务器的配置）
 */
export type MCPServerConfig = {
  transport?: MCPTransport;
  enabled?: boolean;
  description?: string | null;
  env?: Record<string, unknown> | null;
  command?: string | null;
  args?: string[] | null;
  url?: string | null;
  headers?: Record<string, unknown> | null;
  [key: string]: unknown;
};

/**
 * MCP 配置（POST 新增 MCP 服务的请求体）
 */
export type MCPConfig = {
  mcpServers: Record<string, MCPServerConfig>;
  [key: string]: unknown;
};

/**
 * A2A 服务器列表项（GET 响应）
 */
export type ListA2AServerItem = {
  id: string;
  name: string;
  description: string;
  input_modes: string[];
  output_modes: string[];
  streaming: boolean;
  push_notifications: boolean;
  enabled: boolean;
};

/**
 * A2A 服务器列表响应
 */
export type A2AServersData = {
  a2a_servers: ListA2AServerItem[];
};

/**
 * 新增 A2A 服务器请求参数
 */
export type CreateA2AServerParams = {
  base_url: string;
};

// ==================== 文件模块类型 ====================

/**
 * 文件信息
 */
export type FileInfo = {
  id: string;
  filename: string;
  filepath: string;
  key: string;
  extension: string;
  content_type: string;
  size: number;
  [key: string]: unknown;
};

/**
 * 文件上传请求参数
 */
export type FileUploadParams = {
  file: File;
  session_id?: string;
};

// ==================== 会话模块类型 ====================

/**
 * 会话信息
 */
export type Session = {
  session_id: string;
  title: string;
  latest_message: string;
  latest_message_at: string;
  status: SessionStatus;
  unread_message_count: number;
  current_model_id?: string | null;
  [key: string]: unknown;
};

/**
 * 会话列表响应
 */
export type SessionsData = {
  sessions: Session[];
};

/**
 * 创建会话请求参数
 */
export type CreateSessionParams = {
  title?: string;
  [key: string]: unknown;
};

export type ArtifactDeliveryState =
  | "candidate"
  | "selected"
  | "delivered"
  | "rejected"
  | "expired"
  | "quarantined";

export type ArtifactRevisionSourceKind =
  | "tool_write_file"
  | "tool_replace_file"
  | "browser_screenshot"
  | "browser_snapshot"
  | "page_snapshot"
  | "document_input"
  | "user_upload"
  | "final_answer_snapshot"
  | "derived_export"
  | "rag_chunk_index"
  | "manual_registration";

export type ArtifactType =
  | "file"
  | "screenshot"
  | "browser_snapshot"
  | "page_snapshot"
  | "dataset"
  | "report"
  | "log_excerpt"
  | "final_answer_snapshot"
  | "rag_chunk_index";

export type SelectedArtifactRevision = {
  artifact_id: string;
  revision_id: string;
  content_hash: string;
  path: string;
  artifact_type: ArtifactType;
  delivery_state: ArtifactDeliveryState;
  session_id: string;
  run_id?: string | null;
  source_run_id?: string | null;
  source_step_id?: string | null;
  source_event_id?: string | null;
  source_kind: ArtifactRevisionSourceKind;
  selected_reason: string;
  selected_at: string;
};

export type ArtifactRevisionEventRef = {
  artifact_id: string;
  revision_id: string;
  content_hash: string;
  path: string;
  artifact_type: ArtifactType;
  delivery_state: ArtifactDeliveryState;
  source_event_id?: string | null;
};

export type ArtifactEventPayload = {
  artifact_refs?: Array<{
    artifact_id: string;
    path: string;
    artifact_type: ArtifactType;
    delivery_state: ArtifactDeliveryState;
    current_revision_id?: string | null;
    latest_content_hash?: string | null;
  }>;
  revision_refs: ArtifactRevisionEventRef[];
  counts?: Record<string, number>;
  summary?: string;
  source_event_ids?: string[];
  runtime_metadata?: Record<string, unknown>;
};

export type ArtifactEvent = {
  event_id?: string | null;
  created_at?: number;
  runtime: RuntimeEventMeta;
  payload: ArtifactEventPayload;
  [key: string]: unknown;
};

export type ArtifactEventData = ArtifactEvent;

export type SafetyAuditDecision =
  | "allow"
  | "block"
  | "rewrite"
  | "require_confirmation"
  | "confirmation_approved"
  | "confirmation_rejected"
  | "correction"
  | "superseded";

export type SafetyAuditRiskLevel = "low" | "medium" | "high" | "critical";

export type SafetyAuditEventRef = {
  audit_id: string;
  decision: SafetyAuditDecision;
  risk_level: SafetyAuditRiskLevel;
  reason_code: string;
  step_id?: string | null;
  tool_call_id?: string | null;
  function_name: string;
};

export type SafetyAuditEventPayload = {
  audit_refs: SafetyAuditEventRef[];
  source_event_ids: string[];
  decision_counts: Partial<Record<SafetyAuditDecision, number>>;
  risk_counts: Partial<Record<SafetyAuditRiskLevel, number>>;
  blocked_count: number;
  rewrite_count: number;
  confirmation_count: number;
  summary: string;
  runtime_metadata: {
    visibility: "hidden";
    projection_key: string;
    schema_version: "safety_audit_event.v1";
  };
};

export type SafetyAuditEvent = {
  event_id?: string | null;
  created_at?: number;
  runtime: RuntimeEventMeta;
  payload: SafetyAuditEventPayload;
  [key: string]: unknown;
};

export type ArtifactRevisionFileParams = {
  session_id: string;
  artifact_id: string;
  revision_id: string;
  content_hash: string;
  run_id?: string | null;
  source_run_id?: string | null;
};

/**
 * 聊天消息
 */
export type ChatMessage = {
  event_id?: string | null;
  created_at?: number;
  runtime: RuntimeEventMeta;
  role: "user" | "assistant" | "system";
  message: string;
  stage?: "intermediate" | "final";
  attachments?: Array<{
    file_id: string;
    filename: string;
    [key: string]: unknown;
  }>;
  selected_artifact_revisions?: SelectedArtifactRevision[];
  [key: string]: unknown;
};

export type ChatCommand = {
  type: "continue_cancelled_task";
};

export type FeedbackTargetType =
  | "run"
  | "step"
  | "tool_call"
  | "message_event"
  | "wait_event"
  | "evidence"
  | "evidence_gap"
  | "sandbox_fact"
  | "safety_audit"
  | "artifact_revision"
  | "self_review"
  | "final_delivery"
  | "user_goal";

export type FeedbackReasonCode =
  | "user_confirmed"
  | "user_rejected"
  | "user_selected_option"
  | "user_provided_clarification"
  | "user_corrected_requirement"
  | "user_set_preference"
  | "user_cancelled"
  | "user_continued_cancelled"
  | "user_reported_satisfaction"
  | "user_reported_dissatisfaction";

export type MessageFeedbackIntentKind =
  | "correction"
  | "preference"
  | "clarification"
  | "satisfaction"
  | "dissatisfaction";

export type FeedbackInputEventIntentKind =
  | MessageFeedbackIntentKind
  | "confirmation"
  | "selection"
  | "cancel"
  | "continue_cancelled"
  | "takeover";

export type FeedbackTargetRef = {
  target_type: FeedbackTargetType;
  target_id: string;
  target_run_id?: string | null;
  target_revision_id?: string | null;
  target_content_hash?: string | null;
};

export type FeedbackIntent = {
  intent_kind: MessageFeedbackIntentKind;
  target_ref: FeedbackTargetRef;
  reason_code: FeedbackReasonCode;
  summary_hint?: string | null;
};

export type SubmitFeedbackParams = {
  source_action:
    | "final_satisfaction"
    | "artifact_satisfaction"
    | "explicit_correction"
    | "explicit_preference";
  intent_kind:
    | "satisfaction"
    | "dissatisfaction"
    | "correction"
    | "preference";
  target_ref: FeedbackTargetRef;
  reason_code: FeedbackReasonCode;
  summary_hint?: string | null;
  client_request_id?: string | null;
};

export type RuntimeAction = "send_message" | "resume" | "cancel" | "continue_cancelled";

export type RuntimeEventMeta = {
  session_id: string;
  run_id: string | null;
  status_after_event: SessionStatus | null;
  current_step_id: string | null;
  source_event_id: string | null;
  cursor_event_id: string | null;
  durability: "persistent" | "live_only";
  visibility: "timeline" | "draft" | "control" | "hidden";
};

export type LiveOnlyRuntimeEventMeta = RuntimeEventMeta & {
  durability: "live_only";
  visibility: "draft";
  source_event_id: null;
  cursor_event_id: null;
};

export type RuntimeCursor = {
  latest_event_id: string | null;
  has_more: boolean;
};

export type RuntimeCapabilities = {
  can_send_message: boolean;
  can_resume: boolean;
  can_cancel: boolean;
  can_continue_cancelled: boolean;
  disabled_reasons: Partial<Record<RuntimeAction, string>>;
};

export type RuntimeInteraction = {
  kind: "none" | "wait";
  interrupt_id: string | null;
  payload: Record<string, unknown>;
};

export type SandboxProfileProjection = {
  schema_version: "sandbox_capability_profile.v1";
  health_status: string;
  generated_at: string;
  expires_at: string | null;
  stale: boolean;
  unavailable_capabilities: string[];
  requires_confirmation: string[];
};

export type RuntimeObservation = {
  session_id: string;
  run_id: string | null;
  status: SessionStatus;
  current_step_id: string | null;
  cursor: RuntimeCursor;
  capabilities: RuntimeCapabilities;
  interaction: RuntimeInteraction;
  sandbox_profile: SandboxProfileProjection | null;
};

/**
 * 聊天请求参数
 * message 为空时用于流式拉取未完成任务的事件列表
 */
export type ChatParams = {
  message?: string;
  attachments?: string[];
  event_id?: string;
  feedback_intent?: FeedbackIntent;
  resume?: {
    value: unknown;
  };
  command?: ChatCommand;
  [key: string]: unknown;
};

/**
 * 会话详情（含事件列表，与 chat 流式响应格式一致）
 */
export type SessionDetail = Session & {
  runtime: RuntimeObservation;
  events: SSEEventData[];
};

export type UpdateSessionModelParams = {
  model_id: string;
};

export type UpdateSessionModelResponse = {
  session_id: string;
  current_model_id: string;
};

/**
 * 计划步骤
 */
export type PlanStep = {
  id: string;
  description: string;
  status: ExecutionStatus;
  outcome?: StepOutcome | null;
  [key: string]: unknown;
};

/**
 * 计划事件
 */
export type PlanEvent = {
  event_id?: string | null;
  created_at?: number;
  runtime: RuntimeEventMeta;
  steps: PlanStep[];
  [key: string]: unknown;
};

export type StepOutcome = {
  done: boolean;
  summary: string;
  produced_artifacts: string[];
  blockers: string[];
  facts_learned: string[];
  open_questions: string[];
  next_hint?: string | null;
  reused_from_run_id?: string | null;
  reused_from_step_id?: string | null;
  [key: string]: unknown;
};

/**
 * 步骤事件
 */
export type StepEvent = {
  event_id?: string | null;
  created_at?: number;
  runtime: RuntimeEventMeta;
  id: string;
  status: ExecutionStatus;
  description: string;
  outcome?: StepOutcome | null;
  [key: string]: unknown;
};

/**
 * 工具调用事件
 */
export type ToolEvent = {
  event_id?: string | null;
  created_at?: number;
  runtime: RuntimeEventMeta;
  name: string;
  function: string;
  args: Record<string, unknown>;
  content?: unknown;
  status?: ToolEventStatus;
  [key: string]: unknown;
};

export type SandboxFactEventRef = {
  fact_id: string;
  fact_kind: string;
  summary: string;
};

export type SandboxFactEvent = {
  event_id?: string | null;
  created_at?: number;
  runtime: RuntimeEventMeta;
  fact_refs: SandboxFactEventRef[];
  summary: string;
  source_event_id?: string | null;
  step_id?: string | null;
  [key: string]: unknown;
};

export type SearchResultItem = {
  url: string;
  title: string;
  snippet: string;
};

export type SearchToolContent = {
  results: SearchResultItem[];
};

export type BrowserToolContent = {
  screenshot?: string;
  page_type?: string;
  url?: string;
  title?: string;
  structured_page?: unknown;
  main_content?: unknown;
  cards?: unknown[];
  actionable_elements?: unknown[];
  matched_link_text?: string;
  matched_link_url?: string;
  matched_link_selector?: string;
  matched_link_index?: number | null;
  degrade_reason?: string;
};

export type FetchPageToolContent = {
  url: string;
  final_url?: string;
  status_code?: number;
  content_type?: string;
  title?: string;
  content?: string;
  excerpt?: string;
  content_length?: number;
  truncated?: boolean;
  max_chars?: number | null;
};

export type WaitUserTakeover = "none" | "browser";

export type WaitChoice = {
  label: string;
  resume_value: unknown;
  description?: string;
};

export type WaitInputTextPayload = {
  kind: "input_text";
  title?: string;
  prompt: string;
  details?: string;
  attachments?: string[];
  suggest_user_takeover?: WaitUserTakeover;
  placeholder?: string;
  submit_label?: string;
  response_key?: string;
  default_value?: string;
  multiline?: boolean;
  allow_empty?: boolean;
};

export type WaitConfirmPayload = {
  kind: "confirm";
  title?: string;
  prompt: string;
  details?: string;
  attachments?: string[];
  suggest_user_takeover?: WaitUserTakeover;
  confirm_label?: string;
  cancel_label?: string;
  confirm_resume_value?: unknown;
  cancel_resume_value?: unknown;
  emphasis?: "default" | "destructive";
};

export type WaitSelectPayload = {
  kind: "select";
  title?: string;
  prompt: string;
  details?: string;
  attachments?: string[];
  suggest_user_takeover?: WaitUserTakeover;
  options: WaitChoice[];
  default_resume_value?: unknown;
};

export type WaitPayload =
  | WaitInputTextPayload
  | WaitConfirmPayload
  | WaitSelectPayload;

/**
 * 等待事件数据
 */
export type WaitEventData = {
  event_id?: string | null;
  created_at?: number;
  runtime: RuntimeEventMeta;
  interrupt_id?: string | null;
  payload?: WaitPayload;
  [key: string]: unknown;
};

export type TextStreamChannel = "planner_message" | "final_message";

export type TextStreamStartEventData = {
  event_id?: string | null;
  created_at?: number;
  runtime: LiveOnlyRuntimeEventMeta;
  stream_id: string;
  channel: TextStreamChannel;
  run_id?: string | null;
  session_id?: string | null;
  stage: "planner" | "summary" | "final";
  is_replay?: boolean;
  [key: string]: unknown;
};

export type TextStreamDeltaEventData = {
  event_id?: string | null;
  created_at?: number;
  runtime: LiveOnlyRuntimeEventMeta;
  stream_id: string;
  channel: TextStreamChannel;
  text: string;
  sequence: number;
  [key: string]: unknown;
};

export type TextStreamEndEventData = {
  event_id?: string | null;
  created_at?: number;
  runtime: LiveOnlyRuntimeEventMeta;
  stream_id: string;
  channel: TextStreamChannel;
  full_text_length: number;
  reason: "completed" | "cancelled" | "error";
  [key: string]: unknown;
};

/**
 * SSE 事件类型
 */
export type SSEEventType =
  | "message"
  | "title"
  | "plan"
  | "step"
  | "tool"
  | "artifact"
  | "feedback_input"
  | "feedback"
  | "safety_audit"
  | "sandbox_fact"
  | "wait"
  | "done"
  | "error"
  | "text_stream_start"
  | "text_stream_delta"
  | "text_stream_end";

/**
 * SSE 事件数据
 */
export type SSEEventData =
  | { type: "message"; data: ChatMessage }
  | {
      type: "title";
      data: {
        event_id?: string | null;
        created_at?: number;
        runtime: RuntimeEventMeta;
        title: string;
      };
    }
  | { type: "plan"; data: PlanEvent }
  | { type: "step"; data: StepEvent }
  | { type: "tool"; data: ToolEvent }
  | { type: "artifact"; data: ArtifactEvent }
  | {
      type: "feedback_input";
      data: {
        event_id?: string | null;
        created_at?: number;
        runtime: RuntimeEventMeta;
        payload: {
          source_action: string;
          intent_kind: FeedbackInputEventIntentKind;
          target_ref: FeedbackTargetRef;
          reason_code: FeedbackReasonCode;
          sanitized_summary?: string | null;
          input_hash: string;
          runtime_metadata: Record<string, string | number | boolean | null>;
        };
      };
    }
  | {
      type: "feedback";
      data: {
        event_id?: string | null;
        created_at?: number;
        runtime: RuntimeEventMeta;
        payload: {
          feedback_refs: string[];
          counts: Record<string, number>;
          severity_counts: Record<string, number>;
          status_counts: Record<string, number>;
          kind_counts: Record<string, number>;
          summary?: string | null;
          source_event_ids: string[];
          runtime_metadata: Record<string, string | number | boolean | null>;
        };
      };
    }
  | { type: "safety_audit"; data: SafetyAuditEvent }
  | { type: "sandbox_fact"; data: SandboxFactEvent }
  | { type: "wait"; data: WaitEventData }
  | {
      type: "done";
      data: {
        event_id?: string | null;
        created_at?: number;
        runtime: RuntimeEventMeta;
        [key: string]: unknown;
      };
    }
  | { type: "text_stream_start"; data: TextStreamStartEventData }
  | { type: "text_stream_delta"; data: TextStreamDeltaEventData }
  | { type: "text_stream_end"; data: TextStreamEndEventData }
  | {
      type: "error";
      data: {
        event_id?: string | null;
        created_at?: number;
        runtime: RuntimeEventMeta;
        error: string;
        error_key?: string | null;
        error_params?: Record<string, unknown> | null;
      };
    };

/**
 * SSE 事件处理器
 */
export type SSEEventHandler = (event: SSEEventData) => void;

/**
 * 会话文件信息
 */
export type SessionFile = {
  id: string;
  filename: string;
  filepath: string;
  key: string;
  extension: string;
  content_type: string;
  size: number;
  [key: string]: unknown;
};

/**
 * 查看文件内容请求参数
 */
export type ViewFileParams = {
  filepath: string;
  [key: string]: unknown;
};

/**
 * 查看 Shell 输出请求参数
 */
export type ViewShellParams = {
  shell_session_id: string;
  [key: string]: unknown;
};
