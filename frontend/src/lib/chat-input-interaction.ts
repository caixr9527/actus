export type ChatInputInteractionParams = {
  disabled: boolean
  isRunning: boolean
  sending: boolean
  inputValue: string
  modelsLoading: boolean
  modelUpdating: boolean
}

export type ChatInputInteractionState = {
  canEditInput: boolean
  canSend: boolean
  canSwitchModel: boolean
}

/**
 * 统一计算输入区可交互状态。
 * 规则：
 * - 运行中允许继续输入，但禁止发送与切换模型；
 * - sending 阶段禁用输入；
 * - disabled 仅在非运行态禁用输入。
 */
export function resolveChatInputInteractionState(
  params: ChatInputInteractionParams,
): ChatInputInteractionState {
  const {
    disabled,
    isRunning,
    sending,
    inputValue,
    modelsLoading,
    modelUpdating,
  } = params

  const hasInput = inputValue.trim().length > 0
  const canEditInput = !sending && (!disabled || isRunning)
  const canSend = !isRunning && !sending && !disabled && hasInput
  const canSwitchModel = !isRunning && !disabled && !modelsLoading && !modelUpdating

  return {
    canEditInput,
    canSend,
    canSwitchModel,
  }
}
