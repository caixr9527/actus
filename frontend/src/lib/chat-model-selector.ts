import type { ListModelItem } from './api/types'

export type ModelSelectorState = {
  selectedModelId: string
  selectedModel: ListModelItem | null
  defaultModel: ListModelItem | null
}

type PerformModelSelectionParams = {
  nextModelId: string
  selectedModelId: string
  modelsLoading?: boolean
  modelUpdating?: boolean
  onModelChange?: (modelId: string) => Promise<void>
  closeMenu: () => void
}

export function resolveModelSelectorState(
  modelOptions: ListModelItem[],
  currentModelId?: string | null,
  defaultModelId?: string | null,
): ModelSelectorState {
  const selectedModelId = currentModelId ?? 'auto'

  return {
    selectedModelId,
    selectedModel: modelOptions.find((model) => model.id === selectedModelId) ?? null,
    defaultModel: modelOptions.find((model) => model.id === defaultModelId) ?? null,
  }
}

export async function performModelSelection({
  nextModelId,
  selectedModelId,
  modelsLoading = false,
  modelUpdating = false,
  onModelChange,
  closeMenu,
}: PerformModelSelectionParams): Promise<boolean> {
  if (!onModelChange || modelsLoading || modelUpdating || selectedModelId === nextModelId) {
    closeMenu()
    return false
  }

  await onModelChange(nextModelId)
  closeMenu()
  return true
}
