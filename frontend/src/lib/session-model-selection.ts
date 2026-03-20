import type { UpdateSessionModelParams } from "./api/types"

export function buildInitialSessionModelParams(
  selectedModelId: string | null | undefined,
): UpdateSessionModelParams | null {
  if (!selectedModelId || selectedModelId === "auto") {
    return null
  }

  return {
    model_id: selectedModelId,
  }
}
