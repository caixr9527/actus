import type { Translate } from './i18n'
import type { AssistantTurnItem } from './assistant-turns'

export function resolveAssistantTurnSummary(
  status: AssistantTurnItem['status'],
  t: Translate,
): string {
  if (status === 'running' || status === 'waiting' || status === 'idle') {
    return t('sessionDetail.turnSummary.working')
  }
  return t('sessionDetail.turnSummary.done')
}
