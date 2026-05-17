import type { SafetyAuditEvent, SafetyAuditEventRef, SSEEventData } from './api/types'

export type SafetyAuditSummaryItem = {
  auditId: string
  decision: SafetyAuditEventRef['decision']
  riskLevel: SafetyAuditEventRef['risk_level']
  reasonCode: string
  functionName: string
  stepId: string | null
  toolCallId: string | null
}

function shouldShowInSafetyDetails(ref: SafetyAuditEventRef): boolean {
  return (
    ref.decision === 'block' ||
    ref.decision === 'rewrite' ||
    ref.decision === 'require_confirmation' ||
    ref.risk_level === 'high' ||
    ref.risk_level === 'critical'
  )
}

export function collectSafetyAuditSummaryItems(events: SSEEventData[]): SafetyAuditSummaryItem[] {
  const items: SafetyAuditSummaryItem[] = []
  for (const event of events) {
    if (event.type !== 'safety_audit') continue
    const safetyAudit = event.data as SafetyAuditEvent
    const refs = Array.isArray(safetyAudit.payload.audit_refs)
      ? safetyAudit.payload.audit_refs
      : []
    for (const ref of refs) {
      if (!shouldShowInSafetyDetails(ref)) continue
      items.push({
        auditId: ref.audit_id,
        decision: ref.decision,
        riskLevel: ref.risk_level,
        reasonCode: ref.reason_code,
        functionName: ref.function_name,
        stepId: ref.step_id ?? null,
        toolCallId: ref.tool_call_id ?? null,
      })
    }
  }
  return items
}
