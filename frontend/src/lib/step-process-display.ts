export type StepProcessSection = 'tools' | 'summary'

export function resolveStepProcessSectionOrder(params: {
  hasTools: boolean
  hasSummary: boolean
}): StepProcessSection[] {
  const sections: StepProcessSection[] = []
  if (params.hasTools) sections.push('tools')
  if (params.hasSummary) sections.push('summary')
  return sections
}
