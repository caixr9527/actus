export function normalizeUserMessageText(message: string | null | undefined): string {
  if (message == null) return ''
  return message.replace(/\r\n?/g, '\n')
}

export function getUserMessageTextClassName(): string {
  return 'whitespace-break-spaces break-words text-sm leading-relaxed [tab-size:2]'
}
