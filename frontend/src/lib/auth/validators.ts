const PASSWORD_ALLOWED_REGEX = /^[A-Za-z0-9!@#$%^&*._-]+$/

export function validatePasswordStrength(value: string): string | null {
  if (value.length < 8) {
    return "密码长度不能少于8位"
  }
  if (!/[A-Za-z]/.test(value)) {
    return "密码必须包含字母"
  }
  if (!/\d/.test(value)) {
    return "密码必须包含数字"
  }
  if (!PASSWORD_ALLOWED_REGEX.test(value)) {
    return "密码仅允许字母、数字和常见符号 !@#$%^&*._-"
  }
  return null
}
