const PASSWORD_ALLOWED_REGEX = /^[A-Za-z0-9!@#$%^&*._-]+$/

export type PasswordValidationErrorKey =
  | "validation.passwordMinLength"
  | "validation.passwordRequireLetter"
  | "validation.passwordRequireNumber"
  | "validation.passwordAllowedChars"

export function validatePasswordStrength(value: string): PasswordValidationErrorKey | null {
  if (value.length < 8) {
    return "validation.passwordMinLength"
  }
  if (!/[A-Za-z]/.test(value)) {
    return "validation.passwordRequireLetter"
  }
  if (!/\d/.test(value)) {
    return "validation.passwordRequireNumber"
  }
  if (!PASSWORD_ALLOWED_REGEX.test(value)) {
    return "validation.passwordAllowedChars"
  }
  return null
}
