'use client'

import { useMemo } from 'react'
import {cn} from '@/lib/utils'
import {Button} from '@/components/ui/button'
import {getSuggestedQuestions} from '@/config/app.config'
import { useI18n } from '@/lib/i18n'

interface SuggestedQuestionsProps {
  className?: string
  onQuestionClick?: (question: string) => void
}

export function SuggestedQuestions({className, onQuestionClick}: SuggestedQuestionsProps) {
  const { locale } = useI18n()
  const suggestedQuestions = useMemo(() => getSuggestedQuestions(locale), [locale])

  const handleClick = (question: string) => {
    onQuestionClick?.(question)
  }

  return (
    <div className={cn('flex flex-wrap gap-2 sm:gap-3', className)}>
      {suggestedQuestions.map((question, index) => (
        <Button
          key={index}
          variant="outline"
          className="cursor-pointer text-xs sm:text-sm whitespace-normal break-words"
          onClick={() => handleClick(question)}
        >
          {question}
        </Button>
      ))}
    </div>
  )
}
