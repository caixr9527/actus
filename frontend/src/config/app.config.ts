/**
 * 应用配置文件
 * 用于存储应用的静态配置数据
 */
import type { AppLocale } from '@/lib/i18n'

/**
 * 推荐问题/任务配置
 */
const suggestedQuestionsByLocale: Record<AppLocale, readonly string[]> = {
  'zh-CN': [
    '与最高的建筑相比，埃菲尔铁塔有多高？',
    'GitHub 上最热门的存储库有哪些？',
    '如何看待中国的外卖大战？',
    '超加工食品与健康有关吗？超加工食品的历史是怎样的？',
  ],
  'en-US': [
    'How tall is the Eiffel Tower compared with the tallest building?',
    'What are the most popular repositories on GitHub right now?',
    'How should we understand the food-delivery competition in China?',
    'Are ultra-processed foods linked to health, and what is their history?',
  ],
}

export function getSuggestedQuestions(locale: AppLocale): readonly string[] {
  return suggestedQuestionsByLocale[locale] ?? suggestedQuestionsByLocale['zh-CN']
}
