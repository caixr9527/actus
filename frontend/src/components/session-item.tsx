'use client'

import {useCallback} from 'react'
import {Ban, CircuitBoard, Loader2, MoreHorizontal, Trash, XCircle} from 'lucide-react'
import {Button} from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {Item, ItemActions, ItemContent, ItemDescription, ItemMedia} from '@/components/ui/item'
import {Avatar, AvatarGroupCount} from '@/components/ui/avatar'
import {cn, formatRelativeDate} from '@/lib/utils'
import type {Session} from '@/lib/api'
import { useI18n } from '@/lib/i18n'

type SessionItemProps = {
  session: Session
  isActive: boolean
  onClick: (sessionId: string) => void
  onDelete: (session: Session) => void
}

/**
 * 单个会话列表项
 * 展示会话标题、描述、时间及操作菜单
 */
export function SessionItem({session, isActive, onClick, onDelete}: SessionItemProps) {
  const { locale, t } = useI18n()
  const handleClick = useCallback(() => {
    onClick(session.session_id)
  }, [onClick, session.session_id])

  const handleDelete = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    onDelete(session)
  }, [onDelete, session])

  const description = session.latest_message || t('sessionItem.noMessage')
  const dateLabel = formatRelativeDate(session.latest_message_at, locale)
  const isRunning = session.status === 'running' || session.status === 'waiting'
  const isFailed = session.status === 'failed'
  const isCancelled = session.status === 'cancelled'

  return (
    <Item
      className={cn('p-2 hover:bg-white cursor-pointer gap-2 items-start', isActive && 'bg-white')}
      onClick={handleClick}
    >
      {/* 左侧图标 */}
      <ItemMedia>
        <Avatar className="size-8">
          <AvatarGroupCount>
            {isRunning
              ? <Loader2 className="animate-spin"/>
              : isFailed
                ? <XCircle/>
              : isCancelled
                ? <Ban/>
              : <CircuitBoard/>
            }
          </AvatarGroupCount>
        </Avatar>
      </ItemMedia>
      {/* 中间内容 */}
      <ItemContent className="gap-0 min-w-0">
        <p className="text-sm font-medium truncate">
          {session.title || t('session.newTask')}
        </p>
        <p className="text-xs text-muted-foreground truncate">
          {description}
        </p>
      </ItemContent>
      {/* 右侧操作区 */}
      <ItemActions className="flex flex-col pt-0.5 gap-0 self-start">
        <ItemDescription className="text-xs whitespace-nowrap">{dateLabel}</ItemDescription>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              size="icon-xs"
              variant="ghost"
              className="cursor-pointer"
              onClick={(e) => e.stopPropagation()}
            >
              <MoreHorizontal/>
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="center" side="bottom">
            <DropdownMenuItem
              variant="destructive"
              className="cursor-pointer"
              onClick={handleDelete}
            >
              <Trash/>
              {t('session.delete')}
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </ItemActions>
    </Item>
  )
}
