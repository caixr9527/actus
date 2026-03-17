'use client'

import {useState} from 'react'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import {Button} from '@/components/ui/button'
import { useI18n } from '@/lib/i18n'

type DeleteSessionDialogProps = {
  open: boolean
  onOpenChange: (open: boolean) => void
  onConfirm: () => Promise<void>
}

/**
 * 删除任务确认弹窗
 * 确认后才发起 API 删除请求
 */
export function DeleteSessionDialog({open, onOpenChange, onConfirm}: DeleteSessionDialogProps) {
  const { t } = useI18n()
  const [deleting, setDeleting] = useState(false)

  const handleConfirm = async () => {
    setDeleting(true)
    try {
      await onConfirm()
    } finally {
      setDeleting(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[440px]">
        <DialogHeader>
          <DialogTitle className="text-lg font-semibold">
            {t('sessionDeleteDialog.title')}
          </DialogTitle>
          <DialogDescription className="text-sm text-muted-foreground leading-relaxed">
            {t('sessionDeleteDialog.description')}
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button
            variant="outline"
            className="cursor-pointer"
            onClick={() => onOpenChange(false)}
            disabled={deleting}
          >
            {t('common.cancel')}
          </Button>
          <Button
            className="cursor-pointer"
            onClick={handleConfirm}
            disabled={deleting}
          >
            {deleting ? t('sessionDeleteDialog.deleting') : t('common.confirm')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

