'use client'

import { useEffect, useRef } from 'react'
import RFB from '@novnc/novnc/lib/rfb'
import { useI18n } from '@/lib/i18n'

export type VNCStatus = 'connecting' | 'connected' | 'disconnected' | 'error'

interface VNCViewerProps {
  url: string
  viewOnly?: boolean
  onStatusChange?: (status: VNCStatus, detail?: string) => void
}

export function VNCViewer({ url, viewOnly, onStatusChange }: VNCViewerProps) {
  const { t } = useI18n()
  const displayRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!displayRef.current) return

    onStatusChange?.('connecting')

    let rfb: RFB | null = null
    try {
      rfb = new RFB(displayRef.current, url, {
        credentials: { password: '', username: '', target: '' },
      })

      rfb.viewOnly = viewOnly || false
      rfb.scaleViewport = true
      rfb.background = '#000'

      rfb.addEventListener('connect', () => onStatusChange?.('connected'))
      rfb.addEventListener('disconnect', (e: CustomEvent) => {
        if (e.detail?.clean) {
          onStatusChange?.('disconnected', t('vncViewer.disconnected'))
        } else {
          onStatusChange?.('error', t('vncViewer.abnormalDisconnect'))
        }
      })
      rfb.addEventListener('securityfailure', () => {
        onStatusChange?.('error', t('vncViewer.authFailed'))
      })
    } catch {
      onStatusChange?.('error', t('vncViewer.connectFailed'))
    }

    return () => {
      try { rfb?.disconnect() } catch { /* noop */ }
    }
  }, [onStatusChange, t, url, viewOnly])

  return (
    <div
      ref={displayRef}
      style={{ width: '100%', height: '100%', background: '#000' }}
    />
  )
}
