import type { ChatMessage, FeedbackTargetRef, SelectedArtifactRevision } from './api/types'

export function buildFinalMessageFeedbackTarget(message: ChatMessage): FeedbackTargetRef | null {
  const eventId = message.event_id ?? null
  const runId = message.runtime.run_id ?? null
  if (!eventId || !runId) return null
  return {
    target_type: 'message_event',
    target_id: eventId,
    target_run_id: runId,
  }
}

export function buildArtifactRevisionFeedbackTarget(revision: SelectedArtifactRevision): FeedbackTargetRef | null {
  if (!revision.artifact_id || !revision.revision_id || !revision.content_hash || !revision.run_id) {
    return null
  }
  return {
    target_type: 'artifact_revision',
    target_id: revision.artifact_id,
    target_revision_id: revision.revision_id,
    target_content_hash: revision.content_hash,
    target_run_id: revision.run_id,
  }
}
