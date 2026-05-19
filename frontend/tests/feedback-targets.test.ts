import assert from 'node:assert/strict'
import test from 'node:test'

import {
  buildArtifactRevisionFeedbackTarget,
  buildFinalMessageFeedbackTarget,
} from '../src/lib/feedback-targets'
import { eventsToTimeline } from '../src/lib/session-events'
import type { ChatMessage, RuntimeEventMeta, SelectedArtifactRevision, SSEEventData } from '../src/lib/api/types'

function finalMessage(overrides: Partial<ChatMessage> = {}): ChatMessage {
  return {
    event_id: 'evt-final-1',
    role: 'assistant',
    message: '最终答案',
    stage: 'final',
    runtime: {
      session_id: 'session-1',
      run_id: 'run-1',
      status_after_event: 'completed',
      current_step_id: null,
      source_event_id: 'evt-final-1',
      cursor_event_id: 'evt-final-1',
      durability: 'persistent',
      visibility: 'timeline',
    },
    ...overrides,
  }
}

function artifactRevision(overrides: Partial<SelectedArtifactRevision> = {}): SelectedArtifactRevision {
  return {
    artifact_id: 'artifact-1',
    revision_id: 'rev-1',
    content_hash: 'sha256:abc',
    path: '/workspace/report.md',
    artifact_type: 'file',
    delivery_state: 'selected',
    session_id: 'session-1',
    run_id: 'run-1',
    source_kind: 'tool_write_file',
    selected_reason: 'final_delivery',
    selected_at: '2026-05-19T00:00:00Z',
    ...overrides,
  }
}

function runtime(overrides: Partial<RuntimeEventMeta> = {}): RuntimeEventMeta {
  return {
    session_id: 'session-1',
    run_id: 'run-1',
    status_after_event: 'completed',
    current_step_id: null,
    source_event_id: 'evt-final-1',
    cursor_event_id: 'evt-final-1',
    durability: 'persistent',
    visibility: 'timeline',
    ...overrides,
  }
}

test('independent final message feedback should build message event target', () => {
  assert.deepEqual(buildFinalMessageFeedbackTarget(finalMessage()), {
    target_type: 'message_event',
    target_id: 'evt-final-1',
    target_run_id: 'run-1',
  })
})

test('independent final message feedback should be unavailable when target is incomplete', () => {
  assert.equal(buildFinalMessageFeedbackTarget(finalMessage({ event_id: null })), null)
  assert.equal(buildFinalMessageFeedbackTarget(finalMessage({
    runtime: {
      ...finalMessage().runtime,
      run_id: null,
    },
  })), null)
})

test('artifact feedback should require revision-aware target fields', () => {
  assert.deepEqual(buildArtifactRevisionFeedbackTarget(artifactRevision()), {
    target_type: 'artifact_revision',
    target_id: 'artifact-1',
    target_revision_id: 'rev-1',
    target_content_hash: 'sha256:abc',
    target_run_id: 'run-1',
  })
  assert.equal(buildArtifactRevisionFeedbackTarget(artifactRevision({ revision_id: '' })), null)
  assert.equal(buildArtifactRevisionFeedbackTarget(artifactRevision({ content_hash: '' })), null)
  assert.equal(buildArtifactRevisionFeedbackTarget(artifactRevision({ run_id: null })), null)
})

test('final artifact attachment should carry revision identity for preview feedback', () => {
  const revision = artifactRevision({ path: '/workspace/report.md' })
  const timeline = eventsToTimeline([
    {
      type: 'message',
      data: {
        runtime: runtime(),
        event_id: 'evt-final-1',
        role: 'assistant',
        stage: 'final',
        message: '最终答案',
        attachments: [
          {
            file_id: 'file-1',
            filename: 'report.md',
            filepath: '/workspace/report.md',
            size: 100,
          },
        ],
        selected_artifact_revisions: [revision],
      },
    } as SSEEventData,
  ])

  const attachmentItem = timeline.find((item) => item.kind === 'attachments')

  assert.equal(attachmentItem?.kind, 'attachments')
  if (attachmentItem?.kind === 'attachments') {
    assert.deepEqual(attachmentItem.files[0]?.artifactRevision, revision)
  }
})
