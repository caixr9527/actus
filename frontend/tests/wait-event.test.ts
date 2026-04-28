import assert from 'node:assert/strict'
import test from 'node:test'

import { buildResumeValueFromWaitPayload, normalizeWaitPayload, parseWaitEventContext } from '../src/lib/wait-event'

const runtime = {
  session_id: 'session-1',
  run_id: 'run-1',
  status_after_event: 'waiting',
  current_step_id: null,
  source_event_id: 'evt-wait-1',
  cursor_event_id: 'evt-wait-1',
  durability: 'persistent',
  visibility: 'timeline',
} as const

test('normalizeWaitPayload should parse input text payload', () => {
  const payload = normalizeWaitPayload({
    kind: 'input_text',
    title: '需要补充信息',
    prompt: '请输入目标网站地址',
    response_key: 'website',
    default_value: 'https://example.com',
  })

  assert.deepEqual(payload, {
    kind: 'input_text',
    title: '需要补充信息',
    prompt: '请输入目标网站地址',
    details: '',
    attachments: [],
    suggest_user_takeover: 'none',
    placeholder: '',
    submit_label: '继续执行',
    response_key: 'website',
    default_value: 'https://example.com',
    multiline: true,
    allow_empty: false,
  })
})

test('parseWaitEventContext should expose structured payload fields', () => {
  const context = parseWaitEventContext({
    runtime,
    interrupt_id: 'interrupt-1',
    payload: {
      kind: 'select',
      title: '请选择下一步',
      prompt: '你希望我怎么继续？',
      attachments: ['/tmp/spec.md'],
      suggest_user_takeover: 'browser',
      options: [
        { label: '继续', resume_value: { approved: true } },
      ],
    },
  })

  assert.ok(context)
  assert.equal(context?.interruptId, 'interrupt-1')
  assert.equal(context?.title, '请选择下一步')
  assert.equal(context?.prompt, '你希望我怎么继续？')
  assert.deepEqual(context?.attachments, ['/tmp/spec.md'])
  assert.equal(context?.suggestUserTakeover, true)
  assert.equal(context?.payload?.kind, 'select')
})

test('buildResumeValueFromWaitPayload should respect response_key for text input', () => {
  const resumeValue = buildResumeValueFromWaitPayload(
    {
      kind: 'input_text',
      prompt: '请输入账号',
      response_key: 'account',
      submit_label: '提交',
      attachments: [],
      suggest_user_takeover: 'none',
    },
    'demo-user',
  )

  assert.deepEqual(resumeValue, { account: 'demo-user' })
})
