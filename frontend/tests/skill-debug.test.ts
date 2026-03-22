import assert from 'node:assert/strict'
import test from 'node:test'

import { buildSkillDebugItems } from '../src/lib/skill-debug'
import type { SSEEventData } from '../src/lib/api/types'

function eventOf(type: SSEEventData['type'], data: unknown): SSEEventData {
  return { type, data } as SSEEventData
}

test('buildSkillDebugItems should return empty when no skill/subgraph signals exist', () => {
  const events: SSEEventData[] = [
    eventOf('message', { role: 'assistant', message: 'hello' }),
    eventOf('step', { id: 's1', status: 'running', description: 'execute step' }),
    eventOf('done', { result: 'completed' }),
  ]

  const items = buildSkillDebugItems(events)
  assert.equal(items.length, 0)
})

test('buildSkillDebugItems should parse flat runtime skill fields', () => {
  const events: SSEEventData[] = [
    eventOf('step', {
      id: 's1',
      status: 'completed',
      description: 'execute step',
      created_at: 1712222222,
      extensions: {
        runtime: {
          skill_id: 'planner_react.execute_step',
          skill_version: '1.0.0',
          subgraph: 'planner_react.execute_step',
          node: 'execute_step_skill',
          input: { step_description: '收集需求' },
          output: { success: true, result: '已完成' },
        },
      },
    }),
  ]

  const items = buildSkillDebugItems(events)
  assert.equal(items.length, 1)
  assert.equal(items[0]?.skillId, 'planner_react.execute_step')
  assert.equal(items[0]?.skillVersion, '1.0.0')
  assert.equal(items[0]?.subgraph, 'planner_react.execute_step')
  assert.equal(items[0]?.node, 'execute_step_skill')
  assert.deepEqual(items[0]?.input, { step_description: '收集需求' })
  assert.deepEqual(items[0]?.output, { success: true, result: '已完成' })
  assert.equal(items[0]?.stepId, 's1')
  assert.equal(items[0]?.timestamp, 1712222222000)
})

test('buildSkillDebugItems should parse nested skill payload with failure info', () => {
  const events: SSEEventData[] = [
    eventOf('error', {
      error: 'fallback',
      extensions: {
        runtime: {
          skill: {
            id: 'planner_react.execute_step',
            version: '1.0.0',
            subgraph: 'planner_react.execute_step',
            failed_node: 'execute_step_skill',
            status: 'failed',
            error: 'tool timeout',
            input: { step_description: '打开网页' },
          },
        },
      },
    }),
  ]

  const items = buildSkillDebugItems(events)
  assert.equal(items.length, 1)
  assert.equal(items[0]?.skillId, 'planner_react.execute_step')
  assert.equal(items[0]?.failedNode, 'execute_step_skill')
  assert.equal(items[0]?.status, 'failed')
  assert.equal(items[0]?.error, 'tool timeout')
  assert.deepEqual(items[0]?.input, { step_description: '打开网页' })
})
