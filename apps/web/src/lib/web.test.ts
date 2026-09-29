import { describe, expect, it } from 'vitest'
import { SSEParser } from '@/api/sse'
import { citationIndex, citedNumbers, linkCitations } from './citations'

describe('SSEParser', () => {
  it('조각 경계에 걸친 이벤트를 모아서 돌려준다', () => {
    const parser = new SSEParser()
    expect(parser.push('event: hits\ndata: {"hits"')).toEqual([])
    expect(parser.push(': []}\n\nevent: delta\nda')).toEqual([{ event: 'hits', data: '{"hits": []}' }])
    expect(parser.push('ta: {"text": "가"}\n\n')).toEqual([{ event: 'delta', data: '{"text": "가"}' }])
  })

  it('CRLF와 여러 줄 data를 처리한다', () => {
    const parser = new SSEParser()
    expect(parser.push('event: done\r\ndata: a\r\ndata: b\r\n\r\n')).toEqual([{ event: 'done', data: 'a\nb' }])
  })
})

describe('linkCitations', () => {
  it('범위 안의 [n]과 [n, m]을 인용 링크로 바꾼다', () => {
    expect(linkCitations('휴일 [1]. 수당 [1, 3]', 3)).toBe('휴일 [1](#cite-1). 수당 [1](#cite-1)[3](#cite-3)')
  })

  it('범위를 벗어난 번호와 일반 대괄호는 그대로 둔다', () => {
    expect(linkCitations('[4] [별표] [0]', 3)).toBe('[4] [별표] [0]')
  })

  it('인용 번호를 모으고 링크를 해석한다', () => {
    expect([...citedNumbers('a [2] b [1, 2]')].sort()).toEqual([1, 2])
    expect(citationIndex('#cite-12')).toBe(12)
    expect(citationIndex('https://x')).toBeNull()
  })
})
