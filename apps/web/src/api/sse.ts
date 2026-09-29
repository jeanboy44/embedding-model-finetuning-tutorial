// POST 요청의 Server-Sent Events 응답을 읽는다 (EventSource는 GET만 되므로 fetch 스트림을 직접 파싱).

export interface SSEMessage {
  event: string
  data: string
}

/** 조각으로 들어오는 텍스트를 받아 완성된 이벤트만 돌려준다. 이벤트는 빈 줄로 끝난다. */
export class SSEParser {
  private buffer = ''

  push(chunk: string): SSEMessage[] {
    this.buffer += chunk.replace(/\r\n/g, '\n')
    const messages: SSEMessage[] = []
    let end: number
    while ((end = this.buffer.indexOf('\n\n')) !== -1) {
      const block = this.buffer.slice(0, end)
      this.buffer = this.buffer.slice(end + 2)
      const message = parseBlock(block)
      if (message) messages.push(message)
    }
    return messages
  }
}

function parseBlock(block: string): SSEMessage | null {
  let event = 'message'
  const data: string[] = []
  for (const line of block.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim()
    else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''))
  }
  return data.length ? { event, data: data.join('\n') } : null
}

export async function* readSSE(response: Response): AsyncGenerator<SSEMessage> {
  if (!response.body) return
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader()
  const parser = new SSEParser()
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    yield* parser.push(value)
  }
  yield* parser.push('\n\n')
}
