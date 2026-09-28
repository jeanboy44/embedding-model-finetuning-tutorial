import { readSSE } from './sse'
import type { ChatEvent, Health, Hit, Law, Message, Note, Notebook } from './types'

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!res.ok) throw new ApiError(res.status, await errorMessage(res))
  return res.status === 204 ? (undefined as T) : res.json()
}

async function errorMessage(res: Response): Promise<string> {
  try {
    const body = await res.json()
    if (typeof body.detail === 'string') return body.detail
    if (Array.isArray(body.detail)) return body.detail.map((d: { msg: string }) => d.msg).join(', ')
  } catch {
    // 본문이 JSON이 아니면 상태 코드로 안내
  }
  return res.status === 502 || res.status === 504
    ? 'API 서버에 연결할 수 없습니다. ragkit-api가 떠 있는지 확인하세요.'
    : `요청 실패 (${res.status})`
}

const json = (body: unknown): RequestInit => ({ body: JSON.stringify(body) })

export const api = {
  health: () => request<Health>('/health'),
  laws: () => request<Law[]>('/laws'),
  article: (parentId: string) => request<Hit[]>(`/articles/${encodeURIComponent(parentId)}`),

  notebooks: () => request<Notebook[]>('/notebooks'),
  notebook: (id: string) => request<Notebook>(`/notebooks/${id}`),
  createNotebook: (title: string, laws: string[] = []) =>
    request<Notebook>('/notebooks', { method: 'POST', ...json({ title, laws }) }),
  updateNotebook: (id: string, patch: { title?: string; laws?: string[] }) =>
    request<Notebook>(`/notebooks/${id}`, { method: 'PATCH', ...json(patch) }),
  deleteNotebook: (id: string) => request<void>(`/notebooks/${id}`, { method: 'DELETE' }),

  messages: (id: string) => request<Message[]>(`/notebooks/${id}/messages`),
  clearMessages: (id: string) => request<void>(`/notebooks/${id}/messages`, { method: 'DELETE' }),

  notes: (id: string) => request<Note[]>(`/notebooks/${id}/notes`),
  createNote: (id: string, note: { title: string; content: string; citations?: Hit[] }) =>
    request<Note>(`/notebooks/${id}/notes`, { method: 'POST', ...json(note) }),
  updateNote: (id: string, noteId: string, patch: { title?: string; content?: string }) =>
    request<Note>(`/notebooks/${id}/notes/${noteId}`, { method: 'PATCH', ...json(patch) }),
  deleteNote: (id: string, noteId: string) =>
    request<void>(`/notebooks/${id}/notes/${noteId}`, { method: 'DELETE' }),

  /** 노트북 채팅: 근거 조문 → 답 조각들 → 완료 순서로 이벤트를 내보낸다. */
  async *chat(id: string, query: string, signal?: AbortSignal): AsyncGenerator<ChatEvent> {
    const res = await fetch(`/api/notebooks/${id}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
      signal,
    })
    if (!res.ok) throw new ApiError(res.status, await errorMessage(res))
    for await (const message of readSSE(res)) {
      yield { event: message.event, data: JSON.parse(message.data) } as ChatEvent
    }
  },
}
