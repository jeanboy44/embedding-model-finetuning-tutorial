// apps/api/src/ragkit_api/schemas.py 와 1:1

export interface Hit {
  id: string
  parent_id: string | null
  title: string | null
  text: string
  score: number
  law_name: string | null
  law_type: string | null
  theme: string | null
  article_no: string | null
  article_title: string | null
  chapter: string | null
  source_url: string | null
}

export interface Law {
  law_name: string
  law_type: string
  theme: string
  doc_count: number
}

export interface Health {
  status: string
  model_key: string
  doc_count: number
  llm_available: boolean
}

export interface Notebook {
  id: string
  title: string
  laws: string[]
  created_at: string
  updated_at: string
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations: Hit[]
  error: string | null
  created_at: string
}

export interface Note {
  id: string
  title: string
  content: string
  citations: Hit[]
  created_at: string
  updated_at: string
}

export interface DonePayload {
  answer: string
  input_tokens: number
  output_tokens: number
  latency_s: number
  error: string | null
  message_id?: string
}

export type ChatEvent =
  | { event: 'hits'; data: { hits: Hit[] } }
  | { event: 'delta'; data: { text: string } }
  | { event: 'done'; data: DonePayload }
