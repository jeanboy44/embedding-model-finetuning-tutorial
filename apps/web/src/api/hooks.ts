import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useCallback, useRef, useState } from 'react'
import { api } from './client'
import type { Hit, Notebook } from './types'

export const keys = {
  health: ['health'] as const,
  laws: ['laws'] as const,
  notebooks: ['notebooks'] as const,
  notebook: (id: string) => ['notebooks', id] as const,
  messages: (id: string) => ['notebooks', id, 'messages'] as const,
  notes: (id: string) => ['notebooks', id, 'notes'] as const,
  article: (parentId: string) => ['article', parentId] as const,
}

export const useHealth = () => useQuery({ queryKey: keys.health, queryFn: api.health, retry: 1 })
// 법령 목록과 조문은 인덱스를 다시 만들기 전에는 바뀌지 않는다
export const useLaws = () => useQuery({ queryKey: keys.laws, queryFn: api.laws, staleTime: Infinity })
export const useArticle = (parentId: string | null) =>
  useQuery({
    queryKey: keys.article(parentId ?? ''),
    queryFn: () => api.article(parentId!),
    enabled: !!parentId,
    staleTime: Infinity,
  })

export const useNotebooks = () => useQuery({ queryKey: keys.notebooks, queryFn: api.notebooks })
export const useNotebook = (id: string) => useQuery({ queryKey: keys.notebook(id), queryFn: () => api.notebook(id) })
export const useMessages = (id: string) => useQuery({ queryKey: keys.messages(id), queryFn: () => api.messages(id) })
export const useNotes = (id: string) => useQuery({ queryKey: keys.notes(id), queryFn: () => api.notes(id) })

export function useCreateNotebook() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ title, laws }: { title: string; laws?: string[] }) => api.createNotebook(title, laws),
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.notebooks }),
  })
}

export function useUpdateNotebook(id: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (patch: { title?: string; laws?: string[] }) => api.updateNotebook(id, patch),
    onSuccess: (notebook: Notebook) => {
      qc.setQueryData(keys.notebook(id), notebook)
      qc.invalidateQueries({ queryKey: keys.notebooks, exact: true })
    },
  })
}

export function useDeleteNotebook() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => api.deleteNotebook(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.notebooks }),
  })
}

export function useClearMessages(id: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () => api.clearMessages(id),
    onSuccess: () => qc.setQueryData(keys.messages(id), []),
  })
}

export function useNoteMutations(id: string) {
  const qc = useQueryClient()
  const refresh = () => qc.invalidateQueries({ queryKey: keys.notes(id) })
  return {
    create: useMutation({
      mutationFn: (note: { title: string; content: string; citations?: Hit[] }) => api.createNote(id, note),
      onSuccess: refresh,
    }),
    update: useMutation({
      mutationFn: ({ noteId, ...patch }: { noteId: string; title?: string; content?: string }) =>
        api.updateNote(id, noteId, patch),
      onSuccess: refresh,
    }),
    remove: useMutation({ mutationFn: (noteId: string) => api.deleteNote(id, noteId), onSuccess: refresh }),
  }
}

/** 지금 스트리밍 중인 한 턴 (저장되기 전). */
export interface PendingTurn {
  query: string
  hits: Hit[]
  text: string
  error: string | null
  done: boolean
}

/** 노트북 채팅: 답을 받는 동안 pending에 쌓고, 끝나면 저장된 대화 기록을 다시 불러온다. */
export function useChat(id: string) {
  const qc = useQueryClient()
  const [pending, setPending] = useState<PendingTurn | null>(null)
  const abort = useRef<AbortController | null>(null)

  const send = useCallback(
    async (query: string) => {
      abort.current?.abort()
      const controller = new AbortController()
      abort.current = controller
      let turn: PendingTurn = { query, hits: [], text: '', error: null, done: false }
      const update = (patch: Partial<PendingTurn>) => {
        turn = { ...turn, ...patch }
        setPending(turn)
      }
      update({})
      try {
        for await (const ev of api.chat(id, query, controller.signal)) {
          if (ev.event === 'hits') update({ hits: ev.data.hits })
          else if (ev.event === 'delta') update({ text: turn.text + ev.data.text })
          else if (ev.event === 'done') update({ text: ev.data.answer || turn.text, error: ev.data.error, done: true })
        }
        await qc.invalidateQueries({ queryKey: keys.messages(id) })
        qc.invalidateQueries({ queryKey: keys.notebooks, exact: true })
        setPending(null)
      } catch (e) {
        if (controller.signal.aborted) return
        update({ error: e instanceof Error ? e.message : String(e), done: true })
      }
    },
    [id, qc],
  )

  const dismiss = useCallback(() => setPending(null), [])
  return { pending, send, dismiss, busy: !!pending && !pending.done }
}
