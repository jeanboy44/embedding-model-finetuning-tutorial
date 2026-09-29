import { ArrowUp, Check, Copy, NotebookPen, Trash2, TriangleAlert } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { useChat, useClearMessages, useMessages, useNoteMutations } from '@/api/hooks'
import type { Hit, Message, Notebook } from '@/api/types'
import { Answer, CitationChip } from '@/components/Citation'
import { PanelHeader } from '@/components/PanelHeader'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { Textarea } from '@/components/ui/textarea'
import { citedNumbers } from '@/lib/citations'
import { hitTitle } from '@/lib/format'
import { cn } from '@/lib/utils'

const EXAMPLES = [
  '밤 10시 넘어서 일하면 수당을 더 받나요?',
  '아르바이트도 주휴수당을 받을 수 있어?',
  '전세 계약이 끝났는데 집주인이 보증금을 안 돌려줘요',
  '음주운전으로 처음 걸리면 어떤 처벌을 받나요?',
]

interface Turn {
  key: string
  query: string
  answer: string
  hits: Hit[]
  error: string | null
  streaming: boolean
  loading: boolean
}

/** 저장된 메시지를 질문-답 쌍으로 묶는다. */
function toTurns(messages: Message[]): Turn[] {
  const turns: Turn[] = []
  for (const m of messages) {
    if (m.role === 'user') {
      turns.push({ key: m.id, query: m.content, answer: '', hits: [], error: null, streaming: false, loading: false })
    } else if (turns.length) {
      Object.assign(turns[turns.length - 1], { answer: m.content, hits: m.citations, error: m.error })
    }
  }
  return turns
}

export function ChatPanel({ notebook, onOpenHit }: { notebook: Notebook; onOpenHit: (hit: Hit) => void }) {
  const { data: messages, isLoading } = useMessages(notebook.id)
  const { pending, send, busy, dismiss } = useChat(notebook.id)
  const clear = useClearMessages(notebook.id)
  const bottom = useRef<HTMLDivElement>(null)

  const turns = toTurns(messages ?? [])
  if (pending) {
    turns.push({
      key: 'pending',
      query: pending.query,
      answer: pending.text,
      hits: pending.hits,
      error: pending.error,
      streaming: !pending.done && pending.hits.length > 0,
      loading: !pending.done && pending.hits.length === 0,
    })
  }

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: 'end' })
  }, [turns.length, pending?.text, pending?.hits.length])

  const scope = notebook.laws.length
    ? notebook.laws.length === 1
      ? notebook.laws[0]
      : `${notebook.laws[0]} 외 ${notebook.laws.length - 1}개`
    : '전체 법령'

  return (
    <div className="flex h-full min-h-0 flex-col">
      <PanelHeader title="대화" hint={`${scope}에서 찾아 답합니다`}>
        {turns.length > 0 && !busy && (
          <Button
            variant="ghost"
            size="sm"
            className="text-muted-foreground"
            onClick={() => {
              if (confirm('대화 기록을 모두 지울까요? 노트는 남습니다.')) {
                dismiss()
                clear.mutate()
              }
            }}
          >
            <Trash2 /> 비우기
          </Button>
        )}
      </PanelHeader>

      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl px-5 py-6 sm:px-8">
          {isLoading && (
            <div className="space-y-3">
              <Skeleton className="h-6 w-2/3" />
              <Skeleton className="h-24 w-full" />
            </div>
          )}
          {!isLoading && turns.length === 0 && <EmptyChat onPick={send} />}
          <div className="space-y-10">
            {turns.map((turn) => (
              <TurnView key={turn.key} turn={turn} notebookId={notebook.id} onOpenHit={onOpenHit} />
            ))}
          </div>
          <div ref={bottom} className="h-4" />
        </div>
      </div>

      <Composer onSend={send} busy={busy} />
    </div>
  )
}

function EmptyChat({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="animate-rise py-8 sm:py-14">
      <p className="font-serif text-[2rem] leading-tight font-black tracking-tight sm:text-[2.6rem]">
        무엇이든 물어보세요.
        <br />
        <span className="text-seal">조문</span>으로 답합니다.
      </p>
      <p className="mt-4 max-w-md text-sm leading-relaxed text-muted-foreground">
        질문과 가까운 조문을 먼저 찾고, 그 조문만 근거로 답합니다. 답 속 번호를 누르면 원문이 열립니다.
      </p>
      <div className="mt-8 grid gap-2 sm:grid-cols-2">
        {EXAMPLES.map((q, i) => (
          <button
            key={q}
            type="button"
            onClick={() => onPick(q)}
            style={{ animationDelay: `${120 + i * 60}ms` }}
            className="animate-rise rounded-lg border border-dashed border-foreground/20 bg-card/60 px-4 py-3 text-left text-sm leading-snug transition hover:-translate-y-0.5 hover:border-seal/60 hover:bg-card"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  )
}

function TurnView({ turn, notebookId, onOpenHit }: { turn: Turn; notebookId: string; onOpenHit: (hit: Hit) => void }) {
  const { create } = useNoteMutations(notebookId)
  const [copied, setCopied] = useState(false)
  const finished = !turn.streaming && !turn.loading

  return (
    <section className="animate-rise">
      <div className="flex gap-3">
        <Seal>問</Seal>
        <h3 className="pt-0.5 font-serif text-lg leading-snug font-bold text-balance">{turn.query}</h3>
      </div>

      <div className="mt-4 flex gap-3">
        <Seal filled>答</Seal>
        <div className="min-w-0 flex-1 pt-0.5">
          {turn.loading && (
            <p className="flex items-center gap-2 text-sm text-muted-foreground">
              <span className="size-1.5 animate-ping rounded-full bg-seal" /> 조문을 찾는 중…
            </p>
          )}
          {(turn.answer || turn.streaming) && (
            <Answer text={turn.answer} hits={turn.hits} onOpen={onOpenHit} streaming={turn.streaming} />
          )}
          {turn.error && (
            <div className="mt-2 flex gap-2 rounded-md border border-seal/30 bg-seal/5 px-3 py-2 text-sm text-seal">
              <TriangleAlert className="mt-0.5 size-4 shrink-0" />
              <span>{turn.error}</span>
            </div>
          )}
          {turn.hits.length > 0 && <SourceList hits={turn.hits} answer={turn.answer} onOpen={onOpenHit} />}
          {finished && turn.answer && (
            <div className="mt-3 flex gap-1">
              <Button
                variant="ghost"
                size="xs"
                className="text-muted-foreground"
                disabled={create.isPending || create.isSuccess}
                onClick={() => create.mutate({ title: turn.query, content: turn.answer, citations: turn.hits })}
              >
                {create.isSuccess ? <Check /> : <NotebookPen />}
                {create.isSuccess ? '노트에 저장됨' : '노트에 저장'}
              </Button>
              <Button
                variant="ghost"
                size="xs"
                className="text-muted-foreground"
                onClick={async () => {
                  await navigator.clipboard.writeText(turn.answer)
                  setCopied(true)
                  setTimeout(() => setCopied(false), 1500)
                }}
              >
                {copied ? <Check /> : <Copy />} {copied ? '복사됨' : '복사'}
              </Button>
            </div>
          )}
        </div>
      </div>
    </section>
  )
}

/** 근거 조문 목록. 답에서 실제로 인용한 조문은 진하게, 나머지는 흐리게. */
function SourceList({ hits, answer, onOpen }: { hits: Hit[]; answer: string; onOpen: (hit: Hit) => void }) {
  const cited = citedNumbers(answer)
  return (
    <div className="mt-4 border-t border-dashed pt-3">
      <p className="mb-1.5 text-[0.7rem] font-medium tracking-[0.2em] text-muted-foreground">근거 조문</p>
      <ol className="grid grid-cols-1 gap-x-4 gap-y-0.5 sm:grid-cols-2">
        {hits.map((hit, i) => (
          <li key={hit.id} className="min-w-0">
            <button
              type="button"
              onClick={() => onOpen(hit)}
              className={cn(
                'flex w-full items-baseline gap-1.5 rounded px-1 py-0.5 text-left text-[0.8rem] transition hover:bg-muted',
                cited.size > 0 && !cited.has(i + 1) && 'opacity-55',
              )}
            >
              <CitationChip n={i + 1} className="pointer-events-none" />
              <span className="truncate font-serif">{hitTitle(hit)}</span>
              <span className="ml-auto shrink-0 font-mono text-[0.65rem] text-muted-foreground">
                {hit.score.toFixed(2)}
              </span>
            </button>
          </li>
        ))}
      </ol>
    </div>
  )
}

function Seal({ children, filled = false }: { children: string; filled?: boolean }) {
  return (
    <span
      aria-hidden
      className={cn(
        'grid size-7 shrink-0 animate-stamp place-items-center rounded-[4px] font-serif text-[0.8rem] font-black',
        filled ? 'bg-seal text-seal-foreground shadow-[0_1px_0_oklch(0.35_0.12_28)]' : 'border-[1.5px] border-foreground/70',
      )}
    >
      {children}
    </span>
  )
}

function Composer({ onSend, busy }: { onSend: (q: string) => void; busy: boolean }) {
  const [text, setText] = useState('')
  const submit = () => {
    const q = text.trim()
    if (!q || busy) return
    onSend(q)
    setText('')
  }
  return (
    <div className="shrink-0 border-t bg-background/80 px-4 py-3 backdrop-blur sm:px-8">
      <form
        className="mx-auto flex max-w-3xl items-end gap-2 rounded-xl border bg-card p-2 shadow-[0_1px_0_oklch(0.8_0.02_78),0_8px_24px_-12px_oklch(0.3_0.02_60/25%)] focus-within:border-foreground/40"
        onSubmit={(e) => {
          e.preventDefault()
          submit()
        }}
      >
        <Textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault()
              submit()
            }
          }}
          rows={1}
          placeholder="상황을 편하게 적어 보세요"
          aria-label="질문"
          className="max-h-40 min-h-9 resize-none border-0 bg-transparent shadow-none focus-visible:ring-0"
        />
        <Button type="submit" size="icon" disabled={busy || !text.trim()} aria-label="보내기" className="rounded-lg bg-seal hover:bg-seal/85">
          <ArrowUp />
        </Button>
      </form>
      <p className="mx-auto mt-1.5 max-w-3xl text-center text-[0.7rem] text-muted-foreground">
        Enter 보내기 · Shift+Enter 줄바꿈 · 답은 참고용이며 법률 자문이 아닙니다
      </p>
    </div>
  )
}
