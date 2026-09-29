import { ChevronLeft, NotebookPen, Plus, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { useNoteMutations, useNotes } from '@/api/hooks'
import type { Hit, Note } from '@/api/types'
import { Answer } from '@/components/Citation'
import { PanelHeader } from '@/components/PanelHeader'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Textarea } from '@/components/ui/textarea'
import { relativeTime } from '@/lib/format'

export function NotesPanel({ notebookId, onOpenHit }: { notebookId: string; onOpenHit: (hit: Hit) => void }) {
  const { data: notes = [] } = useNotes(notebookId)
  const { create } = useNoteMutations(notebookId)
  const [openId, setOpenId] = useState<string | null>(null)
  const open = notes.find((n) => n.id === openId)

  if (open) {
    return <NoteEditor key={open.id} note={open} notebookId={notebookId} onBack={() => setOpenId(null)} onOpenHit={onOpenHit} />
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <PanelHeader title="노트" hint={notes.length ? `${notes.length}개` : undefined}>
        <Button
          variant="outline"
          size="sm"
          disabled={create.isPending}
          onClick={() =>
            create.mutate({ title: '새 메모', content: '' }, { onSuccess: (note) => setOpenId(note.id) })
          }
        >
          <Plus /> 메모
        </Button>
      </PanelHeader>
      <ScrollArea className="min-h-0 flex-1">
        {notes.length === 0 ? (
          <div className="px-6 py-10 text-center">
            <NotebookPen className="mx-auto size-8 text-muted-foreground/60" strokeWidth={1.25} />
            <p className="mt-3 font-serif font-semibold">노트가 비어 있습니다</p>
            <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
              좋은 답은 &lsquo;노트에 저장&rsquo;으로 모아 두세요.
              <br />
              인용한 조문도 함께 남습니다.
            </p>
          </div>
        ) : (
          <ul className="space-y-2.5 p-3">
            {notes.map((note, i) => (
              <li key={note.id} style={{ animationDelay: `${i * 40}ms` }} className="animate-rise">
                <button
                  type="button"
                  onClick={() => setOpenId(note.id)}
                  className="relative w-full overflow-hidden rounded-md border bg-card px-4 py-3 text-left shadow-[0_1px_0_oklch(0.85_0.02_78)] transition hover:-translate-y-0.5 hover:shadow-[0_6px_18px_-10px_oklch(0.3_0.02_60/40%)]"
                >
                  {/* 메모지 왼쪽 여백선 */}
                  <span className="absolute inset-y-0 left-2 w-px bg-seal/25" />
                  <p className="line-clamp-2 font-serif text-[0.92rem] leading-snug font-bold">{note.title || '제목 없음'}</p>
                  <p className="mt-1 line-clamp-3 text-[0.8rem] leading-relaxed text-muted-foreground">
                    {note.content.replace(/\[(\d+(?:,\s*\d+)*)\]/g, '') || '내용 없음'}
                  </p>
                  <p className="mt-2 flex items-center gap-2 text-[0.7rem] text-muted-foreground">
                    {relativeTime(note.updated_at)}
                    {note.citations.length > 0 && <span>· 인용 {note.citations.length}</span>}
                  </p>
                </button>
              </li>
            ))}
          </ul>
        )}
      </ScrollArea>
    </div>
  )
}

function NoteEditor({
  note,
  notebookId,
  onBack,
  onOpenHit,
}: {
  note: Note
  notebookId: string
  onBack: () => void
  onOpenHit: (hit: Hit) => void
}) {
  const { update, remove } = useNoteMutations(notebookId)
  const [title, setTitle] = useState(note.title)
  const [content, setContent] = useState(note.content)
  // 답에서 저장한 노트는 인용 칩이 살아 있게 읽기 모드로 연다
  const [editing, setEditing] = useState(note.citations.length === 0)
  const dirty = title !== note.title || content !== note.content

  const save = () => dirty && update.mutate({ noteId: note.id, title, content })

  return (
    <div className="flex h-full min-h-0 flex-col animate-rise">
      <div className="flex h-12 shrink-0 items-center gap-1 border-b px-2">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => {
            save()
            onBack()
          }}
        >
          <ChevronLeft /> 노트
        </Button>
        <span className="text-xs text-muted-foreground">
          {update.isPending ? '저장 중…' : dirty ? '저장 안 됨' : relativeTime(note.updated_at)}
        </span>
        <div className="ml-auto flex gap-1">
          {note.citations.length > 0 && (
            <Button variant="ghost" size="sm" onClick={() => setEditing((e) => !e)}>
              {editing ? '보기' : '편집'}
            </Button>
          )}
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label="노트 지우기"
            onClick={() => {
              if (confirm('이 노트를 지울까요?')) remove.mutate(note.id, { onSuccess: onBack })
            }}
          >
            <Trash2 />
          </Button>
        </div>
      </div>
      <ScrollArea className="min-h-0 flex-1">
        <div className="px-5 py-4">
          <Textarea
            value={title}
            onChange={(e) => setTitle(e.target.value.replace(/\n/g, ' '))}
            onBlur={save}
            rows={1}
            aria-label="노트 제목"
            className="min-h-0 resize-none border-0 bg-transparent px-0 py-0 font-serif text-lg leading-snug font-bold shadow-none focus-visible:ring-0 md:text-lg"
          />
          {editing ? (
            <Textarea
              value={content}
              onChange={(e) => setContent(e.target.value)}
              onBlur={save}
              placeholder="메모를 적으세요"
              aria-label="노트 내용"
              className="mt-2 min-h-72 resize-none border-0 bg-transparent px-0 leading-relaxed shadow-none focus-visible:ring-0"
            />
          ) : (
            <div className="mt-2">
              <Answer text={content} hits={note.citations} onOpen={onOpenHit} />
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  )
}
