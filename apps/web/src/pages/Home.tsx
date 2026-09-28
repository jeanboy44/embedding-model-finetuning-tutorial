import { MoreHorizontal, Plus, Trash2 } from 'lucide-react'
import { Link, useNavigate } from 'react-router'
import { useCreateNotebook, useDeleteNotebook, useLaws, useNotebooks } from '@/api/hooks'
import type { Notebook } from '@/api/types'
import { HealthBadge } from '@/components/HealthBadge'
import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { Skeleton } from '@/components/ui/skeleton'
import { relativeTime, themeLabel } from '@/lib/format'

export function Home() {
  const { data: notebooks, isLoading, error } = useNotebooks()
  const { data: laws = [] } = useLaws()
  const create = useCreateNotebook()
  const navigate = useNavigate()
  const themes = [...new Set(laws.map((l) => l.theme))]

  const newNotebook = () =>
    create.mutate({ title: '제목 없는 노트북' }, { onSuccess: (nb) => navigate(`/notebooks/${nb.id}`) })

  return (
    <div className="min-h-svh">
      <header className="mx-auto max-w-6xl px-6 pt-8 sm:px-10">
        <div className="flex items-center justify-between border-b-2 border-foreground pb-2 text-xs">
          <span className="font-mono tracking-widest text-muted-foreground">RAGKIT · 법령 검색 노트</span>
          <HealthBadge />
        </div>
        <div className="grid gap-6 border-b py-10 sm:grid-cols-[1fr_auto] sm:items-end sm:py-14">
          <div className="animate-rise">
            <h1 className="font-serif text-[3.2rem] leading-[0.95] font-black tracking-tighter sm:text-[5.5rem]">
              법령 <span className="text-seal">노트</span>
            </h1>
            <p className="mt-5 max-w-lg text-[0.95rem] leading-relaxed text-ink-soft">
              법령을 골라 노트북을 만들고, 일상의 말로 물어보세요. 답은 고른 법령의 조문만 근거로 하고, 모든 문장에
              출처 번호가 붙습니다.
            </p>
          </div>
          <dl className="grid grid-cols-3 gap-6 text-right font-mono text-xs text-muted-foreground sm:gap-8">
            <Stat label="법령" value={laws.length} />
            <Stat label="조문 조각" value={laws.reduce((s, l) => s + l.doc_count, 0)} />
            <Stat label="분야" value={themes.length} />
          </dl>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-10 sm:px-10">
        <div className="mb-5 flex items-baseline justify-between">
          <h2 className="font-serif text-xl font-bold">내 노트북</h2>
          <span className="text-sm text-muted-foreground">{notebooks?.length ?? 0}권</span>
        </div>
        {error && <p className="mb-4 text-sm text-seal">{error.message}</p>}
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <button
            type="button"
            onClick={newNotebook}
            disabled={create.isPending}
            className="group flex min-h-44 animate-rise flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed border-foreground/25 text-muted-foreground transition hover:border-seal hover:text-seal"
          >
            <span className="grid size-11 place-items-center rounded-full border-2 border-current transition group-hover:rotate-90">
              <Plus />
            </span>
            <span className="font-serif font-semibold">새 노트북</span>
          </button>
          {isLoading && [0, 1].map((i) => <Skeleton key={i} className="min-h-44 rounded-lg" />)}
          {notebooks?.map((nb, i) => <NotebookCard key={nb.id} notebook={nb} index={i} />)}
        </div>
      </main>
    </div>
  )
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <dd className="font-serif text-2xl font-bold text-foreground tabular-nums sm:text-3xl">{value.toLocaleString()}</dd>
      <dt className="mt-1 tracking-widest">{label}</dt>
    </div>
  )
}

function NotebookCard({ notebook, index }: { notebook: Notebook; index: number }) {
  const remove = useDeleteNotebook()
  const { data: laws = [] } = useLaws()
  const themes = [...new Set(notebook.laws.map((n) => laws.find((l) => l.law_name === n)?.theme))].filter(Boolean)

  return (
    <div
      style={{ animationDelay: `${60 + index * 50}ms` }}
      className="group relative flex min-h-44 animate-rise flex-col rounded-lg border bg-card shadow-[0_1px_0_oklch(0.85_0.02_78)] transition hover:-translate-y-1 hover:shadow-[0_14px_30px_-18px_oklch(0.3_0.02_60/50%)]"
    >
      {/* 책등 */}
      <span className="absolute inset-y-0 left-0 w-1.5 rounded-l-lg bg-foreground/85 transition-colors group-hover:bg-seal" />
      <Link to={`/notebooks/${notebook.id}`} className="flex flex-1 flex-col py-5 pr-5 pl-7">
        <p className="text-[0.7rem] tracking-widest text-muted-foreground">
          {themes.length ? themes.map(themeLabel).join(' · ') : '전체 법령'}
        </p>
        <h3 className="mt-2 line-clamp-2 pr-6 font-serif text-lg leading-snug font-bold">{notebook.title}</h3>
        <p className="mt-2 line-clamp-2 text-sm text-muted-foreground">
          {notebook.laws.length ? notebook.laws.join(', ') : '소스를 고르지 않았습니다'}
        </p>
        <p className="mt-auto pt-4 font-mono text-[0.7rem] text-muted-foreground">
          소스 {notebook.laws.length} · {relativeTime(notebook.updated_at)}
        </p>
      </Link>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" size="icon-sm" aria-label="노트북 메뉴" className="absolute top-3 right-3">
            <MoreHorizontal />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem
            variant="destructive"
            onSelect={() => {
              if (confirm(`'${notebook.title}' 노트북을 지울까요? 대화와 노트도 함께 지워집니다.`)) remove.mutate(notebook.id)
            }}
          >
            <Trash2 /> 지우기
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  )
}
