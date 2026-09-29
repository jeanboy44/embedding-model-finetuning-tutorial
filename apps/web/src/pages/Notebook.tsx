import { ArrowLeft } from 'lucide-react'
import { useState } from 'react'
import { Link, useParams } from 'react-router'
import { useNotebook, useUpdateNotebook } from '@/api/hooks'
import type { Hit, Notebook } from '@/api/types'
import { ChatPanel } from '@/components/ChatPanel'
import { HealthBadge } from '@/components/HealthBadge'
import { NotesPanel } from '@/components/NotesPanel'
import { SourcesPanel, type ViewerTarget } from '@/components/SourcesPanel'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { articleKey } from '@/lib/format'
import { cn } from '@/lib/utils'

type Tab = 'sources' | 'chat' | 'notes'
const TABS: { id: Tab; label: string }[] = [
  { id: 'sources', label: '소스' },
  { id: 'chat', label: '대화' },
  { id: 'notes', label: '노트' },
]

export function NotebookPage() {
  const { id = '' } = useParams()
  const { data: notebook, error } = useNotebook(id)
  const [viewer, setViewer] = useState<ViewerTarget | null>(null)
  const [tab, setTab] = useState<Tab>('chat')

  const openHit = (hit: Hit) => {
    setViewer({ parentId: articleKey(hit), highlightId: hit.id })
    setTab('sources') // 좁은 화면에서는 소스 탭으로 넘어간다
  }

  if (error) {
    return (
      <div className="grid min-h-svh place-items-center p-6 text-center">
        <div>
          <p className="font-serif text-2xl font-bold">노트북을 열 수 없습니다</p>
          <p className="mt-2 text-sm text-muted-foreground">{error.message}</p>
          <Button className="mt-5" asChild>
            <Link to="/">노트북 목록으로</Link>
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-svh flex-col">
      <header className="flex h-14 shrink-0 items-center gap-2 border-b-2 border-foreground px-3 sm:px-4">
        <Button variant="ghost" size="icon-sm" aria-label="노트북 목록" asChild>
          <Link to="/">
            <ArrowLeft />
          </Link>
        </Button>
        <Link to="/" className="hidden font-serif text-sm font-black sm:block">
          법령 <span className="text-seal">노트</span>
        </Link>
        <span className="hidden text-muted-foreground/50 sm:block">/</span>
        {notebook ? <TitleEditor notebook={notebook} /> : <Skeleton className="h-6 w-48" />}
        <HealthBadge className="ml-auto hidden sm:inline-flex" />
      </header>

      <nav className="flex shrink-0 border-b lg:hidden">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => setTab(t.id)}
            className={cn(
              'flex-1 py-2.5 font-serif text-sm font-semibold transition-colors',
              tab === t.id ? 'border-b-2 border-seal text-foreground' : 'text-muted-foreground',
            )}
          >
            {t.label}
          </button>
        ))}
      </nav>

      {notebook && (
        <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[20rem_1fr_20rem] xl:grid-cols-[23rem_1fr_21rem]">
          <Column active={tab === 'sources'} className="bg-paper lg:border-r">
            <SourcesPanel notebook={notebook} viewer={viewer} onCloseViewer={() => setViewer(null)} />
          </Column>
          <Column active={tab === 'chat'}>
            <ChatPanel notebook={notebook} onOpenHit={openHit} />
          </Column>
          <Column active={tab === 'notes'} className="bg-paper lg:border-l">
            <NotesPanel notebookId={notebook.id} onOpenHit={openHit} />
          </Column>
        </div>
      )}
    </div>
  )
}

function Column({ active, className, children }: { active: boolean; className?: string; children: React.ReactNode }) {
  return <section className={cn('min-h-0 min-w-0 lg:block', active ? 'block' : 'hidden', className)}>{children}</section>
}

function TitleEditor({ notebook }: { notebook: Notebook }) {
  const update = useUpdateNotebook(notebook.id)
  const [title, setTitle] = useState(notebook.title)
  const commit = () => {
    const next = title.trim()
    if (next && next !== notebook.title) update.mutate({ title: next })
    else setTitle(notebook.title)
  }
  return (
    <input
      value={title}
      onChange={(e) => setTitle(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => e.key === 'Enter' && e.currentTarget.blur()}
      aria-label="노트북 제목"
      className="min-w-0 flex-1 truncate rounded bg-transparent px-1.5 py-1 font-serif text-base font-bold outline-none hover:bg-muted focus:bg-card focus:ring-1 focus:ring-ring sm:max-w-md"
    />
  )
}
