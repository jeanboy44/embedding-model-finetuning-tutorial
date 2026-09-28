import { BookOpen, Plus, X } from 'lucide-react'
import { useState } from 'react'
import { useLaws, useUpdateNotebook } from '@/api/hooks'
import type { Notebook } from '@/api/types'
import { AddSourcesDialog } from '@/components/AddSourcesDialog'
import { ArticleViewer } from '@/components/ArticleViewer'
import { PanelHeader } from '@/components/PanelHeader'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { themeLabel } from '@/lib/format'

export interface ViewerTarget {
  parentId: string
  highlightId: string
}

export function SourcesPanel({
  notebook,
  viewer,
  onCloseViewer,
}: {
  notebook: Notebook
  viewer: ViewerTarget | null
  onCloseViewer: () => void
}) {
  const [adding, setAdding] = useState(false)
  const { data: laws = [] } = useLaws()
  const update = useUpdateNotebook(notebook.id)
  const byName = new Map(laws.map((l) => [l.law_name, l]))
  const total = notebook.laws.reduce((sum, name) => sum + (byName.get(name)?.doc_count ?? 0), 0)

  if (viewer) {
    return <ArticleViewer key={viewer.parentId} {...viewer} onClose={onCloseViewer} />
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <PanelHeader title="소스" hint={notebook.laws.length ? `법령 ${notebook.laws.length} · 조각 ${total.toLocaleString()}` : undefined}>
        <Button variant="outline" size="sm" onClick={() => setAdding(true)}>
          <Plus /> 추가
        </Button>
      </PanelHeader>
      <ScrollArea className="min-h-0 flex-1">
        {notebook.laws.length === 0 ? (
          <div className="px-5 py-10 text-center">
            <BookOpen className="mx-auto size-8 text-muted-foreground/60" strokeWidth={1.25} />
            <p className="mt-3 font-serif font-semibold">아직 고른 법령이 없습니다</p>
            <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
              지금은 법령 {laws.length || ''}개 전체에서 찾습니다.
              <br />
              법령을 고르면 그 안에서만 답합니다.
            </p>
            <Button className="mt-4" size="sm" onClick={() => setAdding(true)}>
              <Plus /> 법령 고르기
            </Button>
          </div>
        ) : (
          <ul className="space-y-1 p-2">
            {notebook.laws.map((name, i) => {
              const law = byName.get(name)
              return (
                <li
                  key={name}
                  style={{ animationDelay: `${i * 30}ms` }}
                  className="group flex animate-rise items-center gap-3 rounded-md px-3 py-2 hover:bg-muted/70"
                >
                  <span className="grid size-7 shrink-0 place-items-center rounded-[4px] border border-foreground/15 bg-card font-serif text-xs font-bold">
                    {law?.law_type === '법률' ? '法' : law?.law_type === '시행령' ? '令' : '規'}
                  </span>
                  <div className="min-w-0 flex-1">
                    <p className="truncate font-serif text-[0.92rem] font-semibold">{name}</p>
                    <p className="text-xs text-muted-foreground">
                      {themeLabel(law?.theme)} · 조각 {law?.doc_count.toLocaleString() ?? '?'}
                    </p>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon-xs"
                    aria-label={`${name} 빼기`}
                    className="lg:opacity-0 lg:group-hover:opacity-100 lg:focus-visible:opacity-100"
                    onClick={() => update.mutate({ laws: notebook.laws.filter((l) => l !== name) })}
                  >
                    <X />
                  </Button>
                </li>
              )
            })}
          </ul>
        )}
      </ScrollArea>
      <AddSourcesDialog
        open={adding}
        onOpenChange={setAdding}
        selected={notebook.laws}
        saving={update.isPending}
        onSave={(next) => update.mutate({ laws: next }, { onSuccess: () => setAdding(false) })}
      />
    </div>
  )
}
