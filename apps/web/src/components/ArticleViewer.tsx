import { ArrowLeft, ExternalLink } from 'lucide-react'
import { useEffect, useRef } from 'react'
import { useArticle } from '@/api/hooks'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Skeleton } from '@/components/ui/skeleton'
import { themeLabel } from '@/lib/format'
import { cn } from '@/lib/utils'

/** 인용한 조문을 같은 조 전체로 펼쳐 보이고, 인용된 조각을 형광펜처럼 강조한다. */
export function ArticleViewer({
  parentId,
  highlightId,
  onClose,
}: {
  parentId: string
  highlightId: string
  onClose: () => void
}) {
  const { data: pieces, isLoading, error } = useArticle(parentId)
  const highlighted = useRef<HTMLDivElement>(null)

  // 조각이 여럿일 때만 인용한 조각으로 스크롤·강조한다 (하나뿐이면 제목부터 보이게)
  const split = (pieces?.length ?? 0) > 1
  useEffect(() => {
    if (split) highlighted.current?.scrollIntoView({ block: 'start', behavior: 'smooth' })
  }, [split, pieces, highlightId])

  const first = pieces?.[0]
  return (
    <div className="flex h-full min-h-0 flex-col animate-rise">
      <div className="flex items-center gap-1 border-b px-2 py-2">
        <Button variant="ghost" size="sm" onClick={onClose}>
          <ArrowLeft /> 소스 목록
        </Button>
        {first?.source_url && (
          <Button variant="ghost" size="sm" className="ml-auto text-muted-foreground" asChild>
            <a href={first.source_url} target="_blank" rel="noreferrer">
              법제처 <ExternalLink />
            </a>
          </Button>
        )}
      </div>
      <ScrollArea className="min-h-0 flex-1">
        <article className="px-5 py-5">
          {isLoading && (
            <div className="space-y-3">
              <Skeleton className="h-4 w-1/3" />
              <Skeleton className="h-7 w-3/4" />
              <Skeleton className="h-32 w-full" />
            </div>
          )}
          {error && <p className="text-sm text-seal">{error.message}</p>}
          {first && (
            <header className="mb-5">
              <p className="text-xs tracking-wide text-muted-foreground">
                {themeLabel(first.theme)} · {first.law_type}
                {first.chapter && ` · ${first.chapter}`}
              </p>
              <h2 className="mt-1.5 font-serif text-xl leading-snug font-bold text-balance">
                {first.law_name} {first.article_no}
                {first.article_title && (
                  <span className="font-semibold text-ink-soft"> ({first.article_title})</span>
                )}
              </h2>
            </header>
          )}
          <div className="space-y-3">
            {pieces?.map((piece) => {
              const active = split && piece.id === highlightId
              return (
                <div
                  key={piece.id}
                  ref={active ? highlighted : undefined}
                  className={cn(
                    'relative rounded-md px-3 py-2 transition-colors',
                    active && 'bg-[oklch(0.93_0.07_90)] ring-1 ring-[oklch(0.8_0.1_85)]',
                  )}
                >
                  {active && (
                    <span className="absolute top-2 -left-1.5 h-[calc(100%-1rem)] w-[3px] rounded-full bg-seal" />
                  )}
                  {split && (
                    <p className="mb-1 font-mono text-[0.7rem] text-muted-foreground">{piece.title}</p>
                  )}
                  <p className="statute">{piece.text}</p>
                </div>
              )
            })}
          </div>
        </article>
      </ScrollArea>
    </div>
  )
}
