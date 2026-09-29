import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Hit } from '@/api/types'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { citationIndex, linkCitations } from '@/lib/citations'
import { hitTitle } from '@/lib/format'
import { cn } from '@/lib/utils'

/** 인주색 도장 모양의 인용 번호. 누르면 조문 보기가 열린다. */
export function CitationChip({
  n,
  hit,
  onOpen,
  className,
}: {
  n: number
  hit?: Hit
  onOpen?: (hit: Hit) => void
  className?: string
}) {
  const chip = (
    <button
      type="button"
      onClick={() => hit && onOpen?.(hit)}
      aria-label={hit ? `인용 ${n}: ${hitTitle(hit)}` : `인용 ${n}`}
      className={cn(
        'mx-0.5 inline-flex h-[1.35em] min-w-[1.35em] -translate-y-[0.1em] items-center justify-center rounded-[3px]',
        'border border-seal/70 bg-seal/8 px-1 align-middle font-mono text-[0.7em] font-medium text-seal',
        'transition-colors hover:bg-seal hover:text-seal-foreground focus-visible:bg-seal focus-visible:text-seal-foreground',
        className,
      )}
    >
      {n}
    </button>
  )
  if (!hit) return chip
  return (
    <Tooltip>
      <TooltipTrigger asChild>{chip}</TooltipTrigger>
      <TooltipContent side="top" className="max-w-xs">
        <p className="font-serif font-semibold">{hitTitle(hit)}</p>
        <p className="mt-1 line-clamp-3 opacity-80">{hit.text}</p>
      </TooltipContent>
    </Tooltip>
  )
}

/** 답변 markdown. 본문 속 [n]은 인용 칩으로 그린다. */
export function Answer({
  text,
  hits,
  onOpen,
  streaming = false,
}: {
  text: string
  hits: Hit[]
  onOpen: (hit: Hit) => void
  streaming?: boolean
}) {
  return (
    <div className={cn('answer', streaming && 'streaming')}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children }) => {
            const n = citationIndex(href)
            if (n !== null) return <CitationChip n={n} hit={hits[n - 1]} onOpen={onOpen} />
            return (
              <a href={href} target="_blank" rel="noreferrer" className="text-seal underline underline-offset-2">
                {children}
              </a>
            )
          },
        }}
      >
        {linkCitations(text, hits.length)}
      </ReactMarkdown>
    </div>
  )
}
