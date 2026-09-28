import { Search } from 'lucide-react'
import { useMemo, useState } from 'react'
import { useLaws } from '@/api/hooks'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { ScrollArea } from '@/components/ui/scroll-area'
import { themeLabel } from '@/lib/format'
import { cn } from '@/lib/utils'

/** 노트북 소스(법령) 고르기: 테마 필터 + 이름 검색 + 여러 개 선택. */
export function AddSourcesDialog({
  open,
  onOpenChange,
  selected,
  onSave,
  saving,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  selected: string[]
  onSave: (laws: string[]) => void
  saving?: boolean
}) {
  const { data: laws = [] } = useLaws()
  const [picked, setPicked] = useState<Set<string>>(new Set(selected))
  const [theme, setTheme] = useState<string | null>(null)
  const [query, setQuery] = useState('')

  const themes = useMemo(() => [...new Set(laws.map((l) => l.theme))], [laws])
  const visible = laws.filter(
    (l) => (!theme || l.theme === theme) && l.law_name.replace(/\s/g, '').includes(query.replace(/\s/g, '')),
  )

  const toggle = (name: string) =>
    setPicked((prev) => {
      const next = new Set(prev)
      if (next.has(name)) next.delete(name)
      else next.add(name)
      return next
    })

  // 법률을 고르면 같은 이름의 시행령·시행규칙도 함께 고를 수 있게 묶음 선택을 둔다
  const family = (name: string) => {
    const names = [name, `${name} 시행령`, `${name} 시행규칙`]
    return laws.filter((l) => names.includes(l.law_name))
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        if (next) setPicked(new Set(selected))
        onOpenChange(next)
      }}
    >
      <DialogContent className="gap-0 overflow-hidden p-0 sm:max-w-2xl">
        <DialogHeader className="border-b px-6 pt-6 pb-4">
          <DialogTitle className="font-serif text-xl">소스 법령 고르기</DialogTitle>
          <DialogDescription>이 노트북의 질문은 고른 법령 안에서만 찾아 답합니다.</DialogDescription>
          <div className="relative mt-3">
            <Search className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              autoFocus
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="법령 이름으로 찾기 (예: 근로기준법)"
              className="pl-8"
            />
          </div>
          <div className="mt-3 flex flex-wrap gap-1.5">
            {[null, ...themes].map((t) => (
              <button
                key={t ?? 'all'}
                type="button"
                onClick={() => setTheme(t)}
                className={cn(
                  'rounded-full border px-3 py-1 text-xs transition-colors',
                  theme === t ? 'border-foreground bg-foreground text-background' : 'hover:bg-muted',
                )}
              >
                {t ? themeLabel(t) : '전체'}
              </button>
            ))}
          </div>
        </DialogHeader>
        <ScrollArea className="h-[min(26rem,55vh)]">
          <ul className="divide-y px-2">
            {visible.map((law) => {
              const checked = picked.has(law.law_name)
              const relatives = family(law.law_name)
              return (
                <li key={law.law_name} className="flex items-center gap-3 px-4 py-2.5">
                  <Checkbox
                    id={`law-${law.law_name}`}
                    checked={checked}
                    onCheckedChange={() => toggle(law.law_name)}
                  />
                  <label htmlFor={`law-${law.law_name}`} className="flex min-w-0 flex-1 cursor-pointer items-baseline gap-2">
                    <span className="truncate font-serif font-semibold">{law.law_name}</span>
                    <span className="shrink-0 text-xs text-muted-foreground">
                      {themeLabel(law.theme)} · 조각 {law.doc_count.toLocaleString()}
                    </span>
                  </label>
                  {law.law_type === '법률' && relatives.length > 1 && (
                    <Button
                      variant="ghost"
                      size="xs"
                      className="text-muted-foreground"
                      onClick={() =>
                        setPicked((prev) => new Set([...prev, ...relatives.map((r) => r.law_name)]))
                      }
                    >
                      +시행령·규칙
                    </Button>
                  )}
                </li>
              )
            })}
            {!visible.length && <li className="px-4 py-10 text-center text-sm text-muted-foreground">맞는 법령이 없습니다</li>}
          </ul>
        </ScrollArea>
        <DialogFooter className="m-0 flex-row items-center justify-between rounded-none border-t px-6 py-4">
          <p className="text-sm text-muted-foreground">
            {picked.size ? `${picked.size}개 선택` : '고르지 않으면 전체 법령에서 찾습니다'}
          </p>
          <div className="flex gap-2">
            {picked.size > 0 && (
              <Button variant="ghost" onClick={() => setPicked(new Set())}>
                모두 해제
              </Button>
            )}
            <Button onClick={() => onSave([...picked])} disabled={saving}>
              {saving ? '저장 중…' : '저장'}
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
