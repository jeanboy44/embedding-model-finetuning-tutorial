import type { ReactNode } from 'react'

export function PanelHeader({ title, hint, children }: { title: string; hint?: string; children?: ReactNode }) {
  return (
    <div className="flex h-12 shrink-0 items-center gap-2 border-b px-4">
      <h2 className="font-serif text-[0.95rem] font-bold tracking-wide">{title}</h2>
      {hint && <span className="truncate text-xs text-muted-foreground">{hint}</span>}
      <div className="ml-auto flex items-center gap-1">{children}</div>
    </div>
  )
}
