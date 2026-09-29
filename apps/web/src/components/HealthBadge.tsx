import { useHealth } from '@/api/hooks'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'

/** API·모델 상태: 초록 = 검색+답변, 노랑 = 검색만(키 없음), 빨강 = 서버 연결 안 됨. */
export function HealthBadge({ className }: { className?: string }) {
  const { data, isError, isLoading } = useHealth()
  const tone = isError ? 'bg-seal' : !data ? 'bg-muted-foreground/40' : data.llm_available ? 'bg-emerald-600' : 'bg-amber-500'
  const label = isError
    ? 'API 연결 안 됨'
    : isLoading || !data
      ? '연결 중'
      : data.llm_available
        ? data.model_key
        : `${data.model_key} · 검색만`
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className={cn('inline-flex items-center gap-1.5 rounded-full border bg-card px-2.5 py-1 font-mono text-[0.68rem]', className)}>
          <span className={cn('size-1.5 rounded-full', tone)} />
          {label}
        </span>
      </TooltipTrigger>
      <TooltipContent>
        {isError && 'ragkit-api 서버를 띄우세요: uv run --package ragkit-api ragkit-api'}
        {data && `임베딩 ${data.model_key} · 조문 조각 ${data.doc_count.toLocaleString()}개`}
        {data && !data.llm_available && ' · GEMINI_API_KEY가 없어 답변 없이 검색 결과만 보여 줍니다'}
      </TooltipContent>
    </Tooltip>
  )
}
