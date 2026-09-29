import type { Hit } from '@/api/types'

export const THEME_LABELS: Record<string, string> = {
  youth: '청년·근로',
  traffic: '교통',
  tax: '세금',
  finance: '금융',
  consumer: '소비자',
  electric: '전기',
}

export const themeLabel = (theme: string | null | undefined) =>
  (theme && THEME_LABELS[theme]) || theme || '기타'

/** "근로기준법 제56조 (연장·야간 및 휴일 근로) 제1항" 같은 조문 제목. */
export const hitTitle = (hit: Hit) => hit.title || hit.id

export const articleKey = (hit: Hit) => hit.parent_id || hit.id

export function relativeTime(iso: string): string {
  const diff = (Date.now() - new Date(iso).getTime()) / 1000
  if (diff < 60) return '방금'
  if (diff < 3600) return `${Math.floor(diff / 60)}분 전`
  if (diff < 86400) return `${Math.floor(diff / 3600)}시간 전`
  if (diff < 86400 * 7) return `${Math.floor(diff / 86400)}일 전`
  return new Date(iso).toLocaleDateString('ko-KR')
}
