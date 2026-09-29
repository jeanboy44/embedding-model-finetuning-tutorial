// 답변 속 인용 표기 [1], [1, 3], [2][3] 을 markdown 링크(#cite-n)로 바꿔 칩으로 그린다.
// n은 그 답의 근거 조문 목록(hits)의 순번(1부터)이다. 범위를 벗어난 숫자는 그대로 둔다.

const CITATION = /\[(\d+(?:\s*[,，]\s*\d+)*)\]/g

export function linkCitations(text: string, count: number): string {
  return text.replace(CITATION, (whole, group: string) => {
    const numbers = group.split(/[,，]/).map((s) => Number(s.trim()))
    if (numbers.some((n) => n < 1 || n > count)) return whole
    return numbers.map((n) => `[${n}](#cite-${n})`).join('')
  })
}

/** 답에서 실제로 인용한 순번들. */
export function citedNumbers(text: string): Set<number> {
  const cited = new Set<number>()
  for (const match of text.matchAll(CITATION)) {
    for (const s of match[1].split(/[,，]/)) cited.add(Number(s.trim()))
  }
  return cited
}

export function citationIndex(href: string | undefined): number | null {
  const match = href?.match(/^#cite-(\d+)$/)
  return match ? Number(match[1]) : null
}
