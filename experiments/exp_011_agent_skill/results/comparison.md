# Claude Code + korean-law-search 스킬: 학습 전 vs 파인튜닝

test situation 질문 10개 (seed 7), claude -p --model sonnet. 정답 조문 인용 = 답에 정답 조(조 단위 id)를 근거로 적었는가.

| 모델 | 정답 조문 인용 | 평균 CLI 호출 | 평균 시간 | 질문당 비용 |
|---|---|---|---|---|
| 학습 전 e5 | 6/10 | 3.9 | 34초 | $0.19 |
| 파인튜닝 e5 (r001_A) | 8/10 | 3.5 | 43초 | $0.16 |
