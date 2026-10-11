# 004 context_header — 2. 실험 계획

## 가설

**코퍼스 25,967개 조문에 `[{law_name} > {chapter}] {title}\n{text}` 형태의 계층적 맥락 헤더를 결합하여 인덱싱하면, 모델 학습 없이도 1위 오답의 51%인 타 법령 혼동이 줄어들어:**
1. **학습 전 베이스 모델(000)**의 dev R@5가 기존 0.5398 대비 유의하게 상승한다.
2. **현재 챔피언 모델(002)**의 dev R@5가 대조군(0.7534) 대비 **0.010 이상 상승(목표 0.765+)**한다.

---

## 제약 및 통제 변수

| 항목 | 고정한 것 | 비고 |
|---|---|---|
| 모델 가중치 | 모델 학습 없음 | 000(base e5), 002(hard_negatives 챔피언) 불변 |
| 평가 셋 | `data/splits_article/dev.jsonl` (1,243문항) | 불변 |
| 코퍼스 문서 집합 | 25,967개 조문 전수 유지 (id 불변) | 행 수 및 id 순서 일치 |
| 바꾸는 것 | **조문 텍스트의 맥락 헤더 부착 여부** | `title\ntext` → `[{law_name} > {chapter}] {title}\n{text}` |

---

## 헤더 생성 규칙

- `chapter`가 있는 경우 (85.0%):
  `[{law_name} > {chapter}] {title}\n{text}`
- `chapter`가 없는 경우 (15.0%):
  `[{law_name}] {title}\n{text}`

---

## run 구성 (총 2개 평가)

| run | 평가 대상 모델 | 코퍼스 | 기대 역할 |
|---|---|---|---|
| `base_header` | `intfloat/multilingual-e5-small` | `law_docs_header.json` | 헤더 자체의 순수 베이스라인 상승분 측정 (대조군: 000의 0.5398) |
| `002_header` | `models/finetuned/002_hard_negatives` | `law_docs_header.json` | 파인튜닝 모델과 헤더의 결합 효과 측정 (대조군: 002의 0.7534) |

---

## 합격 판정 기준

1. **002_header dev R@5 $\ge 0.7634$** ($\Delta \ge +0.010$ vs 002 원본 대조군 0.7534)
2. **002_header dev R@5 > base_header dev R@5**: 파인튜닝 효과가 여전히 우위를 유지해야 함.
3. **대응 검정 95% CI 하한 $> 0$**

---

## 누수 점검

- L1: `law_docs.json` 내부의 기존 필드(`law_name`, `chapter`, `title`, `text`)만 조합하며, 일체의 질문이나 외부 데이터 유입 없음.
- L2: 문서 수(25,967개) 및 고유 ID 완전 일치 확인.
- L3: dev/test 질문과의 매칭 무결성 확인.
