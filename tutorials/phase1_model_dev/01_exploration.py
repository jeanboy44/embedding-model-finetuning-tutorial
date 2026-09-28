"""
Phase 1-1: 임베딩 모델 탐색
=============================

학습 목표:
- 임베딩이 무엇인지 이해한다
- 문장을 벡터로 변환하는 과정을 실습한다
- 코사인 유사도로 문장 간 유사성을 측정한다
- 임베딩 공간을 시각화하고 탐색한다

실행:
    uv run python tutorials/phase1_model_dev/01_exploration.py
"""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from src.config import get_settings
from src.embeddings import create_embedding_fn

# 타입 별칭
EmbedFn = Callable[[list[str]], np.ndarray]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """두 벡터의 코사인 유사도를 계산한다.

    Args:
        a: 첫 번째 벡터.
        b: 두 번째 벡터.

    Returns:
        코사인 유사도 (-1 ~ 1).
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ============================================================
# 1단계: 임베딩이란?
# ============================================================
# 임베딩은 텍스트를 고정 길이의 숫자 벡터로 변환하는 것이다.
# 의미가 비슷한 문장은 벡터 공간에서 가까이 위치한다.
#
#   "고양이가 소파에 앉아있다"  →  [0.12, -0.34, 0.56, ...]
#   "강아지가 침대에 누워있다"  →  [0.11, -0.31, 0.55, ...]  ← 유사!
#   "주식 시장이 폭락했다"     →  [-0.78, 0.22, -0.15, ...] ← 다름!


def step1_generate_embeddings(embed: EmbedFn):
    """임베딩을 생성하고 기본 속성을 확인한다."""
    print("=" * 60)
    print("1단계: 임베딩 생성")
    print("=" * 60)

    settings = get_settings()

    sentences = [
        "머신러닝은 데이터에서 패턴을 학습한다",
        "딥러닝은 신경망을 사용하는 머신러닝이다",
        "오늘 날씨가 매우 좋다",
    ]

    embeddings = embed(sentences)

    print(f"\n모델: {settings.embedding_model_name}")
    print(f"문장 수: {len(sentences)}")
    print(f"임베딩 shape: {embeddings.shape}")
    print(f"임베딩 차원: {embeddings.shape[1]}")
    print("\n첫 번째 문장의 임베딩 (처음 10차원):")
    print(f"  {embeddings[0][:10]}")
    print(f"\n벡터 크기 (L2 norm): {np.linalg.norm(embeddings[0]):.4f}")

    return embeddings, sentences


# ============================================================
# 2단계: 코사인 유사도
# ============================================================


def step2_similarity(embeddings: np.ndarray, sentences: list[str]) -> None:
    """문장 쌍별 코사인 유사도를 계산한다."""
    print("\n" + "=" * 60)
    print("2단계: 코사인 유사도")
    print("=" * 60)

    print("""
두 벡터의 방향이 얼마나 비슷한지를 -1 ~ 1 사이 값으로 측정한다.
  1에 가까울수록: 의미가 유사
  0에 가까울수록: 관련 없음
""")

    print("문장 쌍별 유사도:")
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            bar = "█" * int(sim * 20) + "░" * (20 - int(sim * 20))
            print(f"\n  [{sim:.4f}] {bar}")
            print(f"    A: '{sentences[i]}'")
            print(f"    B: '{sentences[j]}'")

    print("""
참고: multilingual-e5-small은 다국어 범용 모델이라 문장 간 유사도가
전반적으로 높게(0.7 이상) 나오는 경향이 있습니다. 도메인 특화 파인튜닝을 하면
유사/비유사 문장의 점수 차이가 더 벌어집니다.
""")


# ============================================================
# 3단계: 유사도 히트맵 (상대 스케일)
# ============================================================


def step3_similarity_matrix(embed: EmbedFn) -> None:
    """샘플 데이터셋으로 유사도 히트맵을 생성한다."""
    print("=" * 60)
    print("3단계: 유사도 히트맵")
    print("=" * 60)

    data_path = Path("data/sample_docs.json")
    with data_path.open() as f:
        docs = json.load(f)

    selected = []
    seen: set[str] = set()
    for doc in docs:
        if doc["category"] not in seen:
            selected.append(doc)
            seen.add(doc["category"])
        if len(selected) >= 7:
            break

    titles = [d["title"][:6] for d in selected]
    embeddings = embed([d["text"] for d in selected])

    # 유사도 매트릭스 계산
    n = len(selected)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i][j] = cosine_similarity(embeddings[i], embeddings[j])

    # 대각선 제외한 min/max로 상대 스케일 계산
    off_diag = [sim_matrix[i][j] for i in range(n) for j in range(n) if i != j]
    sim_min, sim_max = min(off_diag), max(off_diag)

    heat = " ░▒▓█"

    print(f"\n유사도 히트맵 (상대 스케일: {sim_min:.2f}~{sim_max:.2f}):\n")
    print(f"{'':>8}  " + "  ".join(f"{t:>6}" for t in titles))

    for i, title_i in enumerate(titles):
        row = f"{title_i:>8}  "
        for j in range(n):
            if i == j:
                row += f"  {'━━━━'}  "
            else:
                # 상대 스케일로 정규화
                normalized = (sim_matrix[i][j] - sim_min) / (sim_max - sim_min + 1e-10)
                level = min(int(normalized * 5), 4)
                char = heat[level]
                row += f"  {char * 4}  "
        print(row)

    print(f"\n범례: {'░':>4}=낮음  {'▒':>4}=보통  {'▓':>4}=높음  {'█':>4}=매우 높음")
    print(f"       (범위: {sim_min:.3f} ~ {sim_max:.3f})")


# ============================================================
# 4단계: ASCII 산점도 — 임베딩 공간 시각화
# ============================================================


def step4_ascii_scatter(embed: EmbedFn) -> None:
    """PCA로 2D 축소 후 터미널에 산점도를 그린다."""
    print("\n" + "=" * 60)
    print("4단계: 임베딩 공간 산점도")
    print("=" * 60)

    data_path = Path("data/sample_docs.json")
    with data_path.open() as f:
        docs = json.load(f)

    embeddings = embed([d["text"] for d in docs])

    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)

    print(f"\n{embeddings.shape[1]}차원 → 2차원 (PCA)")
    print(f"설명된 분산: {pca.explained_variance_ratio_.sum():.1%}\n")

    cat_symbols = {}
    all_symbols = "●■▲◆★○□"
    for i, cat in enumerate(sorted({d["category"] for d in docs})):
        cat_symbols[cat] = all_symbols[i % len(all_symbols)]

    W, H = 70, 25
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_pad = (x_max - x_min) * 0.1 or 1
    y_pad = (y_max - y_min) * 0.1 or 1

    grid: list[list[str]] = [[" " for _ in range(W)] for _ in range(H)]

    for i, doc in enumerate(docs):
        gx = int((coords[i, 0] - x_min + x_pad / 2) / (x_max - x_min + x_pad) * (W - 1))
        gy = int((coords[i, 1] - y_min + y_pad / 2) / (y_max - y_min + y_pad) * (H - 1))
        gy = H - 1 - gy
        gx = max(0, min(W - 1, gx))
        gy = max(0, min(H - 1, gy))
        grid[gy][gx] = cat_symbols[doc["category"]]

    print("┌" + "─" * W + "┐")
    for row in grid:
        print("│" + "".join(row) + "│")
    print("└" + "─" * W + "┘")

    print("\n범례:")
    for cat, sym in sorted(cat_symbols.items()):
        count = sum(1 for d in docs if d["category"] == cat)
        print(f"  {sym} {cat} ({count}개)")

    print("\n같은 카테고리(같은 기호)가 모여있는지 확인해보세요!")


# ============================================================
# 5단계: 끼어든 놈 찾기 (Odd One Out)
# ============================================================


def step5_odd_one_out(embed: EmbedFn) -> None:
    """4개 문장 중 이질적인 문장을 임베딩으로 찾는 게임."""
    print("\n" + "=" * 60)
    print("5단계: 끼어든 놈 찾기 (Odd One Out)")
    print("=" * 60)
    print("\n4개 문장 중 나머지와 가장 다른 1개를 임베딩이 찾을 수 있을까?\n")

    quizzes = [
        {
            "sentences": [
                "신경망은 여러 층의 뉴런으로 구성된다",
                "역전파 알고리즘으로 가중치를 업데이트한다",
                "드롭아웃은 과적합을 방지하는 정규화 기법이다",
                "오늘 점심은 김치찌개를 먹었다",
            ],
            "answer": 3,
            "explanation": "딥러닝 vs 일상",
        },
        {
            "sentences": [
                "벡터 검색으로 유사한 문서를 찾는다",
                "코사인 유사도로 관련성을 측정한다",
                "FAISS는 대규모 벡터 인덱싱 라이브러리이다",
                "강화학습 에이전트가 보상을 최대화한다",
            ],
            "answer": 3,
            "explanation": "검색/임베딩 vs 강화학습",
        },
        {
            "sentences": [
                "파이썬은 데이터 과학에 널리 사용된다",
                "R 언어는 통계 분석에 특화되어 있다",
                "Julia는 수치 연산에 최적화된 언어이다",
                "트랜스포머는 어텐션 메커니즘을 사용한다",
            ],
            "answer": 3,
            "explanation": "프로그래밍 언어 vs 딥러닝 아키텍처",
        },
    ]

    correct = 0
    for i, quiz in enumerate(quizzes, 1):
        sents = quiz["sentences"]
        embeddings = embed(sents)

        avg_sims = []
        for j in range(len(sents)):
            sims = [
                cosine_similarity(embeddings[j], embeddings[k])
                for k in range(len(sents))
                if k != j
            ]
            avg_sims.append(sum(sims) / len(sims))

        predicted = int(np.argmin(avg_sims))

        print(f"퀴즈 {i}: 누가 끼어들었을까?")
        for j, sent in enumerate(sents):
            bar_len = int(avg_sims[j] * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            marker = " ← 이질!" if j == predicted else ""
            print(f"  {j + 1}) [{avg_sims[j]:.3f}] {bar} {sent[:35]}...{marker}")

        is_correct = predicted == quiz["answer"]
        correct += is_correct
        status = "정답!" if is_correct else f"오답 (정답: {quiz['answer'] + 1}번)"
        print(f"\n  임베딩 판단: {predicted + 1}번 → {status}")
        print(f"  해석: {quiz['explanation']}\n")

    print(f"결과: {correct}/{len(quizzes)} 정답")


# ============================================================
# 6단계: 벡터 산술 — 의미의 덧셈과 뺄셈
# ============================================================


def step6_vector_arithmetic(embed: EmbedFn) -> None:
    """임베딩 벡터 산술로 의미 관계를 탐색한다."""
    print("\n" + "=" * 60)
    print("6단계: 벡터 산술 — 의미의 덧셈과 뺄셈")
    print("=" * 60)
    print("\n'A - B + C = ?'  벡터 연산으로 의미를 조합할 수 있을까?\n")

    candidates = [
        "지도학습은 레이블 데이터로 모델을 학습시킨다",
        "비지도학습은 레이블 없이 패턴을 찾는다",
        "강화학습은 보상을 통해 에이전트를 학습시킨다",
        "임베딩은 텍스트를 벡터로 변환한다",
        "트랜스포머는 어텐션 기반 아키텍처이다",
        "CNN은 이미지 인식에 주로 사용된다",
        "RNN은 순차 데이터 처리에 적합하다",
        "자연어 처리는 텍스트 데이터를 다루는 분야이다",
        "컴퓨터 비전은 이미지 데이터를 다루는 분야이다",
        "모델 모니터링은 성능 저하를 감지한다",
        "데이터 드리프트는 입력 분포의 변화를 의미한다",
        "하이퍼파라미터 튜닝으로 모델 성능을 최적화한다",
    ]
    cand_embs = embed(candidates)

    experiments = [
        {
            "A": "자연어 처리는 텍스트 데이터를 다루는 분야이다",
            "B": "텍스트",
            "C": "이미지",
            "label": "NLP - 텍스트 + 이미지 = ?",
            "expect": "컴퓨터 비전",
        },
        {
            "A": "지도학습은 레이블 데이터로 모델을 학습시킨다",
            "B": "레이블 데이터",
            "C": "보상 신호",
            "label": "지도학습 - 레이블 + 보상 = ?",
            "expect": "강화학습",
        },
        {
            "A": "CNN은 이미지 인식에 주로 사용된다",
            "B": "이미지",
            "C": "순차 데이터",
            "label": "CNN - 이미지 + 순차데이터 = ?",
            "expect": "RNN",
        },
    ]

    for exp in experiments:
        a_emb = embed([exp["A"]])[0]
        b_emb = embed([exp["B"]])[0]
        c_emb = embed([exp["C"]])[0]

        result_vec = a_emb - b_emb + c_emb

        sims = [cosine_similarity(result_vec, ce) for ce in cand_embs]
        top3_idx = np.argsort(sims)[::-1][:3]

        print(f"  {exp['label']}")
        print(f"  기대: {exp['expect']}")
        print("  결과:")
        for rank, idx in enumerate(top3_idx, 1):
            marker = " ✓" if exp["expect"] in candidates[idx] else ""
            print(f"    {rank}. [{sims[idx]:.3f}] {candidates[idx][:45]}{marker}")
        print()

    print("참고: 문장 임베딩에서의 벡터 산술은 단어 임베딩(Word2Vec)보다")
    print(
        "정확도가 떨어질 수 있습니다. 문장은 여러 의미가 복합적으로 담겨 있기 때문입니다."
    )


# ============================================================
# 7단계: 자동 클러스터링
# ============================================================


def step7_auto_clustering(embed: EmbedFn) -> None:
    """K-Means로 임베딩을 자동 군집화하고 카테고리와 비교한다."""
    print("\n" + "=" * 60)
    print("7단계: 자동 클러스터링")
    print("=" * 60)
    print("\n임베딩에 K-Means를 적용하면 카테고리를 알려주지 않아도")
    print("비슷한 문서끼리 자동으로 묶일까?\n")

    data_path = Path("data/sample_docs.json")
    with data_path.open() as f:
        docs = json.load(f)

    embeddings = embed([d["text"] for d in docs])

    from sklearn.cluster import KMeans

    n_clusters = len({d["category"] for d in docs})
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    cluster_markers = "①②③④⑤⑥⑦⑧⑨⑩"
    for cluster_id in range(n_clusters):
        members = [docs[i] for i in range(len(docs)) if labels[i] == cluster_id]
        cats = [m["category"] for m in members]
        dominant = max(set(cats), key=cats.count)

        marker = (
            cluster_markers[cluster_id]
            if cluster_id < len(cluster_markers)
            else f"C{cluster_id}"
        )

        print(f"  클러스터 {marker}  (주요: {dominant})")
        for doc in members:
            match = "✓" if doc["category"] == dominant else "✗"
            print(f"    {match} [{doc['category']:<14}] {doc['title']}")
        print()

    correct = 0
    for cluster_id in range(n_clusters):
        members = [docs[i] for i in range(len(docs)) if labels[i] == cluster_id]
        cats = [m["category"] for m in members]
        dominant = max(set(cats), key=cats.count)
        correct += sum(1 for c in cats if c == dominant)

    accuracy = correct / len(docs)
    print(f"클러스터링 정확도 (majority voting): {accuracy:.0%}")
    print(f"카테고리 수: {n_clusters}, 문서 수: {len(docs)}")
    print("\n임베딩이 레이블 없이도 의미적 그룹을 형성하는 것을 확인!")


# ============================================================
# 실행
# ============================================================


def main() -> None:
    """모든 단계를 순차 실행한다."""
    # 모델을 1회만 로드하여 모든 step에서 공유
    settings = get_settings()
    embed = create_embedding_fn(settings.embedding_model_name)

    embeddings, sentences = step1_generate_embeddings(embed)
    step2_similarity(embeddings, sentences)
    step3_similarity_matrix(embed)
    step4_ascii_scatter(embed)
    step5_odd_one_out(embed)
    step6_vector_arithmetic(embed)
    step7_auto_clustering(embed)

    print("\n" + "=" * 60)
    print("실습 과제")
    print("=" * 60)
    print("""
1. step5에 자신만의 퀴즈를 추가해보세요 — 임베딩이 맞출 수 있을까?
2. step6에서 자신만의 벡터 산술 실험을 만들어보세요.
3. 한국어 문장과 영어 문장의 유사도는 어떤가요? 왜 그럴까요?
4. data/sample_docs.json에 새로운 카테고리를 추가하고
   클러스터링 정확도가 어떻게 변하는지 확인해보세요.
5. (심화) PCA 대신 t-SNE를 사용하면 산점도가 어떻게 달라질까요?
""")


if __name__ == "__main__":
    main()
