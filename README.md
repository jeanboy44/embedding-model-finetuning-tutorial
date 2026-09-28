# SLM 파인튜닝 & RAG 교육용 저장소

임베딩 모델, 검색 증강 생성(RAG), CLI 도구, MCP 서버, 모니터링을 통해 현대적인 ML 실무를 배우는 교육용 저장소입니다.

## 포함 내용

- **Phase 1: 모델 개발** - 임베딩 방식 비교 (Base vs Fine-tuned vs LLM 쿼리 확장)
- **Phase 2: CLI & MCP** - 모델 접근 도구 개발
- **Phase 3: RAG 에이전트** - 검색과 생성 통합
- **Phase 4: 모니터링** - 시스템 품질 및 성능 추적

## 빠른 시작

### OS별 설치 가이드

- [macOS 설치 가이드](INSTALL_MAC.md)
- [Windows 설치 가이드](INSTALL_WINDOWS.md)

> Python venv 방식을 원하면 [`venv` 브랜치](https://github.com/jeanboy44/embedding-model-finetuning-tutorial/tree/venv)를 확인하세요.

### 임베딩 모델 다운로드

기본 모델은 한국어를 지원하는 [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)(MIT 라이선스)입니다.

```bash
uv run python scripts/download_model_hf.py      # HuggingFace Hub에서
uv run python scripts/download_model_gdrive.py  # 또는 Google Drive에서
```

### 실행 예제

```bash
# Phase 1: 모델 비교 실험
uv run python tutorials/phase1_model_dev/run_experiments.py

# Phase 2: CLI 도구
uv run python -m src.cli.cli_tool rag --query "질문"

# Phase 3: RAG 에이전트
uv run python tutorials/phase3_agent/run_agent.py

# Phase 4: 모니터링
uv run python tutorials/phase4_monitoring/run_monitoring.py
```

## 강의 계획

강의 흐름(DS 본업 → API → 배포 최적화 → CLI·프론트엔드)은 [강의 계획](docs/PLAN.md)을 참고하세요.

## 아키텍처

시스템 설계는 [아키텍처 문서](docs/ARCHITECTURE.md)를 참고하세요.

## 설계 결정 기록

[ADR (Architecture Decision Records)](docs/adr/)에서 주요 결정사항을 확인하세요.

## 강사용

`/tutorials` 폴더의 자료를 통해 학생들을 각 Phase별로 가이드할 수 있습니다.
