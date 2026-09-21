# macOS 설치 가이드 (uv)

macOS에서 `uv`를 사용하여 개발 환경을 설정합니다.

> **참고**: Python venv 방식을 원하면 [`venv` 브랜치](https://github.com/your-repo/tree/venv)를 확인하세요.

---

## 1단계: uv 설치

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

설치 확인:
```bash
uv --version
```

---

## 2단계: 프로젝트 설정

```bash
git clone https://github.com/your-repo/slm-finetuning-example.git
cd slm-finetuning-example
uv sync
```

---

## 3단계: 실행 확인

```bash
uv run python --version
```

---

## Phase별 실행

### Phase 1: 모델 비교 실험

```bash
uv run python tutorials/phase1_model_dev/run_experiments.py
```

### Phase 2: CLI 도구

```bash
uv run python -m src.cli.cli_tool rag --query "질문"
```

### Phase 3: RAG 에이전트

```bash
uv run python tutorials/phase3_agent/run_agent.py
```

### Phase 4: 모니터링

```bash
uv run python tutorials/phase4_monitoring/run_monitoring.py
```

---

## 자주 사용하는 명령어

```bash
# 패키지 추가
uv add package-name

# 테스트 실행
uv run pytest tests/ -v

# 코드 검사
uv run ruff check src/
uv run ruff format src/

# Python 버전 고정
uv python pin 3.11
```

---

## 문제 해결

### "uv 명령어를 찾을 수 없음"

```bash
# 터미널 재시작 후 다시 시도
uv --version

# 경로 확인
which uv
```

### 의존성 설치 실패

```bash
uv cache clean
uv sync
```

---

## 기업 환경 (SSL 프록시)

```bash
mkdir -p ~/.config/uv
cat > ~/.config/uv/uv.toml << EOF
native-tls = true
EOF

uv sync
```

---

## Windows 사용자?

👉 **[Windows 설치 가이드](INSTALL_WINDOWS.md)**
