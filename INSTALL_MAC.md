# macOS 설치 가이드 (uv)

macOS에서 `uv`를 사용하여 개발 환경을 설정합니다.

> **참고**: Python venv 방식을 원하면 [`venv` 브랜치](https://github.com/jeanboy44/embedding-model-finetuning-tutorial/tree/venv)를 확인하세요.

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
git clone https://github.com/jeanboy44/embedding-model-finetuning-tutorial.git
cd embedding-model-finetuning-tutorial
uv sync
```

---

## 3단계: 임베딩 모델 다운로드

기본 임베딩 모델 [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)(약 470MB, 한국어 지원)을 `models/` 폴더에 받습니다. 둘 중 하나만 실행하면 됩니다.

```bash
# 방법 1: HuggingFace Hub에서 받기 (기본)
uv run python scripts/download_model_hf.py

# 방법 2: Google Drive에서 받기 (HuggingFace 접속이 막힌 환경)
uv run python scripts/download_model_gdrive.py
```

한 번 받아두면 이후에는 인터넷 없이도 모델을 불러옵니다. 다시 받으려면 `--force`를 붙이세요.

> 이 단계를 건너뛰어도 첫 실행 때 HuggingFace에서 자동으로 받아 캐시에 저장합니다.

---

## 4단계: 실행 확인

```bash
uv run python --version
```

---

실습별 실행 방법은 [README](README.md)를 보세요.

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
