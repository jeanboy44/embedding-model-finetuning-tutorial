# ragkit-api 배포 이미지. 저장소 루트에서 빌드한다(uv workspace 전체가 필요).
#
#   ONNX(기본, torch 없음):  docker build -f apps/api/Dockerfile -t ragkit-api:onnx .
#   torch 백엔드:            docker build -f apps/api/Dockerfile --build-arg BACKEND=torch -t ragkit-api:torch .
#
# 모델·인덱스는 이미지에 넣지 않고 볼륨으로 연결한다(이미지 크기 = 실행 환경 크기만 비교):
#   docker run --rm -p 8000:8000 -v "$PWD/models:/app/models" -v "$PWD/data:/app/data" \
#     -e GEMINI_API_KEY ragkit-api:onnx
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.10 /uv /bin/uv

ARG BACKEND=onnx
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    RAGKIT_PROJECT_ROOT=/app \
    EMBEDDING_BACKEND=${BACKEND} \
    EMBEDDING_DEVICE=cpu \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY apps ./apps

# ragkit-api와 그 의존성만 설치한다. torch 이미지는 extra torch를 더한다.
RUN if [ "$BACKEND" = "torch" ]; then EXTRA="--extra torch"; else EXTRA=""; fi \
    && uv sync --frozen --no-dev --no-default-groups --package ragkit-api $EXTRA \
    && rm -rf /root/.cache

EXPOSE 8000
CMD ["ragkit-api", "--host", "0.0.0.0", "--port", "8000"]
