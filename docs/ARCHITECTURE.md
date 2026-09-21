# 시스템 아키텍처

## 데이터 흐름

```
사용자 질문
    ↓
[임베딩 생성기] → 질문을 임베딩으로 변환
    ↓
[검색기] → 유사한 문서 찾기
    ↓
[RAG 에이전트] → 프로세스 조율
    ↓
[Gemini 클라이언트] → 답변 생성
    ↓
답변 + 모니터링 로그
```

## 주요 컴포넌트

### src/models/
- embedding_loader.py - HuggingFace에서 모델 로드
- gemini_client.py - Gemini API 래퍼

### src/embeddings/
- generator.py - 텍스트 임베딩 생성

### src/retrieval/
- document_store.py - 문서와 임베딩 저장
- retriever.py - 유사도 기반 검색

### src/agents/
- rag_agent.py - 검색과 생성 조율

### src/cli/
- cli_tool.py - 커맨드라인 인터페이스

### src/mcp/
- mcp_server.py - Model Context Protocol 서버

### src/monitoring/
- logger.py - 구조화된 로깅
- metrics.py - 메트릭 수집
