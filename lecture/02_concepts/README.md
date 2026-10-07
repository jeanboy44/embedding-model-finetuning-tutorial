# 2교시: 용어와 과제 정의

장표: [2교시 용어와 과제 정의](https://claude.ai/artifact/YCojzoPbMkeAamJ7WQhSEG). 이론 시간이라 실습은 없다. 아래는 시간이 남을 때 하는 선택 실습이다.

명령은 모두 저장소 루트에서, 위에서부터 순서대로 실행한다. macOS 터미널과 Windows PowerShell에서 같은 명령 그대로 돈다.

## 선택 실습: 임베딩 공간 탐색

문장이 벡터가 되고, 뜻이 가까우면 벡터도 가깝다는 것을 숫자로 본다(유사도 행렬, 검색, 군집).

```shell
uv run python lecture/02_concepts/01_embedding_exploration.py
```

- 몇 초 걸린다. API 키는 쓰지 않는다.
- e5는 관계없는 문장끼리도 유사도가 0.79처럼 높게 나온다(관련 문장은 0.91). 절대값이 아니라 **순위**를 본다.
- 데이터는 `data/sample_docs.json`(저장소에 들어 있다)과 학습 전 e5 모델이다. 모델이 없으면 3교시 준비를 먼저 한다: `uv run python lecture/03_setup/01_doctor.py --fix`
