"""ragkit: 임베딩 모델 인덱싱·학습·평가·최적화."""

import os

# mlflow를 import할 때마다 찍히는 에이전트 안내 한 줄을 끈다 (실습 화면이 지저분해진다)
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
