// XLM-R(e5) Unigram 토크나이저를 브라우저에서 돌린다. 외부 라이브러리 없이 tokenizer.json만 읽는다.
//
// 파이썬 tokenizers와 같은 순서로 처리한다:
//   정규화(NFKC, 연속 공백 → 한 칸) → Metaspace(공백 → "▁", 맨 앞에 "▁") → 단어마다 Viterbi
//   → <s> ... </s>, 최대 512 토큰
// 정규화는 tokenizer.json의 Precompiled(sentencepiece nmt_nfkc 표)를 NFKC로 대신한다.
// 같은 결과인지는 tests/apps/test_offline_search.py가 코퍼스·질문 전체로 확인한다.

const UNK_PENALTY = 10;

export class UnigramTokenizer {
  constructor(tokenizerJson, { maxLength = 512 } = {}) {
    const { model, added_tokens: added } = tokenizerJson;
    if (model.type !== "Unigram") throw new Error(`Unigram이 아닌 토크나이저: ${model.type}`);
    const special = (name) => added.find((t) => t.content === name).id;
    this.bos = special("<s>");
    this.eos = special("</s>");
    this.unk = model.unk_id;
    this.maxLength = maxLength;
    this.scores = new Float64Array(model.vocab.length);
    // 조각 → id 사전과, 조각 길이(코드포인트 수)의 최댓값
    this.pieces = new Map();
    this.maxPieceLength = 1;
    let minScore = Infinity;
    model.vocab.forEach(([piece, score], id) => {
      this.scores[id] = score;
      if (added.some((t) => t.id === id)) return; // 특수 토큰은 본문에서 찾지 않는다
      this.pieces.set(piece, id);
      this.maxPieceLength = Math.max(this.maxPieceLength, [...piece].length);
      minScore = Math.min(minScore, score);
    });
    this.unkScore = minScore - UNK_PENALTY;
  }

  /** 문장 → 토큰 id 배열 (<s>, </s> 포함). */
  encode(text) {
    const normalized = text
      .normalize("NFKC")
      .replace(/[\t\n\r]/g, " ") // nmt_nfkc: 탭·줄바꿈은 공백, 나머지 제어 문자는 지운다
      .replace(/[\u0000-\u001f\u007f]/g, "")
      .replace(/ {2,}/g, " ");
    const spaced = normalized.replaceAll(" ", "▁");
    const withPrefix = spaced.startsWith("▁") ? spaced : `▁${spaced}`;
    // "▁"마다 끊되 "▁"는 뒤 단어에 붙인다 ("▁근로▁시간" → "▁근로", "▁시간")
    const words = withPrefix.split(/(?=▁)/);
    const ids = [this.bos];
    for (const word of words) ids.push(...this.#viterbi(word));
    if (ids.length > this.maxLength - 1) ids.length = this.maxLength - 1;
    ids.push(this.eos);
    return ids;
  }

  // 단어 하나를 점수 합이 가장 큰 조각들로 나눈다. 어휘에 없는 글자는 <unk> 하나 (이어지면 합친다).
  #viterbi(word) {
    const chars = [...word];
    const n = chars.length;
    const best = new Float64Array(n + 1).fill(-Infinity);
    const prev = new Int32Array(n + 1).fill(-1);
    const prevId = new Int32Array(n + 1).fill(-1);
    best[0] = 0;
    for (let start = 0; start < n; start++) {
      if (best[start] === -Infinity) continue;
      let piece = "";
      let hasSingle = false;
      for (let end = start + 1; end <= Math.min(n, start + this.maxPieceLength); end++) {
        piece += chars[end - 1];
        const id = this.pieces.get(piece);
        if (id === undefined) continue;
        if (end === start + 1) hasSingle = true;
        const score = best[start] + this.scores[id];
        if (score > best[end]) {
          best[end] = score;
          prev[end] = start;
          prevId[end] = id;
        }
      }
      if (!hasSingle) {
        const score = best[start] + this.unkScore;
        if (score > best[start + 1]) {
          best[start + 1] = score;
          prev[start + 1] = start;
          prevId[start + 1] = this.unk;
        }
      }
    }
    const ids = [];
    for (let end = n; end > 0; end = prev[end]) ids.push(prevId[end]);
    ids.reverse();
    return ids.filter((id, i) => !(id === this.unk && ids[i - 1] === this.unk));
  }
}
