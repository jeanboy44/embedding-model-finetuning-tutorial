// 질문 임베딩(평균 풀링 + L2 정규화)과 조문 벡터 전수 비교.
// 파이썬 쪽 ragkit.embeddings.onnx_backend.mean_pool과 같은 계산이다.

/** last_hidden_state [1, seq, dim]에서 패딩 없는 한 문장의 평균을 내고 길이 1로 맞춘다. */
export function meanPoolNormalize(hidden, seqLength, dim) {
  const out = new Float32Array(dim);
  for (let t = 0; t < seqLength; t++) {
    for (let d = 0; d < dim; d++) out[d] += hidden[t * dim + d];
  }
  let norm = 0;
  for (let d = 0; d < dim; d++) {
    out[d] /= seqLength;
    norm += out[d] * out[d];
  }
  norm = Math.max(Math.sqrt(norm), 1e-12);
  for (let d = 0; d < dim; d++) out[d] /= norm;
  return out;
}

/**
 * 질문 벡터와 조문 벡터 전체의 코사인 유사도.
 *
 * @param {Float32Array} query 길이 1로 맞춘 질문 벡터 (dim)
 * @param {Int8Array} codes 조문 벡터 n×dim. 실제 벡터 = codes[i] × scales[i]
 * @param {Float32Array} scales 조문마다의 배율 (n)
 * @returns {Float32Array} 조문마다의 점수 (n)
 */
export function scoreAll(query, codes, scales) {
  const dim = query.length;
  const scores = new Float32Array(scales.length);
  for (let i = 0; i < scales.length; i++) {
    let dot = 0;
    const base = i * dim;
    for (let d = 0; d < dim; d++) dot += query[d] * codes[base + d];
    scores[i] = dot * scales[i];
  }
  return scores;
}

/**
 * 점수 순으로 조(條) 단위 순위를 낸다. 같은 조의 항·호 조각은 가장 높은 것 하나로 센다.
 *
 * @param {Float32Array} scores scoreAll의 결과
 * @param {Int32Array} articleOf 조각마다 속한 조의 번호
 * @param {number} limit 돌려줄 조 수
 * @param {(i: number) => boolean} [keep] 필터. false인 조각은 건너뛴다
 * @returns {{index: number, score: number}[]} 조마다 가장 높은 조각, 점수 내림차순
 */
export function rankArticles(scores, articleOf, limit, keep) {
  const order = [];
  for (let i = 0; i < scores.length; i++) if (!keep || keep(i)) order.push(i);
  order.sort((a, b) => scores[b] - scores[a]);
  const seen = new Set();
  const ranked = [];
  for (const i of order) {
    if (seen.has(articleOf[i])) continue;
    seen.add(articleOf[i]);
    ranked.push({ index: i, score: scores[i] });
    if (ranked.length === limit) break;
  }
  return ranked;
}
