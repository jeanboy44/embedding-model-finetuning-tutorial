// 화면: 파일 안에 든 데이터를 풀고, 질문을 임베딩해 조문을 찾아 보여 준다.
// build.py가 이 파일 앞에 onnxruntime-web(ort), tokenizer.js, search.js를 붙여 한 <script>로 넣는다.
// 모델이 둘(파인튜닝 · 학습 전)이면 같은 질문의 결과를 나란히 놓고, 상대 모델에서의 순위도 보여 준다.
import { UnigramTokenizer } from "./tokenizer.js";
import { meanPoolNormalize, rankArticles, scoreAll } from "./search.js";

const THEME_LABELS = {
  youth: "청년·근로",
  traffic: "교통",
  tax: "세금",
  finance: "금융",
  consumer: "소비자",
  electric: "전기",
};
const QUERY_PREFIX = "query: ";
const DEPTH = 100; // 상대 모델 순위를 찾아볼 깊이 (조 단위)

const $ = (selector) => document.querySelector(selector);
const state = {
  docs: [], // {law, id, parentId, title, text}
  laws: [], // {name, type, theme, url, effectiveDate}
  articleOf: null, // 조각 → 조 번호
  pieces: [], // 조 번호 → 조각 번호들
  models: [], // {key, label, dim, tokenizer, session, codes, scales}
  mode: "compare",
};

// ---- 파일 안의 데이터 풀기 ----------------------------------------------------

function base64ToBytes(text) {
  if (Uint8Array.fromBase64) return Uint8Array.fromBase64(text);
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

async function readPayload(name) {
  const tag = document.getElementById(`payload-${name}`);
  const bytes = base64ToBytes(tag.textContent.trim());
  tag.remove(); // 큰 문자열을 메모리에서 놓아 준다
  if (tag.dataset.gzip !== "1") return bytes;
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

const readJson = async (name) => JSON.parse(new TextDecoder().decode(await readPayload(name)));

function setStatus(text, { busy = true, error = false } = {}) {
  const status = $("#status");
  status.textContent = text;
  status.classList.toggle("busy", busy);
  status.classList.toggle("error", error);
}

async function load() {
  const started = performance.now();
  setStatus("조문 데이터를 푸는 중…");
  const meta = await readJson("meta");
  const corpus = await readJson("docs");
  state.laws = corpus.laws;
  state.docs = corpus.docs.map(([law, id, parentId, title, text]) => ({ law, id, parentId, title, text }));
  indexArticles();

  ort.env.wasm.wasmBinary = await readPayload("wasm");
  ort.env.wasm.numThreads = 1; // 파일로 열면 멀티스레드(SharedArrayBuffer)를 쓸 수 없다
  for (const info of meta.models) {
    setStatus(`${info.label} 모델을 불러오는 중…`);
    const vectors = await readPayload(`vectors-${info.key}`);
    const n = state.docs.length;
    const model = {
      ...info,
      tokenizer: new UnigramTokenizer(await readJson(`tokenizer-${info.key}`)),
      scales: new Float32Array(vectors.buffer.slice(0, n * 4)),
      codes: new Int8Array(vectors.buffer, n * 4, n * info.dim),
      session: await ort.InferenceSession.create(await readPayload(`model-${info.key}`), { executionProviders: ["wasm"] }),
    };
    await embed(model, "준비"); // 첫 호출의 초기화 시간을 검색 전에 치른다
    state.models.push(model);
  }
  state.mode = state.models.length > 1 ? "compare" : state.models[0].key;

  fillModes();
  fillFilters();
  renderAbout(meta);
  const seconds = ((performance.now() - started) / 1000).toFixed(1);
  setStatus(`준비 완료 · 조문 ${state.docs.length.toLocaleString()}건 · ${seconds}초`, { busy: false });
  for (const control of document.querySelectorAll("#form [disabled], .example")) control.disabled = false;
  $("#query").focus();
}

function indexArticles() {
  const ids = new Map();
  state.articleOf = new Int32Array(state.docs.length);
  state.docs.forEach((doc, i) => {
    if (!ids.has(doc.parentId)) {
      ids.set(doc.parentId, state.pieces.length);
      state.pieces.push([]);
    }
    const article = ids.get(doc.parentId);
    state.articleOf[i] = article;
    state.pieces[article].push(i);
  });
}

// ---- 임베딩과 검색 -------------------------------------------------------------

async function embed(model, text) {
  const ids = model.tokenizer.encode(QUERY_PREFIX + text);
  const shape = [1, ids.length];
  const feeds = {
    input_ids: new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), shape),
    attention_mask: new ort.Tensor("int64", new BigInt64Array(ids.length).fill(1n), shape),
  };
  const output = await model.session.run(feeds);
  const hidden = output.last_hidden_state;
  const vector = meanPoolNormalize(hidden.data, ids.length, model.dim);
  hidden.dispose?.();
  return { vector, tokens: ids.length };
}

function currentFilter() {
  const theme = $("#theme").value;
  const lawType = $("#law-type").value;
  const lawName = $("#law-name").value;
  if (!theme && !lawType && !lawName) return undefined;
  const allowed = new Uint8Array(state.laws.length);
  state.laws.forEach((law, i) => {
    allowed[i] = (!theme || law.theme === theme) && (!lawType || law.type === lawType) && (!lawName || law.name === lawName);
  });
  return (i) => allowed[state.docs[i].law] === 1;
}

async function searchWith(model, query, keep) {
  const t0 = performance.now();
  const { vector, tokens } = await embed(model, query);
  const t1 = performance.now();
  const ranked = rankArticles(scoreAll(vector, model.codes, model.scales), state.articleOf, DEPTH, keep);
  const t2 = performance.now();
  const rankOf = new Map(ranked.map((hit, i) => [state.articleOf[hit.index], i + 1]));
  return { model, ranked, rankOf, tokens, embedMs: t1 - t0, searchMs: t2 - t1 };
}

async function search() {
  const query = $("#query").value.trim();
  if (!query || state.models.length === 0) return;
  const keep = currentFilter();
  const results = [];
  for (const model of state.models) results.push(await searchWith(model, query, keep));
  const k = Number($("#top-k").value);
  const shown = state.mode === "compare" ? results : results.filter((r) => r.model.key === state.mode);
  const n = state.docs.length.toLocaleString();
  $("#timing").textContent = shown
    .map((r) => `${r.model.label}: 임베딩 ${r.embedMs.toFixed(0)}ms · ${n}건 비교 ${r.searchMs.toFixed(0)}ms`)
    .join("  |  ") + `  ·  질문 ${results[0].tokens}토큰`;
  const list = $("#results");
  list.className = state.mode === "compare" ? "compare" : "";
  list.replaceChildren(...shown.map((result) => renderColumn(result, results, k)));
}

// ---- 화면 그리기 ---------------------------------------------------------------

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  node.append(...children.filter((c) => c !== null && c !== undefined && c !== false));
  return node;
}

function renderColumn(result, all, k) {
  const compare = state.mode === "compare";
  const others = all.filter((r) => r !== result);
  const hits = result.ranked.slice(0, k);
  return el("section", { class: "column" },
    compare ? el("h2", { class: `column-head ${result.model.key}` }, result.model.label) : null,
    hits.length === 0
      ? el("p", { class: "empty" }, "조건에 맞는 조문이 없습니다. 필터를 풀어 보세요.")
      : el("div", { class: "hits" }, ...hits.map((hit, i) =>
          renderHit(hit.index, hit.score, i + 1, compare ? others : [], compare ? 120 : 220))));
}

function renderHit(index, score, rank, others, previewChars) {
  const doc = state.docs[index];
  const law = state.laws[doc.law];
  const article = state.articleOf[index];
  const preview = doc.text.length > previewChars ? `${doc.text.slice(0, previewChars)}…` : doc.text;
  const body = el("p", { class: "text" }, preview);
  const actions = el("div", { class: "actions" });
  if (preview !== doc.text) {
    actions.append(el("button", {
      type: "button",
      onclick: (event) => {
        const open = body.dataset.open === "1";
        body.textContent = open ? preview : doc.text;
        body.dataset.open = open ? "0" : "1";
        event.currentTarget.textContent = open ? "전문 보기" : "접기";
      },
    }, "전문 보기"));
  }
  const siblings = state.pieces[article];
  if (siblings.length > 1) {
    actions.append(el("button", {
      type: "button",
      onclick: (event) => {
        event.currentTarget.remove();
        body.after(el("div", { class: "article" }, ...siblings.map((i) =>
          el("section", { class: i === index ? "piece current" : "piece" },
            el("h4", {}, state.docs[i].title), el("p", { class: "text" }, state.docs[i].text)))));
      },
    }, `같은 조 전체 (${siblings.length}조각)`));
  }
  if (law.url) actions.append(el("a", { href: law.url, target: "_blank", rel: "noopener" }, "법제처 ↗"));
  const elsewhere = others.map((other) => {
    const otherRank = other.rankOf.get(article);
    return el("span", { class: `elsewhere ${otherRank ? "" : "missing"}` },
      `${other.model.label} ${otherRank ? `${otherRank}위` : `${DEPTH}위 밖`}`);
  });
  return el("article", { class: "hit" },
    el("div", { class: "hit-head" },
      el("span", { class: "rank" }, String(rank)),
      el("h3", {}, doc.title),
      el("span", { class: "score", title: "코사인 유사도" }, score.toFixed(3))),
    el("div", { class: "tags" },
      el("span", { class: "tag" }, THEME_LABELS[law.theme] ?? law.theme),
      el("span", { class: "tag" }, law.type),
      ...elsewhere),
    body,
    actions);
}

function option(value, label) {
  return el("option", { value }, label);
}

function fillModes() {
  const modes = $("#modes");
  if (state.models.length < 2) {
    modes.remove();
    return;
  }
  const choices = [["compare", "나란히 비교"], ...state.models.map((m) => [m.key, m.label])];
  modes.append(...choices.map(([value, label]) => el("button", {
    type: "button",
    "aria-pressed": String(value === state.mode),
    onclick: (event) => {
      state.mode = value;
      for (const button of modes.children) button.setAttribute("aria-pressed", String(button === event.currentTarget));
      if ($("#query").value.trim()) $("#form").requestSubmit();
    },
  }, label)));
}

function fillFilters() {
  const themes = [...new Set(state.laws.map((l) => l.theme))];
  $("#theme").append(...themes.map((t) => option(t, THEME_LABELS[t] ?? t)));
  const types = [...new Set(state.laws.map((l) => l.type))];
  $("#law-type").append(...types.map((t) => option(t, t)));
  fillLawNames();
}

function fillLawNames() {
  const theme = $("#theme").value;
  const lawType = $("#law-type").value;
  const select = $("#law-name");
  const selected = select.value;
  const names = state.laws
    .filter((l) => (!theme || l.theme === theme) && (!lawType || l.type === lawType))
    .map((l) => l.name)
    .sort((a, b) => a.localeCompare(b, "ko"));
  select.replaceChildren(option("", `법령 전체 (${names.length})`), ...names.map((name) => option(name, name)));
  select.value = names.includes(selected) ? selected : "";
}

function renderAbout(meta) {
  $("#about-model").replaceChildren(...meta.models.map((m) => el("div", {}, `${m.label}: ${m.description}`)));
  $("#about-built").textContent = meta.built_at;
  $("#about-docs").textContent = `${state.docs.length.toLocaleString()}개 조각, ${state.pieces.length.toLocaleString()}개 조 (법령 ${state.laws.length}개)`;
}

// ---- 이벤트 --------------------------------------------------------------------

$("#form").addEventListener("submit", (event) => {
  event.preventDefault();
  search().catch((error) => setStatus(`검색 실패: ${error.message}`, { busy: false, error: true }));
});
for (const id of ["#theme", "#law-type"]) $(id).addEventListener("change", fillLawNames);
for (const id of ["#theme", "#law-type", "#law-name", "#top-k"]) {
  $(id).addEventListener("change", () => $("#query").value.trim() && $("#form").requestSubmit());
}
for (const chip of document.querySelectorAll(".example")) {
  chip.addEventListener("click", () => {
    $("#query").value = chip.textContent;
    $("#form").requestSubmit();
  });
}

load().catch((error) => {
  console.error(error);
  setStatus(`불러오기 실패: ${error.message}`, { busy: false, error: true });
});
