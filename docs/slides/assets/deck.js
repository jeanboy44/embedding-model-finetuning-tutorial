// 강의 슬라이드 공용 동작: 1600x900 무대를 화면에 맞추고 한 장씩 넘긴다.
// 조작: ←→ / 스페이스 / 클릭(왼쪽 1/3은 이전), Home·End, F 전체 화면. 주소 끝 #3이면 3번째 장부터.
(() => {
  const stage = document.querySelector(".stage");
  const slides = [...stage.querySelectorAll(".slide")];
  const pager = Object.assign(document.createElement("div"), { className: "pager" });
  const progress = Object.assign(document.createElement("div"), { className: "progress" });
  stage.append(pager, progress);
  let index = 0;

  const fit = () => { stage.style.transform = `scale(${Math.min(innerWidth / 1600, innerHeight / 900)})`; };
  addEventListener("resize", fit);
  fit();

  const show = (i) => {
    index = Math.max(0, Math.min(slides.length - 1, i));
    slides.forEach((el, n) => el.classList.toggle("active", n === index));
    pager.textContent = `${index + 1} / ${slides.length}`;
    progress.style.width = `${((index + 1) / slides.length) * 100}%`;
    history.replaceState(null, "", `#${index + 1}`);
    // 막대(.fill[data-w])는 그 장에 도착할 때 차오른다
    slides[index].querySelectorAll(".fill").forEach((f) => {
      f.style.width = "0";
      requestAnimationFrame(() => requestAnimationFrame(() => { f.style.width = `${f.dataset.w}%`; }));
    });
  };

  document.addEventListener("keydown", (e) => {
    if (["ArrowRight", "ArrowDown", "PageDown", " "].includes(e.key)) { e.preventDefault(); show(index + 1); }
    if (["ArrowLeft", "ArrowUp", "PageUp"].includes(e.key)) { e.preventDefault(); show(index - 1); }
    if (e.key === "Home") show(0);
    if (e.key === "End") show(slides.length - 1);
    if (e.key === "f" || e.key === "F") {
      document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
    }
  });
  stage.addEventListener("click", (e) => {
    const r = stage.getBoundingClientRect();
    show(index + (e.clientX - r.left > r.width / 3 ? 1 : -1));
  });
  show((parseInt(location.hash.slice(1), 10) || 1) - 1);
})();
