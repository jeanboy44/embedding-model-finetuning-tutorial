"""claude.ai Slides 덱 소스(deck.json + slides/<id>.html)를 브라우저로 바로 여는 HTML 파일 하나로 묶는다.

    uv run python scripts/deck_to_html.py <덱 폴더> <출력.html>

덱 폴더는 덱에서 받은 project/ 폴더(deck.json과 slides/가 있는 곳)다.
장마다 <section>, 발표 대본은 <aside>로 그대로 두고, 화면 맞춤 · 넘기기 · 대본 보기 · 인쇄(PDF)용
스타일과 스크립트, 덱 전용 요소(x-icon · x-shape · x-connector)를 그리는 코드를 붙인다.

키: ← → / Space / PageUp · PageDown 넘기기, Home · End 처음 · 끝, N 대본, F 전체 화면. 인쇄하면 장마다 한 쪽.
"""

import html
import json
import re
from pathlib import Path

from cyclopts import run
from rich.console import Console
from rich.markup import escape

console = Console(emoji=False, soft_wrap=True)  # 긴 경로·URL을 중간에서 끊지 않게

VIEWER_CSS = """
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; }
  body { background: #2b2f36; overflow: hidden; font-family: 'Noto Sans KR', Arial, sans-serif; color: #1f2a37; }
  #stage { position: absolute; left: 50%; top: 50%; width: 1920px; height: 1080px; transform-origin: 0 0; }
  #stage > section { position: absolute; inset: 0; width: 1920px; height: 1080px; overflow: hidden;
    font-size: 32px; line-height: 1.4; word-break: keep-all; opacity: 0; visibility: hidden; transition: opacity .25s; }
  #stage > section.on { opacity: 1; visibility: visible; }
  section:not([style*="display"]), section div:not([style*="display"]) { display: flex; flex-direction: column; }
  section div { min-width: 0; }
  h1, h2, h3, p, ul, ol { margin: 0; }
  h1 { font-size: 96px; font-weight: 600; line-height: 1.1; }
  h2 { font-size: 64px; font-weight: 600; line-height: 1.15; }
  h3 { font-size: 44px; font-weight: 600; line-height: 1.2; }
  p, li { font-size: inherit; }
  ul, ol { padding-left: 1.2em; display: flex; flex-direction: column; gap: .3em; }
  table { border-collapse: collapse; }
  th, td { padding: .35em .6em; border-bottom: 1px solid rgba(31, 42, 55, .14); text-align: left; vertical-align: top; }
  th { font-weight: 700; border-bottom: 2px solid rgba(31, 42, 55, .22); }
  aside { display: none; }
  x-icon, x-shape { display: block; flex: none; }
  x-icon svg { width: 100%; height: 100%; display: block; }
  x-connector { display: block; }
  x-shape[kind="ellipse"] { border-radius: 50%; }
  x-shape[kind="arrow-right"] { clip-path: polygon(0 30%, 60% 30%, 60% 0, 100% 50%, 60% 100%, 60% 70%, 0 70%); }
  x-shape[kind="arrow-left"] { clip-path: polygon(100% 30%, 40% 30%, 40% 0, 0 50%, 40% 100%, 40% 70%, 100% 70%); }
  x-shape[kind="arrow-down"] { clip-path: polygon(30% 0, 70% 0, 70% 60%, 100% 60%, 50% 100%, 0 60%, 30% 60%); }
  x-shape[kind="arrow-up"] { clip-path: polygon(30% 100%, 70% 100%, 70% 40%, 100% 40%, 50% 0, 0 40%, 30% 40%); }
  x-shape[kind="diamond"] { clip-path: polygon(50% 0, 100% 50%, 50% 100%, 0 50%); }
  #bar { position: fixed; right: 16px; bottom: 12px; font: 13px/1.4 Arial, sans-serif; color: #c9ced6;
    background: rgba(0, 0, 0, .35); padding: 4px 10px; border-radius: 6px; user-select: none; }
  #notes { position: fixed; left: 0; right: 0; bottom: 0; max-height: 34vh; overflow: auto; display: none;
    background: rgba(20, 23, 28, .94); color: #e8eaed; font: 16px/1.7 'Noto Sans KR', Arial, sans-serif; padding: 16px 24px 40px; }
  body.show-notes #notes { display: block; }
  @media print {
    @page { size: 1920px 1080px; margin: 0; }
    html, body { height: auto; overflow: visible; background: none; }
    #stage { position: static; transform: none !important; width: auto; height: auto; }
    #stage > section { position: relative; opacity: 1; visibility: visible; break-after: page; }
    #bar, #notes { display: none !important; }
  }
"""

VIEWER_JS = r"""
(() => {
  // 덱 전용 요소: x-icon (Lucide 모양), x-connector (선 · 화살표)
  const ICONS = {
    Chart: '<path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>',
    Chat: '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    Check: '<path d="M20 6 9 17l-5-5"/>',
    CheckCircle: '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    Database: '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/>',
    Globe: '<circle cx="12" cy="12" r="10"/><path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/><path d="M2 12h20"/>',
    Home: '<path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M9 22V12h6v10"/>',
    Lightbulb: '<path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/>',
    PaperPlane: '<path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/>',
    Search: '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    Settings: '<path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/>',
    Warning: '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
  };
  document.querySelectorAll('x-icon').forEach((el) => {
    const body = ICONS[el.getAttribute('name')] || '<circle cx="12" cy="12" r="9"/>';
    el.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' + body + '</svg>';
  });

  const NS = 'http://www.w3.org/2000/svg';
  let markerId = 0;
  document.querySelectorAll('x-connector').forEach((el) => {
    const host = el.parentElement;
    if (getComputedStyle(host).position === 'static') host.style.position = 'relative';
    const cs = getComputedStyle(el);
    const color = cs.color;
    const w = parseFloat(el.style.borderWidth) || 2;
    const dash = (el.style.borderStyle || '').includes('dash') ? `${w * 3} ${w * 2}` : (el.style.borderStyle || '').includes('dot') ? `${w} ${w * 1.5}` : '';
    const num = (v, size) => (String(v).endsWith('%') ? (parseFloat(v) / 100) * size : parseFloat(v));
    const hw = host.offsetWidth, hh = host.offsetHeight;
    const x1 = num(el.getAttribute('x1'), hw), y1 = num(el.getAttribute('y1'), hh);
    const x2 = num(el.getAttribute('x2'), hw), y2 = num(el.getAttribute('y2'), hh);
    const head = el.getAttribute('head') || 'end';
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('style', 'position:absolute;left:0;top:0;width:1px;height:1px;overflow:visible;pointer-events:none');
    const id = 'xc' + markerId++;
    svg.innerHTML = `<defs><marker id="${id}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="${Math.max(4, 12 / w * 2)}" markerHeight="${Math.max(4, 12 / w * 2)}" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="${color}"/></marker></defs>`;
    const line = document.createElementNS(NS, 'line');
    Object.entries({ x1, y1, x2, y2, stroke: color, 'stroke-width': w }).forEach(([k, v]) => line.setAttribute(k, v));
    if (dash) line.setAttribute('stroke-dasharray', dash);
    if (head === 'end' || head === 'both') line.setAttribute('marker-end', `url(#${id})`);
    if (head === 'both') line.setAttribute('marker-start', `url(#${id})`);
    svg.appendChild(line);
    el.replaceWith(svg);
  });

  // 넘기기 · 화면 맞춤 · 대본
  const stage = document.getElementById('stage');
  const slides = [...stage.children].filter((s) => s.tagName === 'SECTION');
  const bar = document.getElementById('bar');
  const notes = document.getElementById('notes');
  let cur = 0;
  const fit = () => {
    const s = Math.min(innerWidth / 1920, innerHeight / 1080);
    stage.style.transform = `scale(${s}) translate(-50%, -50%)`;
  };
  const show = (i) => {
    cur = Math.max(0, Math.min(slides.length - 1, i));
    slides.forEach((s, k) => s.classList.toggle('on', k === cur));
    bar.textContent = `${cur + 1} / ${slides.length}`;
    const aside = slides[cur].querySelector('aside');
    notes.textContent = aside ? aside.textContent.trim() : '(대본 없음)';
    history.replaceState(null, '', '#' + slides[cur].id);
  };
  addEventListener('resize', fit);
  addEventListener('keydown', (e) => {
    if (['ArrowRight', 'ArrowDown', 'PageDown', ' '].includes(e.key)) { show(cur + 1); e.preventDefault(); }
    else if (['ArrowLeft', 'ArrowUp', 'PageUp'].includes(e.key)) { show(cur - 1); e.preventDefault(); }
    else if (e.key === 'Home') show(0);
    else if (e.key === 'End') show(slides.length - 1);
    else if (e.key === 'n' || e.key === 'N') document.body.classList.toggle('show-notes');
    else if (e.key === 'f' || e.key === 'F') document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
  });
  stage.addEventListener('click', (e) => { if (e.target.closest('a')) return; show(cur + (e.clientX > innerWidth / 3 ? 1 : -1)); });
  const fromHash = () => {
    const i = slides.findIndex((s) => '#' + s.id === location.hash);
    if (i >= 0 && i !== cur) show(i);
  };
  addEventListener('hashchange', fromHash);
  fit();
  const start = slides.findIndex((s) => '#' + s.id === location.hash);
  show(start < 0 ? 0 : start);
})();
"""


def build(deck_dir: Path) -> str:
    deck = json.loads((deck_dir / "deck.json").read_text(encoding="utf-8"))
    starts = {s["start"]: s["description"] for s in deck.get("sections", {}).values()}
    sections = []
    for sid in deck["order"]:
        src = (deck_dir / "slides" / f"{sid}.html").read_text(encoding="utf-8").strip()
        if sid in starts:
            src = re.sub(
                r"<section\b", f'<section data-section="{html.escape(starts[sid], quote=True)}"', src, count=1
            )
        sections.append(src)
    links = "\n".join(
        f'<link rel="stylesheet" href="{html.escape(f["href"], quote=True)}">'
        for f in deck.get("faces", {}).values()
        if f.get("href")
    )
    title = html.escape(deck["title"])
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
{links}
<style>{VIEWER_CSS}</style>
</head>
<body>
<!-- 장표 원본: claude.ai Slides 덱. 장 하나 = <section>, 발표 대본 = <aside>. 키: ← → 넘기기, N 대본, F 전체 화면 -->
<div id="stage">
{chr(10).join(sections)}
</div>
<div id="bar"></div>
<div id="notes"></div>
<script>{VIEWER_JS}</script>
</body>
</html>
"""


def main(deck_dir: Path, out: Path) -> None:
    """Slides 덱 소스를 HTML 파일 하나로 묶는다.

    Parameters
    ----------
    deck_dir
        deck.json과 slides/가 있는 폴더.
    out
        출력 HTML 경로.
    """
    out.write_text(build(deck_dir), encoding="utf-8")
    order = json.loads((deck_dir / "deck.json").read_text(encoding="utf-8"))["order"]
    console.print(f"[dim]{escape(str(out))}[/] ([bold]{len(order)}[/]장)")


if __name__ == "__main__":
    run(main)
