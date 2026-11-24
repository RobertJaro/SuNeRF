"""Static review page of :mod:`sunerf.data.frame_review`.

One self-contained HTML file next to a ``thumbs/`` directory. Served by
``frame_review review``, every decision is saved to ``decisions.csv`` in the
review directory; opened as a plain file, decisions stay in the browser
(localStorage) until **Download decisions.csv**.
"""

PAGE_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Frame review</title>
<style>
:root {
  --bg: #111317; --panel: #191c22; --line: #2a2f38; --text: #e4e6ea; --muted: #8b929e;
  --ok: #6b7280; --warn: #f0a020; --reject: #e5484d; --select: #4cc2ff; --user: #b48cff;
}
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; background: var(--bg); color: var(--text);
  font: 13px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
body { display: flex; flex-direction: column; }
header { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 16px; padding: 8px 16px;
  border-bottom: 1px solid var(--line); background: var(--panel); }
header h1 { font-size: 15px; margin: 0; font-weight: 600; max-width: 40vw; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; }
.counts { color: var(--muted); }
.counts b { color: var(--text); font-weight: 600; }
.counts .r { color: var(--reject); } .counts .w { color: var(--warn); } .counts .u { color: var(--user); }
#save-status { font-size: 12px; color: var(--muted); }
#save-status.ok { color: #7fd18b; } #save-status.err { color: #ff9a9d; }
.controls { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-left: auto; }
select, button { background: #232730; color: var(--text); border: 1px solid var(--line);
  border-radius: 5px; padding: 4px 8px; font: inherit; cursor: pointer; }
button:hover, select:hover { border-color: #4a5160; }
button.primary { background: #1f4f7a; border-color: #2c6aa0; }
#saturation-bar { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 12px;
  padding: 6px 16px; border-bottom: 1px solid var(--line); background: #15181d; font-size: 12px; }
#saturation-bar input { width: 96px; background: #232730; color: var(--text); border: 1px solid var(--line);
  border-radius: 5px; padding: 3px 6px; font: inherit; }
#sat-info { color: var(--muted); }
#sat-info b { color: var(--text); font-weight: 600; }
#sat-info code { color: #ffd27a; }
#sat-status.ok { color: #7fd18b; } #sat-status.err { color: #ff9a9d; }
main { display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(0, 1fr);
  flex: 1; min-height: 0; }
.pane { overflow: auto; padding: 12px 16px; }
.pane + .pane { border-left: 1px solid var(--line); }
h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted);
  margin: 4px 0 8px; font-weight: 600; display: flex; gap: 8px; align-items: center; }
h2 .hint { text-transform: none; letter-spacing: 0; font-weight: 400; }
.plot-row { display: grid; grid-template-columns: 120px 1fr; align-items: center; gap: 6px; }
.plot-row .label { color: var(--muted); font-size: 12px; text-align: right; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; }
.plot-row canvas { width: 100%; height: 64px; display: block; background: #15181d;
  border: 1px solid var(--line); border-radius: 3px; cursor: crosshair; }
.axis { margin-left: 126px; display: flex; justify-content: space-between; color: var(--muted);
  font-size: 11px; margin-bottom: 14px; }
/* Observation viewer: previous (small), current (all channels), next (small). */
.review-stage { height: 100%; display: flex; flex-direction: column; }
.review-stage > * { flex: none; }
#plots { max-height: 24dvh; overflow-y: auto; }
#viewer { flex: 1; min-height: 0; display: grid;
  grid-template-rows: minmax(0, 1fr) auto minmax(0, 2.4fr) auto minmax(0, 1fr) auto; }
@media (max-width: 799px) {
  main { grid-template-columns: minmax(0, 1fr); grid-template-rows: 100% auto; overflow: auto; }
  .plot-row canvas { height: 36px; }
}
@media (min-width: 800px) and (max-height: 850px) {
  main { grid-template-columns: minmax(0, 2fr) minmax(0, 1fr); }
  .plot-row canvas { height: clamp(32px, 6dvh, 48px); }
  #viewer { grid-auto-flow: column; column-gap: 6px;
    grid-template-columns: minmax(0, 1fr) minmax(0, 2.6fr) minmax(0, 1fr);
    grid-template-rows: minmax(0, 1fr) auto; }
}
.strip { container-type: size; display: flex; justify-content: center; align-items: center; gap: 6px; }
.strip-time { color: var(--muted); font-size: 11px; text-align: center; margin: 3px 0 6px;
  font-variant-numeric: tabular-nums; }
.strip-time.current { color: var(--text); font-size: 12px; font-weight: 600; }
.vcell { position: relative; --aspect: 1; aspect-ratio: var(--aspect);
  width: min(calc((100% - (var(--n) - 1) * 6px) / var(--n)), calc(100cqh * var(--aspect)));
  min-width: 0; background: #000; border: 2px solid var(--line);
  border-radius: 4px; overflow: hidden; cursor: pointer; }
.vcell img { width: 100%; height: 100%; display: block; }
.vcell.blank { background: transparent; border: 1px dashed #3a404b; cursor: default;
  display: grid; place-items: center; color: var(--muted); font-size: 10px; text-align: center; }
.strip.small .vcell.blank { font-size: 0; }
.vcell.warn { border-color: var(--warn); }
.vcell.rej { border-color: var(--reject); }
.vcell.sel { outline: 2px solid var(--select); outline-offset: 1px; }
.vcell .lbl { position: absolute; left: 4px; bottom: 2px; font-size: 10px; color: #fff;
  text-shadow: 0 0 3px #000, 0 0 3px #000; pointer-events: none; }
.strip.small .vcell .lbl { display: none; }
.info { margin-top: 12px; display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.badge { display: inline-block; padding: 1px 8px; border-radius: 10px; font-weight: 600; font-size: 12px; }
.badge.reject { background: #4a1d20; color: #ff9a9d; } .badge.warn { background: #4a3510; color: #ffd27a; }
.badge.ok { background: #263040; color: #b8c4d6; }
.reasons { margin: 6px 0; padding-left: 18px; }
.reasons li.r { color: #ff9a9d; } .reasons li.w { color: #ffd27a; }
table.stats { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
table.stats td { padding: 2px 6px; border-bottom: 1px solid var(--line); }
table.stats td:first-child { color: var(--muted); }
.path { color: var(--muted); font-size: 11px; word-break: break-all; margin-top: 4px; }
.keys { color: var(--muted); font-size: 11px; margin-top: 10px; }
kbd { background: #232730; border: 1px solid var(--line); border-radius: 3px; padding: 0 4px; font-size: 11px; }
table.grid { border-collapse: separate; border-spacing: 3px; }
table.grid th { position: sticky; top: -12px; background: var(--bg); color: var(--muted); font-weight: 600;
  font-size: 11px; padding: 4px 2px; z-index: 1; }
table.grid td.time { color: var(--muted); font-size: 11px; white-space: nowrap; padding-right: 4px;
  font-variant-numeric: tabular-nums; }
.cell { position: relative; width: 84px; height: 84px; border: 2px solid transparent; border-radius: 4px;
  cursor: pointer; background: #000; overflow: hidden; }
.cell img { width: 100%; height: 100%; display: block; }
.cell.warn { border-color: var(--warn); }
.cell.rej { border-color: var(--reject); }
.cell.rej img { opacity: .35; }
.cell.rej::after { content: "✕"; position: absolute; inset: 0; display: grid; place-items: center;
  color: var(--reject); font-size: 34px; font-weight: 700; }
.cell.user::before { content: ""; position: absolute; top: 3px; right: 3px; width: 8px; height: 8px;
  border-radius: 50%; background: var(--user); z-index: 1; }
.cell.sel { outline: 2px solid var(--select); outline-offset: 1px; }
.cell .noimg { color: var(--muted); font-size: 10px; display: grid; place-items: center; height: 100%; }
#mask-controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 6px 0; }
#mask-controls[hidden] { display: none; }
#mask-canvas { position: absolute; inset: 0; width: 100%; height: 100%; opacity: .55;
  touch-action: none; cursor: crosshair; }
.mask-overlay { position: absolute; inset: 0; opacity: .55; pointer-events: none; }
</style>
</head>
<body>
<header>
  <h1 id="title">__TITLE__</h1>
  <div class="counts" id="counts"></div>
  <div class="controls">
    <label>Rows <select id="filter">
      <option value="all">all</option>
      <option value="flagged">flagged</option>
      <option value="rejected">rejected</option>
      <option value="accepted" title="Every available frame in the row is kept">accepted</option>
    </select></label>
    <label>Metric <select id="metric"></select></label>
    <button id="accept-all" title="Keep every frame in this dataset, including frames hidden by the row filter">Accept all frames</button>
    <button id="reset" title="Discard manual decisions">Reset to automatic</button>
    <span id="save-status"></span>
    <button id="import" title="Load decisions from a decisions.csv or a list of paths">Import</button>
    <input type="file" id="import-file" accept=".csv,.txt" hidden>
    <button id="export" class="primary">Download decisions.csv</button>
  </div>
</header>
<div id="saturation-bar">
  <b>Saturation level</b>
  <input type="number" id="sat-level" min="0" step="100" placeholder="none"
    title="Detector DN, the units of instruments[].saturation_dn; empty for none"> DN
  <button id="sat-recommended">Use recommended</button>
  <span id="sat-info"></span>
  <button id="sat-save">Save saturation.csv</button>
  <span id="sat-status"></span>
</div>
<main>
  <section class="pane">
    <div class="review-stage">
      <h2>Sequence <span class="hint">click a point to open the frame; scroll to zoom time, drag to pan, double-click to reset</span></h2>
      <div id="plots"></div>
      <div class="axis"><span id="t0"></span><span id="t1"></span></div>
      <h2>Observation <span class="hint">previous · current (all channels) · next; click a panel to select it</span></h2>
      <div id="mask-controls" hidden>
        <label><input type="checkbox" id="mask-enable" title="Check to paint. Uncheck to remove this frame’s mask."> Edit exclusion mask</label>
        <label id="mask-pair-control"><input type="checkbox" id="mask-pair" checked> Apply to both tB and pB</label>
        <label>Brush radius <select id="mask-radius">
          <option>1</option><option>3</option><option selected>8</option><option>16</option>
          <option>32</option><option>64</option></select> px</label>
        <button id="mask-duplicate">Edit with previous mask <kbd>M</kbd></button>
        <button id="mask-later" title="Add these exclusions to every later frame on the same pixel grid, including hidden rows">Apply mask to all later frames</button>
        <span id="mask-status"></span><span id="mask-save" role="status"></span>
      </div>
      <div id="viewer">
        <div class="strip small" id="vprev"></div>
        <div class="strip-time" id="cprev"></div>
        <div class="strip" id="vcur"></div>
        <canvas id="mask-canvas" hidden></canvas>
        <div class="strip-time current" id="ccur"></div>
        <div class="strip small" id="vnext"></div>
        <div class="strip-time" id="cnext"></div>
      </div>
    </div>
    <div class="info">
      <div>
        <div><span id="badge" class="badge"></span> <span id="decision"></span></div>
        <ul class="reasons" id="reasons"></ul>
        <button id="toggle">Reject frame <kbd>x</kbd></button>
        <button id="toggle-row">Reject observation <kbd>r</kbd></button>
        <div class="path" id="path"></div>
        <div class="keys">
          <kbd>←</kbd><kbd>→</kbd> frame &nbsp; <kbd>↑</kbd><kbd>↓</kbd> row &nbsp;
          <kbd>n</kbd><kbd>p</kbd> next/previous flagged &nbsp; <kbd>x</kbd> toggle frame &nbsp;
          <kbd>r</kbd> toggle row. Each channel uses its instrument colormap on one log range over the whole sequence (hover a column title). Red marks saturation and green marks invalid pixels.
          Copy decisions.csv to the server and run <code>frame_review apply</code>.
        </div>
      </div>
      <table class="stats" id="stats"></table>
    </div>
  </section>
  <section class="pane" id="grid-pane">
    <table class="grid" id="grid"></table>
  </section>
</main>
<script>
const DATA = __DATA__;
const F = DATA.frames, G = DATA.groups;
const channelFolder = f => f.path.slice(0, f.path.lastIndexOf('/'));
const channelFolders = new Set(F.map(channelFolder));
function pairedFolder(f) {
  const folder = channelFolder(f);
  if (!/\/[tp]B$/.test(folder)) return null;
  const other = folder.slice(0, -2) + (folder.endsWith('/tB') ? 'pB' : 'tB');
  return channelFolders.has(other) ? other : null;
}
const STORE = 'frame-review:' + DATA.id;
const byGroup = G.map(() => []);
F.forEach(f => byGroup[f.g].push(f.i));
byGroup.forEach(list => list.forEach((idx, k) => { F[idx].k = k; }));
const rows = Array.from({length: DATA.rows}, () => new Array(G.length).fill(null));
F.forEach(f => { rows[f.row][f.g] = f.i; });

// Served by "frame_review review" every change is saved to decisions.csv;
// opened as a file, decisions stay in the browser until downloaded.
const SERVED = location.protocol === 'http:' || location.protocol === 'https:';
// decisions: path -> rejected (true/false), only where it differs from auto.
let decisions = {};
const masks = SERVED ? {...DATA.masks} : {};
const editing = new Set(Object.keys(masks));
function fromRows(rows) {
  const byPath = {};
  rows.forEach(([path, decision]) => { byPath[path] = decision; });
  decisions = {};
  F.forEach(f => {
    if (!(f.path in byPath)) return;
    const rejected = byPath[f.path] === 'reject';
    if (rejected !== (f.auto === 'reject')) decisions[f.path] = rejected;
  });
}
fromRows(Object.entries(DATA.decisions || {}));
if (!SERVED) {
  try {
    const stored = JSON.parse(localStorage.getItem(STORE) || 'null');
    if (stored) decisions = stored;
  } catch (e) {}
}
let selected = F.findIndex(f => f.auto !== 'ok');
if (selected < 0) selected = 0;
let metric = DATA.metrics[0][0];
const finiteT = F.map(f => f.t).filter(t => t !== null);
const T0 = finiteT.length ? Math.min(...finiteT) : 0, T1 = finiteT.length ? Math.max(...finiteT) : 1;
let view = [T0 - 1, T1 + 1];

const $ = id => document.getElementById(id);
const isRejected = f => (f.path in decisions) ? decisions[f.path] : f.auto === 'reject';
const isChanged = f => (f.path in decisions);
const flagged = f => f.auto !== 'ok' || isRejected(f) || !!masks[f.path];
function csvField(value) {
  const text = value === null || value === undefined ? '' : String(value);
  return /[",\n\r]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
}
function parseCsv(text) {
  const rows = []; let row = [], field = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (c === '"') quoted = false;
      else field += c;
    } else if (c === '"') quoted = true;
    else if (c === ',') { row.push(field); field = ''; }
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++;
      row.push(field); rows.push(row); row = []; field = '';
    } else field += c;
  }
  if (field !== '' || row.length) { row.push(field); rows.push(row); }
  return rows.filter(r => r.length > 1 || r[0] !== '');
}
function decisionsCsv() {
  const lines = [DATA.decisionColumns.join(',')];
  F.forEach(f => lines.push([f.path, isRejected(f) ? 'reject' : 'keep', f.auto, G[f.g], f.time,
    f.reasons.join('; ')].map(csvField).join(',')));
  return lines.join('\n') + '\n';
}
function setStatus(text, cls) { const el = $('save-status'); el.textContent = text; el.className = cls || ''; }
let saveQueue = Promise.resolve(), revision = 0, dirty = false;
window.addEventListener('beforeunload', e => { if (dirty || pendingMasks.size || applyingLater) { e.preventDefault(); e.returnValue = ''; } });
function save() {
  const current = ++revision;
  if (!SERVED) {
    try {
      localStorage.setItem(STORE, JSON.stringify(decisions));
    } catch (e) { setStatus('Browser storage full. Download decisions.csv to save your edits.', 'err'); return; }
    setStatus('opened as a file: download decisions.csv to keep the decisions', 'err');
    return;
  }
  dirty = true;
  setStatus('saving…');
  saveQueue = saveQueue.then(async () => {
    if (current !== revision) return;
    const response = await fetch('decisions.csv', {method: 'POST', body: decisionsCsv(),
      headers: {'Content-Type': 'text/csv'}});
    if (!response.ok) throw Error(response.statusText);
    const info = await response.json();
    if (current === revision) {
      dirty = false;
      setStatus(`saved decisions.csv · ${info.rejected} rejected`, 'ok');
    }
  }).catch(error => setStatus('Save failed. Download decisions.csv to keep your edits. ' + error, 'err'));
}
function setRejected(f, value) {
  if (value === (f.auto === 'reject')) delete decisions[f.path]; else decisions[f.path] = value;
}
function fmt(v) {
  if (v === null || v === undefined || v === '') return '–';
  if (typeof v !== 'number') return String(v);
  // Detector DN and similar counts read best as whole numbers.
  if (Math.abs(v) >= 1e4 && Math.abs(v) < 1e7) return String(Math.round(v));
  if (v !== 0 && (Math.abs(v) >= 1e4 || Math.abs(v) < 1e-2)) return v.toExponential(2);
  return Number.isInteger(v) ? String(v) : v.toFixed(3);
}
const timeLabel = f => f && f.time ? f.time.replace('T', ' ').slice(0, 19) : '';

// ---------------------------------------------------------------- counts
function updateCounts() {
  const auto = F.filter(f => f.auto === 'reject').length;
  const warn = F.filter(f => f.auto === 'warn').length;
  const now = F.filter(isRejected).length;
  const changed = Object.keys(decisions).length;
  $('counts').innerHTML = `<b>${F.length}</b> frames · <span class="r">${auto}</span> auto-rejected · ` +
    `<span class="w">${warn}</span> warnings · <b class="r">${now}</b> rejected now · ` +
    `<span class="u">${changed}</span> manual changes`;
}

// ---------------------------------------------------------------- grid
const cells = {};
function buildGrid() {
  const table = $('grid');
  const head = document.createElement('tr');
  const range = name => {
    const c = (DATA.channels || {})[name];
    const scale = DATA.cme ? 'asinh' : 'log10';
    return c ? `${c.cmap}, ${scale} ${c[scale + '_min'].toFixed(2)} .. ${c[scale + '_max'].toFixed(2)}` : '';
  };
  head.innerHTML = '<th></th>' + G.map(g => `<th title="${range(g)}">${g}</th>`).join('');
  table.appendChild(head);
  const fragment = document.createDocumentFragment();
  rows.forEach((row, r) => {
    const tr = document.createElement('tr');
    tr.id = 'row' + r;
    const first = row.find(i => i !== null);
    const td = document.createElement('td');
    td.className = 'time';
    td.textContent = timeLabel(F[first]);
    tr.appendChild(td);
    row.forEach(i => {
      const cellTd = document.createElement('td');
      if (i !== null) {
        const f = F[i];
        const div = document.createElement('div');
        div.className = 'cell';
        div.title = `${G[f.g]} ${timeLabel(f)}\n${f.reasons.join('\n')}`;
        div.innerHTML = f.thumb ? `<img loading="lazy" src="${f.thumb}" alt="">` : '<div class="noimg">no image</div>';
        div.addEventListener('click', () => select(i, true));
        div.addEventListener('dblclick', () => { setRejected(f, !isRejected(f)); save(); refresh(); });
        cells[i] = div;
        cellTd.appendChild(div);
      }
      tr.appendChild(cellTd);
    });
    fragment.appendChild(tr);
  });
  table.appendChild(fragment);
}
function rowVisible(r) {
  const mode = $('filter').value;
  if (mode === 'all') return true;
  if (mode === 'flagged') return rows[r].some(i => i !== null && flagged(F[i]));
  const rejected = rows[r].some(i => i !== null && isRejected(F[i]));
  return mode === 'accepted' ? !rejected : rejected;
}
function updateGrid() {
  rows.forEach((row, r) => {
    $('row' + r).style.display = rowVisible(r) ? '' : 'none';
    row.forEach(i => {
      if (i === null) return;
      const f = F[i];
      cells[i].className = 'cell' + (isRejected(f) ? ' rej' : (f.auto === 'warn' ? ' warn' : '')) +
        ((isChanged(f) || masks[f.path]) ? ' user' : '') + (i === selected ? ' sel' : '');
    });
  });
}

// ---------------------------------------------------------------- saturation
// Level for instruments[].saturation_dn, in the detector-DN convention of the
// per-frame peaks (metric peak_dn); null means "none".
const SAT = DATA.saturation || {channels: {}, recommended_dn: null, fraction: 0.8};
const HAS_PEAKS = F.some(f => f.stats.peak_dn !== null);
function parseLevel(text) {
  if (text === undefined || text === null || String(text).trim() === '') return undefined;
  if (String(text).trim().toLowerCase() === 'none') return null;
  const value = Number(text);
  return isFinite(value) && value > 0 ? value : undefined;
}
let satLevel = parseLevel(DATA.saturationSelected);
if (satLevel === undefined) satLevel = SAT.recommended_dn ?? null;
function describeSaturation() {
  const parts = [];
  const saturated = Object.entries(SAT.channels).filter(([, c]) => c.saturation_dn);
  if (SAT.recommended_dn) {
    const lowest = saturated.reduce((a, b) => (a[1].saturation_dn <= b[1].saturation_dn ? a : b));
    parts.push(`recommended <b>${SAT.recommended_dn}</b> = ${SAT.fraction} × plateau ` +
      `${fmt(lowest[1].saturation_dn)} (${lowest[0]}, ${lowest[1].saturated_frames} saturated frames)`);
  } else {
    const ceilings = Object.values(SAT.channels).map(c => c.ceiling_dn).filter(v => v);
    parts.push(`no saturation plateau found (highest peak ${ceilings.length ? fmt(Math.max(...ceilings)) : '–'}); ` +
      'keep the current level');
  }
  if (satLevel !== null) {
    const counts = G.map((name, g) => {
      const members = byGroup[g].map(i => F[i]).filter(f => f.stats.peak_dn !== null);
      const above = members.filter(f => f.stats.peak_dn >= satLevel).length;
      return `${name} ${above}/${members.length}`;
    });
    parts.push('frames reaching the level: ' + counts.join(', '));
  }
  parts.push(`config <code>saturation_dn: ${satLevel === null ? 'null' : satLevel}</code>`);
  $('sat-info').innerHTML = parts.join(' · ');
}
function setSatLevel(value) {
  satLevel = value;
  $('sat-level').value = value === null ? '' : value;
  describeSaturation();
  drawPlots();
}
function saturationCsv() {
  const lines = [DATA.saturationColumns.join(',')];
  G.forEach(name => {
    const c = SAT.channels[name] || {};
    lines.push([name, c.ceiling_dn, c.saturation_dn, c.saturated_frames, c.recommended_dn,
      satLevel === null ? 'none' : satLevel].map(csvField).join(','));
  });
  return lines.join('\n') + '\n';
}
function saveSaturation() {
  if (!SERVED) { download('saturation.csv', saturationCsv()); return; }
  fetch('saturation.csv', {method: 'POST', body: saturationCsv(), headers: {'Content-Type': 'text/csv'}})
    .then(r => r.ok ? r.json() : Promise.reject(r.statusText))
    .then(info => { $('sat-status').textContent = `saved saturation.csv · ${info.selected_dn}`; $('sat-status').className = 'ok'; })
    .catch(error => { $('sat-status').textContent = 'not saved: ' + error; $('sat-status').className = 'err'; });
}

// ---------------------------------------------------------------- plots
const canvases = [];
function buildPlots() {
  const container = $('plots');
  G.forEach((name, g) => {
    const row = document.createElement('div');
    row.className = 'plot-row';
    row.innerHTML = `<div class="label" title="${name}">${name}</div>`;
    const canvas = document.createElement('canvas');
    row.appendChild(canvas);
    container.appendChild(row);
    canvases.push(canvas);
    attachPlotEvents(canvas, g);
  });
  DATA.metrics.forEach(([key, label]) => {
    const option = document.createElement('option');
    option.value = key; option.textContent = label;
    $('metric').appendChild(option);
  });
}
function yRange(values) {
  const sorted = values.filter(v => v !== null && isFinite(v)).sort((a, b) => a - b);
  if (!sorted.length) return [0, 1];
  let lo = sorted[Math.floor(0.005 * (sorted.length - 1))];
  let hi = sorted[Math.ceil(0.995 * (sorted.length - 1))];
  if (hi <= lo) { lo -= 1; hi += 1; }
  const pad = 0.08 * (hi - lo);
  return [lo - pad, hi + pad];
}
function drawPlots() {
  const dpr = window.devicePixelRatio || 1;
  canvases.forEach((canvas, g) => {
    const width = canvas.clientWidth, height = canvas.clientHeight;
    canvas.width = width * dpr; canvas.height = height * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);
    const members = byGroup[g].map(i => F[i]);
    let [lo, hi] = yRange(members.map(f => f.stats[metric]));
    const channelSaturation = (SAT.channels[G[g]] || {}).saturation_dn;
    if (metric === 'peak_dn') {
      // Keep the selected level and the detected saturation in view.
      [satLevel, channelSaturation].forEach(v => {
        if (v) { lo = Math.min(lo, v * 0.95); hi = Math.max(hi, v * 1.03); }
      });
    }
    const x = t => (t - view[0]) / (view[1] - view[0]) * width;
    const y = v => height - 4 - (Math.min(Math.max(v, lo), hi) - lo) / (hi - lo) * (height - 8);
    ctx.strokeStyle = '#2f3540'; ctx.lineWidth = 1;
    ctx.beginPath();
    let started = false;
    members.forEach(f => {
      const v = f.stats[metric];
      if (f.t === null || v === null) { started = false; return; }
      if (!started) { ctx.moveTo(x(f.t), y(v)); started = true; } else ctx.lineTo(x(f.t), y(v));
    });
    ctx.stroke();
    const layers = [['ok', '#6b7280', 1.6], ['warn', '#f0a020', 2.4], ['rej', '#e5484d', 3]];
    layers.forEach(([kind, color, radius]) => {
      ctx.fillStyle = color;
      members.forEach(f => {
        const v = f.stats[metric];
        if (f.t === null || v === null) return;
        const k = isRejected(f) ? 'rej' : (f.auto === 'warn' ? 'warn' : 'ok');
        if (k !== kind) return;
        const px = x(f.t);
        if (px < -4 || px > width + 4) return;
        ctx.beginPath(); ctx.arc(px, y(v), radius, 0, 2 * Math.PI); ctx.fill();
      });
    });
    const s = F[selected];
    if (s && s.t !== null) {
      ctx.strokeStyle = '#4cc2ff'; ctx.globalAlpha = 0.7;
      ctx.beginPath(); ctx.moveTo(x(s.t), 0); ctx.lineTo(x(s.t), height); ctx.stroke();
      ctx.globalAlpha = 1;
      if (s.g === g && s.stats[metric] !== null) {
        ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(x(s.t), y(s.stats[metric]), 5, 0, 2 * Math.PI); ctx.stroke();
      }
    }
    if (metric === 'peak_dn') {
      const line = (value, color, dash, label) => {
        if (!value) return;
        ctx.save(); ctx.strokeStyle = color; ctx.setLineDash(dash); ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(0, y(value)); ctx.lineTo(width, y(value)); ctx.stroke();
        ctx.fillStyle = color; ctx.font = '10px sans-serif';
        ctx.fillText(label, width - ctx.measureText(label).width - 4, y(value) - 3);
        ctx.restore();
      };
      line(channelSaturation, '#b8c4d6', [2, 3], `saturation ${fmt(channelSaturation)}`);
      line(satLevel, '#e5484d', [6, 4], `level ${fmt(satLevel)}`);
    }
    ctx.fillStyle = '#8b929e'; ctx.font = '10px sans-serif';
    ctx.fillText(fmt(hi), 3, 10); ctx.fillText(fmt(lo), 3, height - 3);
  });
  $('t0').textContent = new Date(view[0]).toISOString().slice(0, 16).replace('T', ' ');
  $('t1').textContent = new Date(view[1]).toISOString().slice(0, 16).replace('T', ' ');
}
function attachPlotEvents(canvas, g) {
  let drag = null;
  canvas.addEventListener('mousedown', e => { drag = {x: e.clientX, view: view.slice(), moved: false}; });
  window.addEventListener('mousemove', e => {
    if (!drag) return;
    const dx = e.clientX - drag.x;
    if (Math.abs(dx) > 3) drag.moved = true;
    if (!drag.moved) return;
    const span = drag.view[1] - drag.view[0];
    const shift = -dx / canvas.clientWidth * span;
    view = [drag.view[0] + shift, drag.view[1] + shift];
    drawPlots();
  });
  window.addEventListener('mouseup', e => {
    if (!drag) return;
    const wasClick = !drag.moved && e.target === canvas;
    drag = null;
    if (!wasClick) return;
    const rect = canvas.getBoundingClientRect();
    const t = view[0] + (e.clientX - rect.left) / rect.width * (view[1] - view[0]);
    let best = null, distance = Infinity;
    byGroup[g].forEach(i => {
      if (F[i].t === null) return;
      const d = Math.abs(F[i].t - t);
      if (d < distance) { distance = d; best = i; }
    });
    if (best !== null) select(best, true);
  });
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const t = view[0] + (e.clientX - rect.left) / rect.width * (view[1] - view[0]);
    const factor = Math.exp(e.deltaY * 0.0015);
    const minimum = 60e3;
    let a = t - (t - view[0]) * factor, b = t + (view[1] - t) * factor;
    if (b - a < minimum) { a = t - minimum / 2; b = t + minimum / 2; }
    view = [a, b];
    drawPlots();
  }, {passive: false});
  canvas.addEventListener('dblclick', () => { view = [T0 - 1, T1 + 1]; drawPlots(); });
}

// ---------------------------------------------------------------- viewer
// One panel per channel; channels without a frame at this time stay blank.
function stripHtml(r) {
  if (r < 0 || r >= rows.length) return G.map(() => '<div class="vcell blank"></div>').join('');
  return rows[r].map((i, g) => {
    if (i === null) return `<div class="vcell blank" title="no ${G[g]}">no ${G[g]}</div>`;
    const f = F[i];
    const cls = (isRejected(f) ? ' rej' : (f.auto === 'warn' ? ' warn' : '')) + (i === selected ? ' sel' : '');
    const src = f.preview || f.thumb;
    const image = src ? `<img src="${src}" alt="">` : '';
    const overlay = masks[f.path] ? `<img class="mask-overlay" src="${masks[f.path]}" alt="Excluded pixels">` : '';
    const aspect = f.width && f.height ? ` style="--aspect:${f.width / f.height}"` : '';
    return `<div class="vcell${cls}"${aspect} data-i="${i}" title="${G[g]} ${timeLabel(f)}">${image}${overlay}` +
      `<span class="lbl">${G[g]}</span></div>`;
  }).join('');
}
function rowLabel(r) {
  if (r < 0 || r >= rows.length) return '';
  const first = rows[r].find(i => i !== null);
  return first === undefined ? '' : timeLabel(F[first]);
}
function updateViewer() {
  const f = F[selected];
  if (!f) return;
  const list = byGroup[f.g];
  ['vprev', 'vcur', 'vnext'].forEach(id => $(id).style.setProperty('--n', G.length));
  $('vprev').innerHTML = stripHtml(f.row - 1);
  $('vcur').innerHTML = stripHtml(f.row);
  $('vnext').innerHTML = stripHtml(f.row + 1);
  $('cprev').textContent = f.row > 0 ? '▲ ' + rowLabel(f.row - 1) : 'start of sequence';
  $('ccur').textContent = `${rowLabel(f.row)} · observation ${f.row + 1}/${rows.length} · ` +
    `selected ${G[f.g]} (${f.k + 1}/${list.length})` +
    (f.tbBackgroundSubtracted ? ' · daily background subtracted' : '');
  $('cnext').textContent = f.row < rows.length - 1 ? '▼ ' + rowLabel(f.row + 1) : 'end of sequence';
  const badge = $('badge');
  badge.className = 'badge ' + f.auto;
  badge.textContent = 'auto: ' + f.auto;
  $('decision').innerHTML = isRejected(f)
    ? '<b style="color:var(--reject)">rejected</b>' + (isChanged(f) ? ' (manual)' : '')
    : '<b>kept</b>' + (isChanged(f) ? ' (manual)' : '');
  $('reasons').innerHTML = f.reasons.length
    ? f.reasons.map((r, n) => `<li class="${n < f.nr ? 'r' : 'w'}">${r}</li>`).join('')
    : '<li style="color:var(--muted)">no automatic findings</li>';
  $('toggle').innerHTML = (isRejected(f) ? 'Keep frame' : 'Reject frame') + ' <kbd>x</kbd>';
  $('path').textContent = f.path;
  if (DATA.cme) updateMaskEditor(f);
  const labels = Object.fromEntries(DATA.metrics);
  Object.assign(labels, {nsatpix: 'Saturated pixels', quality: 'QUALITY', aectype: 'AECTYPE',
    low: '1st percentile', high: '99th percentile', maximum: 'Maximum'});
  $('stats').innerHTML = Object.entries(f.stats)
    .map(([key, value]) => `<tr><td>${labels[key] || key}</td><td>${fmt(value)}</td></tr>`).join('');
}

// The editing canvas has exactly one pixel per source pixel, in JPEG display order.
const canvas = $('mask-canvas'), ink = canvas.getContext('2d');
let stroke = null, editorReady = false, editorFrame = null, editorSources = [], editorVersion = 0;
let applyingLater = false;
function maskTargets(f) {
  const partner = F[f.partner];
  return $('mask-pair').checked && partner ? [f, partner] : [f];
}
function maskSources(f) {
  return [...new Set(maskTargets(f).map(target => masks[target.path]).filter(Boolean))];
}
async function loadLayers(sources) {
  return Promise.all(sources.map(async src => {
    const image = new Image();
    image.src = src;
    await image.decode();
    return image;
  }));
}
function applyMask(f, image) {
  for (const target of maskTargets(f)) {
    if (image) {
      masks[target.path] = image;
      editing.add(target.path);
    } else {
      editing.delete(target.path);
      if (!masks[target.path]) continue;
      delete masks[target.path];
    }
    saveMask(target, image);
  }
}

function previousMask(f) {
  for (let k = f.k - 1; k >= 0; k--) {
    const previous = F[byGroup[f.g][k]];
    if (previous.width === f.width && previous.height === f.height &&
        maskTargets(previous).some(target => masks[target.path])) return previous;
  }
  return null;
}
function updateMaskEditor(f) {
  const active = editing.has(f.path) && !!f.preview;
  $('mask-controls').hidden = false;
  $('mask-enable').disabled = !SERVED || !f.preview || applyingLater;
  $('mask-enable').checked = active;
  $('mask-pair-control').hidden = !pairedFolder(f);
  $('mask-pair').disabled = applyingLater;
  $('mask-radius').disabled = !active;
  $('mask-duplicate').disabled = !SERVED || !f.preview || applyingLater;
  $('mask-later').disabled = !SERVED || !active || applyingLater || !maskSources(f).length;
  $('mask-duplicate').title = 'Enable editing and copy the most recent earlier mask on this grid, if available';
  canvas.hidden = !active;
  const hint = masks[f.path] ? 'Uncheck to clear the mask' : 'Drag to exclude pixels';
  $('mask-status').textContent = SERVED ? (active ? hint : '') : 'Run frame_review review to edit masks';
  if (!active) { editorFrame = null; editorReady = false; editorVersion++; return; }
  const panel = $('vcur').querySelector(`[data-i="${f.i}"]`);
  const overlay = panel.querySelector('.mask-overlay');
  if (overlay) overlay.remove();
  if (canvas.parentElement !== panel) panel.appendChild(canvas);
  const sources = maskSources(f);
  if (editorFrame === f && sources.length === editorSources.length && sources.every((src, i) => src === editorSources[i])) return;
  editorFrame = f;
  editorSources = sources;
  const version = ++editorVersion;
  editorReady = !sources.length;
  canvas.width = f.width;
  canvas.height = f.height;
  if (sources.length) {
    loadLayers(sources).then(images => {
      if (version !== editorVersion) return;
      // Painting adds to the union of any existing channel-specific exclusions.
      images.forEach(image => ink.drawImage(image, 0, 0));
      editorReady = true;
    }).catch(error => {
      if (version === editorVersion) $('mask-save').textContent = 'Could not load mask for ' + f.name + ': ' + error;
    });
  }
}
$('mask-pair').addEventListener('change', () => {
  editorFrame = null;
  updateMaskEditor(F[selected]);
});
$('mask-enable').addEventListener('change', e => {
  const f = F[selected];
  if (e.target.checked) {
    editing.add(f.path);
  } else {
    applyMask(f, null);
  }
  refresh();
});
const pendingMasks = new Map();
let maskWrites = Promise.resolve();
function saveMask(f, image) {
  const change = {image};
  pendingMasks.set(f.path, change);
  $('mask-save').textContent = 'Saving mask…';
  maskWrites = maskWrites.then(async () => {
    if (pendingMasks.get(f.path) !== change) return;
    const body = image ? await (await fetch(image)).blob() : null;
    const response = await fetch('pixel-mask/' + encodeURIComponent(f.maskFile), {method: 'POST', body});
    if (!response.ok) throw Error(response.statusText);
    if (pendingMasks.get(f.path) === change) pendingMasks.delete(f.path);
    if (!pendingMasks.size) $('mask-save').textContent = image ? 'Saved .npy mask' : 'Removed .npy mask';
  }).catch(error => { $('mask-save').textContent = 'Mask save failed for ' + f.name + ': ' + error; });
}
async function editWithPreviousMask() {
  const f = F[selected];
  if (!DATA.cme || !SERVED || !f.preview || stroke || applyingLater) return;
  editing.add(f.path);
  updateMaskEditor(f);
  const previous = previousMask(f);
  if (!previous) return;
  const version = ++editorVersion;
  editorReady = false;
  try {
    const images = await loadLayers(maskSources(previous));
    if (version !== editorVersion) return;
    ink.clearRect(0, 0, canvas.width, canvas.height);
    images.forEach(image => ink.drawImage(image, 0, 0));
    const image = canvas.toDataURL();
    applyMask(f, image);
    editorSources = [image];
    editorReady = true;
    refresh();
  } catch (error) {
    if (version !== editorVersion) return;
    editorFrame = null;
    updateMaskEditor(f);
    $('mask-save').textContent = 'Could not duplicate mask: ' + error;
  }
}
$('mask-duplicate').addEventListener('click', editWithPreviousMask);
$('mask-later').addEventListener('click', async () => {
  const f = F[selected];
  const sources = maskTargets(f).filter(target => masks[target.path]).map(target => target.maskFile);
  const channels = new Set([channelFolder(f)]);
  if ($('mask-pair').checked) channels.add(pairedFolder(f));
  const targets = F.filter(target => target.t > f.t && target.i !== f.partner &&
    target.preview && target.width === f.width && target.height === f.height && channels.has(channelFolder(target)));
  const status = $('mask-save');
  if (!targets.length) { status.textContent = 'No later frames on this grid'; return; }
  applyingLater = true;
  updateMaskEditor(f);
  try {
    await maskWrites;
    if (pendingMasks.size) throw Error('A previous mask edit still needs to be saved.');
    status.textContent = `Applying mask to ${targets.length} frames…`;
    const response = await fetch('merge-masks', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({sources, targets: targets.map(target => target.maskFile)})});
    if (!response.ok) throw Error(response.statusText);
    const result = await response.json();
    const applied = new Set(result.applied);
    targets.forEach(target => {
      if (applied.has(target.maskFile)) {
        masks[target.path] = 'mask-preview/' + encodeURIComponent(target.maskFile);
        editing.add(target.path);
      }
    });
    status.textContent = `Applied mask to ${applied.size} later frames` + (result.error ? '. Stopped at ' + result.error : '');
  } catch (error) {
    status.textContent = 'Could not apply mask to later frames: ' + error;
  } finally {
    applyingLater = false;
    editorFrame = null;
    refresh();
  }
});
function stamp(x, y, r) {
  // Integer spans keep the alpha plane strictly binary, including along the rim.
  ink.fillStyle = '#ff496c';
  for (let dy = -r; dy <= r; dy++) {
    const dx = Math.floor(Math.sqrt(r * r - dy * dy));
    ink.fillRect(Math.round(x) - dx, Math.round(y) + dy, 2 * dx + 1, 1);
  }
}
function paint(e) {
  if (!stroke || stroke.pointer !== e.pointerId) return;
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) * canvas.width / rect.width - 0.5;
  const y = (e.clientY - rect.top) * canvas.height / rect.height - 0.5;
  const r = Number($('mask-radius').value);
  const dx = x - (stroke.x ?? x), dy = y - (stroke.y ?? y);
  const n = Math.max(1, Math.ceil(Math.hypot(dx, dy) / Math.max(1, r / 2)));
  for (let i = 1; i <= n; i++) stamp(x - dx * (1 - i/n), y - dy * (1 - i/n), r);
  stroke.x = x; stroke.y = y;
}
function finishStroke() {
  if (!stroke) return;
  const f = stroke.frame;
  stroke = null;
  const image = canvas.toDataURL();
  applyMask(f, image);
  editorSources = [image];
  refresh();
}
canvas.addEventListener('pointerdown', e => {
  if (!editorReady || stroke || applyingLater || e.button !== 0 || !editing.has(F[selected].path)) return;
  canvas.setPointerCapture(e.pointerId);
  stroke = {frame: F[selected], pointer: e.pointerId};
  paint(e);
});
canvas.addEventListener('pointermove', paint);
canvas.addEventListener('pointerup', e => { paint(e); finishStroke(); });
canvas.addEventListener('lostpointercapture', finishStroke);

// ---------------------------------------------------------------- selection
function select(i, scroll) {
  if (i === null || i === undefined || !F[i]) return;
  finishStroke();
  const previous = selected;
  selected = i;
  if (cells[previous]) cells[previous].classList.remove('sel');
  if (cells[i]) {
    cells[i].classList.add('sel');
    if (scroll) cells[i].scrollIntoView({block: 'nearest', container: 'nearest'});
  }
  const t = F[i].t;
  if (t !== null && (t < view[0] || t > view[1])) {
    const span = view[1] - view[0];
    view = [t - span / 2, t + span / 2];
  }
  updateViewer();
  drawPlots();
}
function refresh() { updateCounts(); updateGrid(); updateViewer(); drawPlots(); }
function stepFrame(delta) {
  const f = F[selected], list = byGroup[f.g];
  const k = Math.min(Math.max(f.k + delta, 0), list.length - 1);
  select(list[k], true);
}
function stepFlagged(delta) {
  const f = F[selected], list = byGroup[f.g];
  for (let k = f.k + delta; k >= 0 && k < list.length; k += delta) {
    if (flagged(F[list[k]])) { select(list[k], true); return; }
  }
}
function stepRow(delta) {
  const f = F[selected];
  for (let r = f.row + delta; r >= 0 && r < rows.length; r += delta) {
    if (!rowVisible(r)) continue;
    const row = rows[r];
    let i = row[f.g];
    if (i === null) i = row.find(x => x !== null);
    if (i !== null && i !== undefined) { select(i, true); return; }
  }
}
function toggleFrame() { const f = F[selected]; setRejected(f, !isRejected(f)); save(); refresh(); }
function toggleRow() {
  const members = rows[F[selected].row].filter(i => i !== null).map(i => F[i]);
  const value = !members.every(isRejected);
  members.forEach(f => setRejected(f, value));
  save(); refresh();
}

function importDecisions(rows) {
  const p = rows[0].indexOf('path'), d = rows[0].indexOf('decision');
  fromRows(rows.slice(1).map(r => [r[p], (r[d] || '').trim().toLowerCase()]));
}

// ---------------------------------------------------------------- export / import
function download(name, text) {
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([text], {type: 'text/plain'}));
  link.download = name;
  document.body.appendChild(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}
$('export').addEventListener('click', () => download('decisions.csv', decisionsCsv()));
$('import').addEventListener('click', () => $('import-file').click());
$('import-file').addEventListener('change', e => {
  const file = e.target.files[0];
  if (!file) return;
  file.text().then(text => {
    const rows = parseCsv(text);
    const header = rows.length ? rows[0].map(h => h.trim()) : [];
    const pathIndex = header.indexOf('path'), decisionIndex = header.indexOf('decision');
    if (pathIndex >= 0 && decisionIndex >= 0) {
      importDecisions(rows);
    } else {
      // A list of rejected paths (or file names), one per line.
      const listed = new Set(text.split(/\r?\n/).map(s => s.trim()).filter(s => s && !s.startsWith('#')));
      const names = new Set([...listed].map(p => p.split('/').pop()));
      decisions = {};
      F.forEach(f => setRejected(f, listed.has(f.path) || names.has(f.name)));
    }
    save(); refresh();
    e.target.value = '';
  });
});
$('accept-all').addEventListener('click', () => {
  F.forEach(f => setRejected(f, false));
  save(); refresh();
});
$('reset').addEventListener('click', () => {
  if (!Object.keys(decisions).length || confirm('Discard all manual decisions?')) {
    decisions = {}; save(); refresh();
  }
});
$('filter').addEventListener('change', () => { updateGrid(); });
$('metric').addEventListener('change', e => { metric = e.target.value; drawPlots(); });
$('sat-level').addEventListener('input', e => {
  const value = parseLevel(e.target.value);
  satLevel = value === undefined ? null : value;
  describeSaturation(); drawPlots();
});
$('sat-level').addEventListener('focus', () => {
  if (HAS_PEAKS && metric !== 'peak_dn') { metric = 'peak_dn'; $('metric').value = metric; drawPlots(); }
});
$('sat-recommended').addEventListener('click', () => setSatLevel(SAT.recommended_dn ?? null));
$('sat-save').addEventListener('click', saveSaturation);
$('viewer').addEventListener('click', e => {
  if (e.target === canvas) return;
  const cell = e.target.closest('[data-i]');
  if (cell) select(Number(cell.dataset.i), true);
});
$('viewer').addEventListener('dblclick', e => {
  if (e.target === canvas) return;
  const cell = e.target.closest('[data-i]');
  if (cell) { select(Number(cell.dataset.i), true); toggleFrame(); }
});
$('toggle').addEventListener('click', toggleFrame);
$('toggle-row').addEventListener('click', toggleRow);
document.addEventListener('keydown', e => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (DATA.cme && e.key.toLowerCase() === 'm' &&
      (e.target.tagName !== 'INPUT' || e.target.type === 'checkbox') && e.target.tagName !== 'SELECT') {
    e.preventDefault();
    if (!e.repeat) editWithPreviousMask();
    return;
  }
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
  const actions = {
    ArrowRight: () => stepFrame(1), l: () => stepFrame(1),
    ArrowLeft: () => stepFrame(-1), h: () => stepFrame(-1),
    ArrowDown: () => stepRow(1), j: () => stepRow(1),
    ArrowUp: () => stepRow(-1), k: () => stepRow(-1),
    n: () => stepFlagged(1), p: () => stepFlagged(-1),
    x: toggleFrame, ' ': toggleFrame, r: toggleRow,
  };
  if (actions[e.key]) { e.preventDefault(); actions[e.key](); }
});
window.addEventListener('resize', drawPlots);

document.title = 'Frame review · ' + DATA.title;
if (DATA.cme) {
  $('saturation-bar').style.display = 'none';
  document.querySelector('.keys').lastChild.textContent = ' toggle row. Green marks invalid pixels. The mask editor uses native pixels. Upload decisions.csv and the masks directory, then apply the review on the server.';
} else if (HAS_PEAKS) {
  $('sat-level').value = satLevel === null ? '' : satLevel;
  $('sat-recommended').disabled = !SAT.recommended_dn;
  describeSaturation();
} else {
  $('saturation-bar').innerHTML = '<span style="color:var(--muted)">Saturation level: no peak ' +
    'detector DN in this scan (prepare the products again to record PEAKDN, or use ' +
    '<code>scan --source-dir</code> with the ' +
    'preparation inputs).</span>';
}
buildPlots();
buildGrid();
refresh();
select(selected, true);
if (SERVED) {
  // The page may be older than decisions.csv (saved since it was built).
  fetch('decisions.csv', {cache: 'no-store'})
    .then(r => r.ok ? r.text() : null)
    .then(text => {
      if (revision) return;
      if (text) {
        importDecisions(parseCsv(text));
        refresh();
        setStatus(`loaded decisions.csv · ${F.filter(isRejected).length} rejected`, 'ok');
      } else setStatus('no decisions.csv yet: automatic flags', '');
    })
    .catch(() => setStatus('no decisions.csv yet: automatic flags', ''));
} else if (Object.keys(decisions).length) save();
else setStatus('opened as a file: run "frame_review review" to save decisions.csv', '');
</script>
</body>
</html>
"""
