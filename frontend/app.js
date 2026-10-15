/* Landing Control - browser viewer for the rocket-landing RL project.
   Vanilla JS, canvas 2D. Talks to /api/* or, with ?fixture=1, to ./fixtures/*.json. */
(() => {
'use strict';

/* ---------- helpers ---------- */
const $ = (id) => document.getElementById(id);
const qs = new URLSearchParams(location.search);
const FIXTURE = qs.get('fixture') === '1';
const FIXLIVE = FIXTURE && qs.get('live') === '1';
const STATIC_FORCED = qs.get('static') === '1', STATIC_OFF = qs.get('static') === '0';
let STATIC = false; // set at boot: ?static=1, or /api unreachable but ./demo/ exists
let MANIFEST = null;
const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const lerp = (a, b, t) => a + (b - a) * t;
const isNum = (v) => typeof v === 'number' && isFinite(v);
const fmt = (v, d = 2) => isNum(v) ? v.toFixed(d) : '—';
const sgn = (v, d = 2) => isNum(v) ? (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d) : '—';
const pct = (v, d = 0) => isNum(v) ? (v * 100).toFixed(d) + '%' : '—';
const kfmt = (v) => !isNum(v) ? '—' : v >= 1e6 ? (v / 1e6).toFixed(2) + 'M' : v >= 1e4 ? (v / 1e3).toFixed(1) + 'k' : String(Math.round(v));
const angDiff = (a, b) => { let d = b - a; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI; return d; };
const ACTIONS = ['NOOP', 'MAIN', 'LEFT', 'RIGHT'];

/* ---------- API layer (live or fixture) ---------- */
class HttpError extends Error { constructor(status, msg) { super(msg); this.status = status; } }
const t0 = performance.now();
const fixCache = {};
async function getJSON(url) {
  let r;
  try { r = await fetch(url, { cache: 'no-store' }); }
  catch (e) { throw new HttpError(0, 'network error'); }
  if (!r.ok) throw new HttpError(r.status, r.status + ' ' + r.statusText);
  try { return await r.json(); } catch (e) { throw new HttpError(500, 'invalid JSON'); }
}
const DEMO = 'demo/'; // relative on purpose: works under a GitHub Pages sub-path
async function demo(name) { return fixCache['d:' + name] || (fixCache['d:' + name] = await getJSON(DEMO + name + '.json')); }
const clone = (o) => JSON.parse(JSON.stringify(o));
async function detectStatic() {
  if (STATIC_OFF || FIXTURE) return;
  if (!STATIC_FORCED) {
    try { const r = await fetch('/api/runs', { cache: 'no-store' }); if (r.ok && /json/.test(r.headers.get('content-type') || '')) return; } catch (e) { /* unreachable */ }
  }
  try { MANIFEST = await getJSON(DEMO + 'manifest.json'); STATIC = true; } catch (e) { /* no demo data: stay in live mode and show its normal errors */ }
}
async function fix(name) { return fixCache[name] || (fixCache[name] = await getJSON('fixtures/' + name + '.json')); }
function fixRevealed() { return FIXLIVE ? clamp(24 + Math.floor((performance.now() - t0) / 2500), 24, 60) : 60; }
const api = {
  async runs() {
    if (STATIC) return clone(await demo('runs'));
    if (!FIXTURE) return getJSON('/api/runs');
    const runs = JSON.parse(JSON.stringify(await fix('runs')));
    const n = fixRevealed();
    runs[0].mtime = Date.now() / 1000 - 3; runs[0].updates = n; runs[0].global_step = n * 4096; runs[0].finished = FIXLIVE ? false : false;
    if (n >= 60 && FIXLIVE) runs[0].finished = true;
    return runs;
  },
  async metrics(run) {
    if (STATIC) return clone(await demo('metrics_' + run));
    if (!FIXTURE) return getJSON('/api/runs/' + encodeURIComponent(run) + '/metrics');
    if (run !== 'ppo_v2_seed1') return { train: [], eval: [], config: null };
    const m = await fix('metrics'), n = fixRevealed();
    return { train: m.train.filter(r => r.update <= n), eval: m.eval.filter(r => r.update <= n), config: m.config };
  },
  async checkpoints(run) {
    if (STATIC) return clone(await demo('checkpoints_' + run));
    if (!FIXTURE) return getJSON('/api/runs/' + encodeURIComponent(run) + '/checkpoints');
    if (run !== 'ppo_v2_seed1') throw new HttpError(404, 'no checkpoints');
    const n = fixRevealed();
    return (await fix('checkpoints')).map(c => c.id === 'latest' || c.id === 'best' || c.id === 'ckpt_00060' ? Object.assign({}, c, { update: n }) : c).filter(c => c.id !== 'ckpt_00060' || n >= 60).filter(c => c.update == null || c.update <= n);
  },
  async episode(run, ck, seed) {
    if (STATIC) {
      if (!MANIFEST.seeds.includes(+seed)) throw new HttpError(404, 'seed ' + seed + ' is not in the recorded demo');
      return clone(await demo('episode_' + run + '__' + ck + '__' + seed));
    }
    if (!FIXTURE) return getJSON('/api/runs/' + encodeURIComponent(run) + '/episode?seed=' + seed + '&checkpoint=' + encodeURIComponent(ck));
    if (run !== 'ppo_v2_seed1') throw new HttpError(404, 'no checkpoint');
    const n = fixRevealed();
    const m = /(\d+)$/.exec(ck);
    const upd = ck === 'latest' || ck === 'best' || ck === 'final' ? n : m ? +m[1] : n;
    const ep = JSON.parse(JSON.stringify(await fix(upd < 35 ? 'episode_early' : 'episode_latest')));
    ep.meta.checkpoint = ck; ep.meta.update = upd; ep.meta.run = run;
    return ep;
  },
  async baseline(seed) {
    if (STATIC) return clone(await demo('baseline_' + seed));
    if (!FIXTURE) return getJSON('/api/baseline/episode?seed=' + seed);
    return JSON.parse(JSON.stringify(await fix('baseline')));
  },
};

/* ---------- state ---------- */
const S = {
  runs: [], run: null, userPickedRun: false,
  metrics: null, metricsSig: '', checkpoints: [], ck: 'latest', seed: 0,
  ep: null, pending: null, base: null, compare: false,
  tp: null, playing: !REDUCED, speed: 1, f: 0, endHold: 0,
  lastLatestUpdate: undefined, stepSamples: [],
  connOK: null, lastOK: 0, epWaiting: false, epToken: 0, epError: null,
  cam: null, trail: [], particles: [], lastTeleIdx: -1, hoverX: {},
};

/* ---------- connection / banner ---------- */
function setConn(ok, msg) {
  const el = $('conn'); S.connOK = ok;
  if (STATIC) { el.className = 'conn fx'; $('connText').textContent = 'recorded demo · not live'; return; }
  if (FIXTURE) { el.className = 'conn fx'; $('connText').textContent = FIXLIVE ? 'fixture · simulated live' : 'fixture data'; return; }
  el.className = 'conn ' + (ok ? 'ok' : 'bad');
  $('connText').textContent = ok ? 'API online' : (msg || 'backend unreachable');
  const b = $('banner');
  if (!ok) { b.hidden = false; b.className = 'banner'; b.textContent = 'Cannot reach the backend (' + (msg || 'network error') + '). Retrying every 2 s - start it with the viewer server and this page will recover on its own.'; }
  else b.hidden = true;
}

/* ---------- runs ---------- */
function liveState(r) {
  if (STATIC && r) return { cls: 'badge-done', txt: 'RECORDED', title: 'A recorded training run - not live' };
  if (!r) return { cls: 'badge-idle', txt: '—', title: '' };
  if (r.finished) return { cls: 'badge-done', txt: 'FINISHED', title: 'Training run finished' };
  const age = Date.now() / 1000 - r.mtime;
  if (age < 90) return { cls: 'badge-live', txt: 'LIVE', title: 'Updated ' + Math.max(0, Math.round(age)) + ' s ago' };
  return { cls: 'badge-idle', txt: 'IDLE', title: 'No update for ' + Math.round(age / 60) + ' min (stopped or paused?)' };
}
function curRun() { return S.runs.find(r => r.name === S.run) || null; }

function renderRunSelect() {
  const sel = $('runSelect');
  const sig = S.runs.map(r => r.name).join('|');
  if (sel.dataset.sig !== sig) {
    sel.dataset.sig = sig;
    sel.innerHTML = '';
    for (const r of S.runs) { const o = document.createElement('option'); o.value = r.name; o.textContent = r.name + (r.reward_version ? '  ·  ' + r.reward_version : ''); sel.appendChild(o); }
  }
  if (S.run) sel.value = S.run;
  const r = curRun(), l = liveState(r), b = $('liveBadge');
  b.className = 'badge ' + l.cls; b.textContent = l.txt; b.title = l.title;
  const frac = r && r.total_timesteps ? clamp(r.global_step / r.total_timesteps, 0, 1) : 0;
  $('progressFill').style.width = (frac * 100) + '%';
  $('progressText').textContent = r ? (kfmt(r.global_step) + ' / ' + kfmt(r.total_timesteps) + ' steps · ' + (frac * 100).toFixed(0) + '%') : '—';
}

async function pollRuns() {
  try {
    const runs = await api.runs();
    setConn(true); S.lastOK = Date.now();
    S.runs = Array.isArray(runs) ? runs : [];
    if (!S.runs.length) { showOverlay('No training runs yet', 'Start a training run and it will appear here automatically.'); renderRunSelect(); return; }
    if (!S.run || !S.runs.some(r => r.name === S.run)) {
      selectRun(S.runs[0].name, false);
    } else if (!S.userPickedRun) {
      // follow the newest run until the user chooses one explicitly
      const newest = S.runs.slice().sort((a, b) => b.mtime - a.mtime)[0];
      if (newest.name !== S.run) selectRun(newest.name, false);
    }
    const r = curRun();
    if (r) {
      S.stepSamples.push([performance.now() / 1000, r.global_step]);
      S.stepSamples = S.stepSamples.filter(s => s[0] > performance.now() / 1000 - 30);
    }
    renderRunSelect();
    await pollRun();
  } catch (e) {
    setConn(false, e.status ? 'runs request failed: ' + e.message : e.message);
    if (!S.run) showOverlay('Backend unreachable', 'Waiting for the API to come online...', true);
  }
}

function selectRun(name, user) {
  if (user) S.userPickedRun = true;
  S.run = name; S.metrics = null; S.metricsSig = ''; S.checkpoints = []; S.ck = 'latest';
  S.stepSamples = []; S.lastLatestUpdate = undefined; S.ep = null; S.pending = null; S.base = null; S.f = 0; S.trail = []; S.particles = [];
  S.epError = null;
  const r = curRun(); if (r && isNum(r.seed) && S.seed === 0) { /* keep seed 0 default so successive runs compare */ }
  renderRunSelect(); renderCkptSelect(); renderTelemetry(true); renderTraining(true);
  showOverlay('Loading run', name);
  pollRun(true);
}

async function pollRun(force) {
  const run = S.run; if (!run) return;
  const [m, c] = await Promise.allSettled([api.metrics(run), api.checkpoints(run)]);
  if (run !== S.run) return;
  if (m.status === 'fulfilled') {
    const d = m.value || {}; S.metrics = { train: d.train || [], eval: d.eval || [], config: d.config || null };
    const sig = S.metrics.train.length + ':' + S.metrics.eval.length + ':' + (S.metrics.train.length ? S.metrics.train[S.metrics.train.length - 1].global_step : 0);
    if (sig !== S.metricsSig || force) { S.metricsSig = sig; renderTraining(); }
  }
  let latestUpd;
  if (c.status === 'fulfilled' && Array.isArray(c.value)) {
    S.checkpoints = c.value; renderCkptSelect();
    const l = c.value.find(x => x.id === 'latest'); latestUpd = l ? (l.update != null ? l.update : l.mtime) : undefined;
  } else if (c.status === 'rejected' && c.reason.status === 404) { S.checkpoints = []; renderCkptSelect(); }
  const has = S.checkpoints.length > 0;
  if (!has && !S.ep) showOverlay('Waiting for first checkpoint', 'The run has not saved a policy yet - this view will start as soon as it does.', false, true);
  if (has && S.ck === 'latest' && latestUpd !== S.lastLatestUpdate) { S.lastLatestUpdate = latestUpd; fetchEpisode(!S.ep); }
  else if (has && !S.ep && !S.epWaiting && !S.epError) fetchEpisode(true);
  renderStats();
}

/* ---------- checkpoints select ---------- */
function renderCkptSelect() {
  const sel = $('ckptSelect');
  const sig = S.checkpoints.map(c => c.id + ':' + c.update).join('|') + '#' + S.ck;
  if (sel.dataset.sig === sig) return; sel.dataset.sig = sig;
  sel.innerHTML = '';
  const list = S.checkpoints.length ? S.checkpoints : [{ id: 'latest', update: null }];
  const rank = (id) => id === 'latest' ? 0 : id === 'best' ? 1 : id === 'final' ? 2 : 3;
  const sorted = list.slice().sort((a, b) => rank(a.id) - rank(b.id) || (b.update || 0) - (a.update || 0));
  for (const c of sorted) {
    const o = document.createElement('option'); o.value = c.id;
    o.textContent = c.id + (c.update != null ? ' · upd ' + c.update : '') + (c.id === 'latest' ? ' (follow)' : '');
    sel.appendChild(o);
  }
  if (!sorted.some(c => c.id === S.ck)) S.ck = sorted[0].id;
  sel.value = S.ck;
}

/* ---------- episodes ---------- */
async function fetchEpisode(immediate) {
  const run = S.run; if (!run || !S.checkpoints.length) return;
  const tok = ++S.epToken, ck = S.ck, seed = S.seed;
  S.epWaiting = true;
  try {
    const ep = await api.episode(run, ck, seed);
    if (tok !== S.epToken) return;
    S.epError = null;
    if (!ep || !Array.isArray(ep.frames) || !ep.frames.length) throw new HttpError(500, 'empty episode');
    hideOverlay();
    const cur = S.ep;
    const same = cur && cur.meta.update === ep.meta.update && cur.meta.seed === ep.meta.seed && cur.meta.checkpoint === ep.meta.checkpoint;
    if (immediate || !cur) setEpisode(ep);
    else if (!same) { S.pending = ep; if (!S.playing || S.f >= cur.frames.length - 1) swapPending(); }
    if (S.compare) fetchBaseline();
  } catch (e) {
    if (tok !== S.epToken) return;
    if (e.status === 404 && STATIC) { S.epError = e.message; if (!S.ep) showOverlay('Not in the recorded demo', e.message, true); }
    else if (e.status === 404) { S.epError = 'waiting'; if (!S.ep) showOverlay('Waiting for first checkpoint', 'No policy is available for this selection yet.', false, true); }
    else if (e.status === 0) { /* conn handled by runs poll */ if (!S.ep) showOverlay('Backend unreachable', 'Waiting for the API...', true); }
    else { S.epError = e.message; if (!S.ep) showOverlay('Could not load episode', e.message, true); }
  } finally { if (tok === S.epToken) S.epWaiting = false; }
}
async function fetchBaseline() {
  const seed = S.seed;
  try {
    const b = await api.baseline(seed);
    if (seed !== S.seed) return;
    S.base = b && b.frames && b.frames.length ? b : null;
  } catch (e) { S.base = null; }
  renderChips();
}
function setEpisode(ep) {
  S.ep = ep; S.pending = null; S.f = 0; S.endHold = 0; S.trail = []; S.particles = []; S.cam = null; S.lastTeleIdx = -1;
  if (isNum(ep.meta.seed) && ep.meta.seed !== S.seed) { S.seed = ep.meta.seed; $('seedInput').value = S.seed; $('seedSelect').value = S.seed; if (S.compare) fetchBaseline(); }
  const sc = $('scrub'); sc.max = ep.frames.length - 1; sc.value = 0;
  $('outcome').hidden = true;
  renderChips(); renderTelemetry(true); renderTraining();
}
function swapPending() { if (S.pending) setEpisode(S.pending); }

function renderChips() {
  const ep = S.ep;
  $('chipCkpt').innerHTML = ep ? 'from <b>' + ep.meta.checkpoint + '</b>' + (ep.meta.update != null ? ' · update <b>' + ep.meta.update + '</b>' : '') : 'checkpoint —';
  $('chipSeed').innerHTML = ep ? 'seed <b>' + ep.meta.seed + '</b> · ' + ep.frames.length + ' frames' : 'seed —';
  const cb = $('chipBase');
  cb.hidden = !(S.compare && S.base);
  if (!cb.hidden) cb.textContent = 'baseline · ' + S.base.outcome.replace(/_/g, ' ') + ' · ' + fmt(S.base.total_reward, 1);
  $('legendHud').hidden = !S.compare;
}

/* ---------- overlay ---------- */
function showOverlay(title, text, err, waiting) {
  if (S.ep) return;
  const o = $('overlay'); o.hidden = false; o.className = 'overlay' + (err ? ' err' : '');
  $('overlayTitle').textContent = title; $('overlayText').textContent = text || '';
}
function hideOverlay() { $('overlay').hidden = true; }

/* ---------- telemetry ---------- */
const TELE = [
  { k: 'x', lbl: 'X', u: 'm', f: (f) => sgn(f.x, 2), lim: (m) => '|x| ≤ ' + fmt(m.world.pad_half_width, 0) + ' pad', ok: (f, m) => Math.abs(f.x) <= m.world.pad_half_width, frac: (f, m) => Math.abs(f.x) / (m.world.half_width) },
  { k: 'y', lbl: 'Y (alt)', u: 'm', f: (f) => fmt(f.y, 2) },
  { k: 'ang', lbl: 'Angle', u: '°', f: (f) => sgn(f.angle * 180 / Math.PI, 1), lim: (m) => '|θ| ≤ ' + fmt(m.limits.safe_angle, 2) + ' rad', ok: (f, m) => Math.abs(f.angle) <= m.limits.safe_angle, frac: (f, m) => Math.abs(f.angle) / (m.limits.safe_angle * 3) },
  { k: 'vx', lbl: 'Vx', u: 'm/s', f: (f) => sgn(f.vx, 2), lim: (m) => '|vx| ≤ ' + fmt(m.limits.safe_vx, 1), ok: (f, m) => Math.abs(f.vx) <= m.limits.safe_vx, frac: (f, m) => Math.abs(f.vx) / (m.limits.safe_vx * 5) },
  { k: 'vy', lbl: 'Vy', u: 'm/s', f: (f) => sgn(f.vy, 2), lim: (m) => '|vy| ≤ ' + fmt(m.limits.safe_vy, 1), ok: (f, m) => Math.abs(f.vy) <= m.limits.safe_vy, frac: (f, m) => Math.abs(f.vy) / (m.limits.safe_vy * 5) },
  { k: 'av', lbl: 'Ang. vel', u: 'rad/s', f: (f) => sgn(f.av, 3), lim: (m) => '|ω| ≤ ' + fmt(m.limits.safe_angular_velocity, 2), ok: (f, m) => Math.abs(f.av) <= m.limits.safe_angular_velocity, frac: (f, m) => Math.abs(f.av) / (m.limits.safe_angular_velocity * 4) },
];
function buildTele() {
  const g = $('teleGrid'); g.innerHTML = '';
  for (const t of TELE) {
    const c = document.createElement('div'); c.className = 'cell'; c.id = 'c_' + t.k;
    c.innerHTML = '<span class="lbl">' + t.lbl + '</span><div><span class="v" id="v_' + t.k + '">—</span><span class="u">' + t.u + '</span></div>' +
      (t.lim ? '<div class="lim" id="l_' + t.k + '">&nbsp;</div><div class="meter"><i id="m_' + t.k + '"></i></div>' : '<div class="lim">&nbsp;</div><div class="meter"><i style="display:none"></i></div>');
    g.appendChild(c);
  }
}
function renderTelemetry(reset) {
  const ep = S.ep;
  if (!ep) {
    for (const t of TELE) { $('v_' + t.k).textContent = '—'; $('c_' + t.k).className = 'cell'; }
    $('actionChip').textContent = '—'; $('actionChip').className = 'action-chip mono';
    $('fuelFill').style.width = '0'; $('fuelVal').textContent = '—'; $('rewVal').textContent = '—'; $('cumVal').textContent = '—'; $('outVal').textContent = '—'; $('outVal').className = 'mono';
    return;
  }
  const i = clamp(Math.floor(S.f + 1e-6), 0, ep.frames.length - 1);
  if (i === S.lastTeleIdx && !reset) return; S.lastTeleIdx = i;
  const fr = ep.frames[i], m = ep.meta;
  for (const t of TELE) {
    $('v_' + t.k).textContent = t.f(fr);
    const c = $('c_' + t.k);
    if (t.ok) {
      const ok = t.ok(fr, m); c.className = 'cell' + (fr.y > 25 && i < ep.frames.length - 1 ? '' : (ok ? ' ok' : ' bad'));
      $('l_' + t.k).textContent = (ok ? '✓ ' : '✗ ') + t.lim(m);
      $('m_' + t.k).style.width = clamp(t.frac(fr, m) * 100, 2, 100) + '%';
    }
  }
  const a = fr.action;
  const ac = $('actionChip'); ac.textContent = a >= 0 ? ACTIONS[a] : 'START'; ac.className = 'action-chip mono' + (a >= 1 ? ' a' + a : '');
  $('fuelFill').style.width = clamp(fr.fuel, 0, 1) * 100 + '%'; $('fuelFill').className = fr.fuel < 0.15 ? 'low' : '';
  $('fuelVal').textContent = pct(fr.fuel, 1);
  $('rewVal').textContent = sgn(fr.reward, 3); $('cumVal').textContent = sgn(fr.cum_reward, 2);
  const end = i >= ep.frames.length - 1;
  $('outVal').textContent = end ? ep.outcome.replace(/_/g, ' ').toUpperCase() : 'IN FLIGHT';
  $('outVal').className = 'mono' + (end ? ' ' + ep.outcome : '');
  const ts = i * m.dt;
  $('hud').textContent = 'T+ ' + ts.toFixed(2).padStart(5, '0') + 's · frame ' + i + '/' + (ep.frames.length - 1) + ' · ' + m.run + (m.checkpoint ? ' @ ' + m.checkpoint : '');
  $('timeLabel').textContent = ts.toFixed(2) + ' / ' + ((ep.frames.length - 1) * m.dt).toFixed(2) + ' s';
  const sc = $('scrub'); if (+sc.value !== S.f) sc.value = S.f; sc.style.setProperty('--fill', (sc.max > 0 ? (S.f / sc.max) * 100 : 0) + '%');
  if (end && $('outcome').hidden) showOutcome(ep);
  if (!end && !$('outcome').hidden) $('outcome').hidden = true;
}
function showOutcome(ep) {
  const o = $('outcome'); o.hidden = false; o.className = 'outcome ' + ep.outcome;
  const names = { landed: 'LANDED', crashed: 'CRASHED', out_of_bounds: 'OUT OF BOUNDS', timeout: 'TIMEOUT' };
  $('outcomeTitle').textContent = names[ep.outcome] || ep.outcome.toUpperCase();
  const last = ep.frames[ep.frames.length - 1];
  $('outcomeSub').textContent = 'total reward ' + sgn(ep.total_reward, 1) + ' · touchdown vy ' + sgn(last.vy, 2) + ' m/s · tilt ' + sgn(last.angle * 180 / Math.PI, 1) + '°';
}

/* ---------- training stats + charts ---------- */
function stepsPerSec() {
  const s = S.stepSamples; if (s.length < 2) return null;
  const a = s[0], b = s[s.length - 1];
  if (b[1] === a[1] || b[0] - a[0] < 3) return s.length > 3 && b[1] === a[1] ? 0 : null;
  return (b[1] - a[1]) / (b[0] - a[0]);
}
function renderStats() {
  const m = S.metrics, r = curRun();
  const lt = m && m.train.length ? m.train[m.train.length - 1] : null;
  const le = m && m.eval.length ? m.eval[m.eval.length - 1] : (r && r.last_eval) || null;
  const sps = stepsPerSec();
  const items = [
    ['Update', lt ? lt.update + (r && r.updates ? ' / ' + r.updates : '') : (r ? String(r.updates) : '—')],
    ['Steps', kfmt(lt ? lt.global_step : r && r.global_step)],
    ['Steps/s', sps == null ? '—' : sps === 0 ? '0' : Math.round(sps).toLocaleString()],
    ['Episodes', lt && isNum(lt.episodes) ? lt.episodes.toLocaleString() : '—'],
    ['Eval success', le ? pct(le.success_rate) : '—'],
    ['Eval crash', le ? pct(le.crash_rate) : '—'],
    ['Mean reward', le && isNum(le.mean_reward) ? fmt(le.mean_reward, 1) : (lt && isNum(lt.ep_reward_mean) ? fmt(lt.ep_reward_mean, 1) : '—')],
    ['Entropy', lt && isNum(lt.entropy) ? fmt(lt.entropy, 3) : '—'],
  ];
  const el = $('stats'); const html = items.map(([k, v]) => '<div><span class="lbl">' + k + '</span><b class="mono">' + v + '</b></div>').join('');
  if (el.dataset.h !== html) { el.dataset.h = html; el.innerHTML = html; }
  $('chipRun').textContent = S.run || '—';
}

function setupCanvas(cv) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2), w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const c = cv.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0); return { c, w, h };
}
/* generic line chart. o: {series:[{label,color,pts,width,alpha,dots,dash}], yMin,yMax,yFmt,marker} */
function drawChart(cv, o) {
  const { c, w, h } = setupCanvas(cv); if (!w) return;
  cv._o = o;
  c.clearRect(0, 0, w, h);
  const L = 40, R = 8, T = 8, B = 18, pw = w - L - R, ph = h - T - B;
  const all = o.series.flatMap(s => s.pts);
  c.font = '10px ' + getComputedStyle(document.body).getPropertyValue('--mono');
  if (!all.length) { c.fillStyle = '#4a5b70'; c.textAlign = 'center'; c.fillText('no data yet', w / 2, h / 2); return; }
  let xMin = Math.min(...all.map(p => p[0])), xMax = Math.max(...all.map(p => p[0]));
  if (xMax === xMin) xMax = xMin + 1;
  let yMin = o.yMin, yMax = o.yMax;
  if (yMin == null || yMax == null) {
    const ys = all.map(p => p[1]); let lo = Math.min(...ys), hi = Math.max(...ys); if (hi === lo) { hi += 1; lo -= 1; }
    const pad = (hi - lo) * 0.08; yMin = o.yMin != null ? o.yMin : lo - pad; yMax = o.yMax != null ? o.yMax : hi + pad;
  }
  const X = (v) => L + (v - xMin) / (xMax - xMin) * pw, Y = (v) => T + (1 - (v - yMin) / (yMax - yMin)) * ph;
  c.lineWidth = 1; c.textAlign = 'right'; c.textBaseline = 'middle';
  for (let i = 0; i <= 4; i++) {
    const v = lerp(yMin, yMax, i / 4), y = Math.round(Y(v)) + .5;
    c.strokeStyle = i === 0 ? '#26364a' : '#172230'; c.beginPath(); c.moveTo(L, y); c.lineTo(w - R, y); c.stroke();
    c.fillStyle = '#5d7088'; c.fillText(o.yFmt ? o.yFmt(v) : fmt(v, 1), L - 6, y);
  }
  c.textAlign = 'center'; c.textBaseline = 'top'; c.fillStyle = '#5d7088';
  const nt = Math.max(2, Math.min(6, Math.floor(pw / 70)));
  for (let i = 0; i <= nt; i++) { const v = lerp(xMin, xMax, i / nt); c.fillText(Math.round(v), clamp(X(v), L + 8, w - R - 8), h - B + 5); }
  if (o.marker != null && o.marker >= xMin && o.marker <= xMax) {
    const x = Math.round(X(o.marker)) + .5; c.strokeStyle = '#ffb34780'; c.setLineDash([3, 3]); c.beginPath(); c.moveTo(x, T); c.lineTo(x, T + ph); c.stroke(); c.setLineDash([]);
    c.fillStyle = '#ffb347'; c.beginPath(); c.moveTo(x - 4, T); c.lineTo(x + 4, T); c.lineTo(x, T + 5); c.fill();
  }
  c.save(); c.beginPath(); c.rect(L, T - 2, pw, ph + 4); c.clip();
  for (const s of o.series) {
    if (!s.pts.length) continue;
    c.globalAlpha = s.alpha == null ? 1 : s.alpha; c.strokeStyle = s.color; c.lineWidth = s.width || 1.75; c.lineJoin = 'round'; c.setLineDash(s.dash || []);
    c.beginPath(); s.pts.forEach((p, i) => i ? c.lineTo(X(p[0]), Y(p[1])) : c.moveTo(X(p[0]), Y(p[1]))); c.stroke(); c.setLineDash([]);
    if (s.fill) { c.globalAlpha = .08; c.fillStyle = s.color; c.lineTo(X(s.pts[s.pts.length - 1][0]), Y(yMin)); c.lineTo(X(s.pts[0][0]), Y(yMin)); c.fill(); c.globalAlpha = s.alpha == null ? 1 : s.alpha; }
    if (s.dots) { c.fillStyle = s.color; for (const p of s.pts) { c.beginPath(); c.arc(X(p[0]), Y(p[1]), 2.2, 0, 7); c.fill(); } }
    c.globalAlpha = 1;
  }
  c.restore();
  // hover crosshair + tooltip
  const hx = cv._hover;
  if (hx != null && hx >= L && hx <= w - R) {
    const xv = xMin + (hx - L) / pw * (xMax - xMin);
    const rows = [];
    for (const s of o.series) {
      if (s.noTip || !s.pts.length) continue;
      let best = s.pts[0]; for (const p of s.pts) if (Math.abs(p[0] - xv) < Math.abs(best[0] - xv)) best = p;
      rows.push({ s, p: best });
    }
    if (rows.length) {
      const px = X(rows[0].p[0]);
      c.strokeStyle = '#7a8ca3'; c.lineWidth = 1; c.beginPath(); c.moveTo(Math.round(px) + .5, T); c.lineTo(Math.round(px) + .5, T + ph); c.stroke();
      for (const r of rows) { c.fillStyle = r.s.color; c.beginPath(); c.arc(X(r.p[0]), Y(r.p[1]), 3.5, 0, 7); c.fill(); c.strokeStyle = '#080b10'; c.lineWidth = 1.5; c.stroke(); }
      const lines = ['update ' + Math.round(rows[0].p[0])].concat(rows.map(r => r.s.label + '  ' + (o.yFmt ? o.yFmt(r.p[1], 1) : fmt(r.p[1], 2))));
      c.font = '10.5px ' + getComputedStyle(document.body).getPropertyValue('--mono');
      const bw = Math.max(...lines.map(l => c.measureText(l).width)) + 16, bh = lines.length * 14 + 8;
      let bx = px + 10; if (bx + bw > w - 4) bx = px - 10 - bw; const by = T + 4;
      c.fillStyle = '#0b1118ee'; c.strokeStyle = '#2f4561'; c.beginPath(); c.roundRect(bx, by, bw, bh, 5); c.fill(); c.stroke();
      c.textAlign = 'left'; c.textBaseline = 'middle';
      lines.forEach((l, i) => { c.fillStyle = i ? rows[i - 1].s.color : '#7a8ca3'; c.fillText(l, bx + 8, by + 11 + i * 14); });
    }
  }
}
function wireHover(cv) {
  cv.addEventListener('mousemove', (e) => { cv._hover = e.clientX - cv.getBoundingClientRect().left; if (cv._o) drawChart(cv, cv._o); });
  cv.addEventListener('mouseleave', () => { cv._hover = null; if (cv._o) drawChart(cv, cv._o); });
}
const col = (r, k) => r.filter(x => isNum(x[k])).map(x => [x.update, x[k]]);
function ema(pts, a) { let v = null; return pts.map(p => [p[0], v = v == null ? p[1] : v + a * (p[1] - v)]); }
function renderTraining() {
  renderStats();
  const m = S.metrics || { train: [], eval: [] };
  const mk = S.ep && isNum(S.ep.meta.update) ? S.ep.meta.update : null;
  const ev = m.eval, tr = m.train;
  const rates = [['success_rate', 'success', '#45e0a0'], ['crash_rate', 'crash', '#ff5d6c'], ['out_of_bounds_rate', 'out of bounds', '#ffb347'], ['timeout_rate', 'timeout', '#8a97a8']];
  const useEv = ev.length >= 2 ? ev : tr;
  drawChart($('chRates'), { yMin: 0, yMax: 1, yFmt: (v, d) => (v * 100).toFixed(0) + '%', marker: mk,
    series: rates.map(([k, l, c]) => ({ label: l, color: c, pts: col(useEv, k), width: k === 'success_rate' ? 2.25 : 1.5, dots: useEv === ev && ev.length < 25, fill: k === 'success_rate' })) });
  $('legendRates').innerHTML = rates.map(([k, l, c]) => '<span><i style="background:' + c + '"></i>' + l + '</span>').join('') + (useEv === ev ? '' : '<span>(train)</span>');
  const rw = col(tr, 'ep_reward_mean');
  drawChart($('chReward'), { yFmt: (v) => fmt(v, 0), marker: mk, series: [
    { label: 'train raw', color: '#58d0ff', pts: rw, alpha: .3, width: 1, noTip: true },
    { label: 'train (smoothed)', color: '#58d0ff', pts: ema(rw, .25), width: 2 },
    { label: 'eval mean', color: '#e6edf5', pts: col(ev, 'mean_reward'), width: 1.5, dots: true, dash: [] }] });
  $('legendReward').innerHTML = '<span><i style="background:#58d0ff"></i>train</span><span><i style="background:#e6edf5"></i>eval</span>';
  drawChart($('chEnt'), { yFmt: (v) => fmt(v, 2), marker: mk, series: [{ label: 'entropy', color: '#a58bff', pts: col(tr, 'entropy'), width: 1.75 }] });
  drawChart($('chVal'), { yFmt: (v) => v >= 100 ? v.toFixed(0) : fmt(v, 1), marker: mk, series: [{ label: 'value loss', color: '#ffb347', pts: col(tr, 'value_loss'), width: 1.75 }] });
}

/* ---------- replay renderer ---------- */
const view = $('view');
const STARS = (() => { let s = 12345; const r = () => (s = (s * 16807) % 2147483647) / 2147483647; return Array.from({ length: 90 }, () => [r(), r(), r() * 1.1 + .3, r()]); })();
const ROCKET_H = 9;
const RSCALE = 1.5; // visual exaggeration so the rocket stays legible at full-world zoom

function frameAt(ep, f) { // interpolated state
  const n = ep.frames.length, i = clamp(Math.floor(f), 0, n - 1), j = Math.min(i + 1, n - 1), t = clamp(f - i, 0, 1);
  const a = ep.frames[i], b = ep.frames[j];
  return { x: lerp(a.x, b.x, t), y: lerp(a.y, b.y, t), angle: a.angle + angDiff(a.angle, b.angle) * t, fuel: lerp(a.fuel, b.fuel, t), action: i === j ? a.action : b.action, i, j, t };
}
function updateCamera(w, h, targets, dt, snap) {
  // needs: half-width and height of the window that must stay visible
  let hw = 30, hh = 34;
  for (const p of targets) { hw = Math.max(hw, Math.abs(p.x) + 22); hh = Math.max(hh, p.y + ROCKET_H + 20); }
  const margin = 10; // metres of ground shown below y=0
  const worldW = (S.ep ? S.ep.meta.world.half_width : 100) * 2 * 1.04;
  let sc = Math.min(w / (2 * hw), h / (hh + margin));
  sc = Math.max(sc, Math.min(w / worldW, h / (210)));
  if (!S.cam || snap) S.cam = { s: sc };
  else S.cam.s = Math.exp(lerp(Math.log(S.cam.s), Math.log(sc), REDUCED ? 1 : 1 - Math.exp(-dt * 3.2)));
  S.cam.margin = margin;
}
function drawRocket(c, s, alpha, opts) {
  // origin at base centre, y up = negative. Units: metres * s.
  const u = s; c.save(); c.globalAlpha = alpha;
  const body = opts.ghost ? ['#6e5cc2', '#a58bff', '#3a2f78'] : ['#e9eef5', '#aab6c6', '#6b7a8e'];
  // legs
  c.strokeStyle = body[2]; c.lineWidth = Math.max(1, .35 * u); c.lineCap = 'round';
  c.beginPath(); c.moveTo(-1.0 * u, -1.6 * u); c.lineTo(-2.3 * u, .4 * u); c.moveTo(1.0 * u, -1.6 * u); c.lineTo(2.3 * u, .4 * u); c.stroke();
  // fins
  c.fillStyle = opts.ghost ? '#4a3d99' : '#c2483f';
  c.beginPath(); c.moveTo(-1.1 * u, -1 * u); c.lineTo(-2.2 * u, .2 * u); c.lineTo(-1.1 * u, -3.4 * u); c.fill();
  c.beginPath(); c.moveTo(1.1 * u, -1 * u); c.lineTo(2.2 * u, .2 * u); c.lineTo(1.1 * u, -3.4 * u); c.fill();
  // nozzle
  c.fillStyle = '#39434f'; c.beginPath(); c.moveTo(-.55 * u, -.5 * u); c.lineTo(-.8 * u, .45 * u); c.lineTo(.8 * u, .45 * u); c.lineTo(.55 * u, -.5 * u); c.fill();
  // body
  const g = c.createLinearGradient(-1.2 * u, 0, 1.2 * u, 0); g.addColorStop(0, body[2]); g.addColorStop(.35, body[0]); g.addColorStop(1, body[1]);
  c.fillStyle = g; c.beginPath(); c.moveTo(-1.15 * u, -.4 * u); c.lineTo(-1.15 * u, -6.3 * u);
  c.quadraticCurveTo(-1.1 * u, -8.6 * u, 0, -9.6 * u); c.quadraticCurveTo(1.1 * u, -8.6 * u, 1.15 * u, -6.3 * u); c.lineTo(1.15 * u, -.4 * u); c.closePath(); c.fill();
  // stripe + window
  c.fillStyle = opts.ghost ? '#a58bff' : '#ffb347'; c.fillRect(-1.15 * u, -4.4 * u, 2.3 * u, .5 * u);
  c.fillStyle = '#1b3a52'; c.beginPath(); c.arc(0, -6.6 * u, .5 * u, 0, 7); c.fill();
  c.restore();
}
function drawFlames(c, s, action, time, intensity) {
  const fl = REDUCED ? 1 : .85 + .3 * Math.sin(time * 0.045) * Math.sin(time * 0.071 + 1);
  if (action === 1) {
    const L = 7.5 * s * fl * intensity;
    const g = c.createLinearGradient(0, 0, 0, L); g.addColorStop(0, '#fff6d8'); g.addColorStop(.25, '#ffcf5a'); g.addColorStop(.7, '#ff7a1c99'); g.addColorStop(1, '#ff3b0000');
    c.fillStyle = g; c.beginPath(); c.moveTo(-.75 * s, .35 * s); c.quadraticCurveTo(-.9 * s, L * .55, 0, L); c.quadraticCurveTo(.9 * s, L * .55, .75 * s, .35 * s); c.fill();
    const g2 = c.createRadialGradient(0, .8 * s, 0, 0, .8 * s, 3 * s); g2.addColorStop(0, '#ffd27a66'); g2.addColorStop(1, '#ffd27a00');
    c.fillStyle = g2; c.fillRect(-3 * s, -2 * s, 6 * s, 6 * s);
  } else if (action === 2 || action === 3) {
    // LEFT = CCW torque: puff from top-right nozzle (to the right) + bottom-left; RIGHT mirrored
    const d = action === 2 ? 1 : -1, L = 2.6 * s * fl;
    for (const [py, dir] of [[-7.4, d], [-1.4, -d]]) {
      const x0 = dir * 1.15 * s, y0 = py * s;
      const g = c.createLinearGradient(x0, 0, x0 + dir * L, 0); g.addColorStop(0, '#e8fbff'); g.addColorStop(.4, '#58d0ffcc'); g.addColorStop(1, '#58d0ff00');
      c.fillStyle = g; c.beginPath(); c.moveTo(x0, y0 - .38 * s); c.lineTo(x0 + dir * L, y0); c.lineTo(x0, y0 + .38 * s); c.fill();
    }
  }
}

let lastTs = 0;
function frame(ts) {
  requestAnimationFrame(frame);
  const dt = Math.min(0.1, (ts - (lastTs || ts)) / 1000); lastTs = ts;
  const ep = S.ep;
  if (ep && S.playing) {
    const n = ep.frames.length - 1;
    if (S.f < n) { S.f = Math.min(n, S.f + dt / ep.meta.dt * S.speed); S.endHold = 0; }
    else {
      S.endHold += dt;
      if (S.endHold > 1.6 && S.tp) { advanceTraining(); }
      else if (S.endHold > 1.6) { if (S.pending) swapPending(); else { S.f = 0; S.trail = []; S.particles = []; S.endHold = 0; S.cam = null; $('outcome').hidden = true; } }
    }
    $('scrub').value = S.f;
  }
  renderTelemetry();
  draw(ts, dt);
}

function draw(ts, dt) {
  const { c, w, h } = setupCanvas(view); if (!w || !h) return;
  const ep = S.ep;
  // sky
  const sky = c.createLinearGradient(0, 0, 0, h); sky.addColorStop(0, '#05080d'); sky.addColorStop(.7, '#0a121c'); sky.addColorStop(1, '#101c2a');
  c.fillStyle = sky; c.fillRect(0, 0, w, h);
  for (const st of STARS) { c.globalAlpha = .25 + .5 * st[3]; c.fillStyle = '#b6c8e0'; c.fillRect(Math.round(st[0] * w), Math.round(st[1] * h * .8), st[2], st[2]); }
  c.globalAlpha = 1;
  if (!ep) return;
  const m = ep.meta, hw = m.world.half_width, ph = m.pad_half_width || m.world.pad_half_width;
  const cur = frameAt(ep, S.f);
  const base = S.compare && S.base ? frameAt(S.base, Math.min(S.f, S.base.frames.length - 1)) : null;
  const targets = [cur].concat(base ? [base] : []);
  updateCamera(w, h, targets, dt, !S.cam);
  const s = S.cam.s, cx = w / 2, gy = h - S.cam.margin * s;
  const SX = (x) => cx + x * s, SY = (y) => gy - y * s;

  // grid
  c.lineWidth = 1; c.font = '10px ' + getComputedStyle(document.body).getPropertyValue('--mono'); c.textBaseline = 'middle';
  const step = s > 6 ? 10 : s > 2.5 ? 20 : 50;
  c.strokeStyle = '#ffffff0a'; c.fillStyle = '#4a5b70'; c.textAlign = 'left';
  for (let y = step; y < m.world.height; y += step) { const py = Math.round(SY(y)) + .5; if (py < 0) break; c.beginPath(); c.moveTo(0, py); c.lineTo(w, py); c.stroke(); c.fillText(y + ' m', 6, py - 7); }
  for (let x = -Math.floor(hw / step) * step; x <= hw; x += step) { const px = Math.round(SX(x)) + .5; if (px < 0 || px > w) continue; c.beginPath(); c.moveTo(px, 0); c.lineTo(px, gy); c.stroke(); }
  // bounds
  c.setLineDash([6, 5]); c.strokeStyle = '#ff5d6c55'; c.lineWidth = 1.25;
  for (const x of [-hw, hw]) { const px = Math.round(SX(x)) + .5; if (px > -2 && px < w + 2) { c.beginPath(); c.moveTo(px, 0); c.lineTo(px, gy); c.stroke(); } }
  const topY = SY(m.world.height); if (topY > 0) { c.beginPath(); c.moveTo(0, topY); c.lineTo(w, topY); c.stroke(); }
  c.setLineDash([]);
  // danger tint outside bounds
  c.fillStyle = '#ff5d6c0d'; if (SX(-hw) > 0) c.fillRect(0, 0, SX(-hw), gy); if (SX(hw) < w) c.fillRect(SX(hw), 0, w - SX(hw), gy);
  // ground
  const gg = c.createLinearGradient(0, gy, 0, h); gg.addColorStop(0, '#1d2a3a'); gg.addColorStop(1, '#0c131c');
  c.fillStyle = gg; c.fillRect(0, gy, w, h - gy);
  c.strokeStyle = '#34506f'; c.lineWidth = 1.5; c.beginPath(); c.moveTo(0, gy + .5); c.lineTo(w, gy + .5); c.stroke();
  c.save(); c.beginPath(); c.rect(0, gy, w, h - gy); c.clip(); c.strokeStyle = '#ffffff0b'; c.lineWidth = 1;
  for (let x = -hw - 20; x <= hw + 20; x += 10) { const px = SX(x); c.beginPath(); c.moveTo(px, gy); c.lineTo(px - 8 * s * .6, h); c.stroke(); } c.restore();
  // pad
  const padL = SX(-ph), padR = SX(ph), padH = Math.max(3, .9 * s);
  const pg = c.createLinearGradient(0, gy - padH, 0, gy); pg.addColorStop(0, '#59e8b0'); pg.addColorStop(1, '#1d8f68');
  c.fillStyle = pg; c.fillRect(padL, gy - padH, padR - padL, padH);
  c.fillStyle = '#04140d'; for (let i = 0; i < 6; i++) c.fillRect(padL + (padR - padL) * (i + .5) / 6 - 1, gy - padH + 1, 2, padH - 2);
  const blink = REDUCED ? 1 : .5 + .5 * Math.sin(ts / 400);
  for (const px of [padL, padR]) { c.fillStyle = '#45e0a0'; c.globalAlpha = .4 + .6 * blink; c.beginPath(); c.arc(px, gy - padH - 3, 2.2, 0, 7); c.fill(); c.globalAlpha = 1; }
  c.fillStyle = '#45e0a0'; c.globalAlpha = .75; c.textAlign = 'center'; c.fillText('PAD', SX(0), gy + 14); c.globalAlpha = 1;
  c.fillStyle = '#4a5b70'; c.fillText('0', SX(0), gy + 28);
  for (const x of [-hw, hw]) { if (SX(x) > 30 && SX(x) < w - 30) c.fillText(x > 0 ? '+' + x : x, SX(x), gy + 14); }

  // trails
  const trailOf = (e, f, color, dash, n) => {
    const i = Math.floor(f), from = Math.max(0, i - n);
    c.lineWidth = 2; c.setLineDash(dash);
    for (let k = from; k < i; k++) {
      const a = e.frames[k], b = e.frames[k + 1], al = (1 - (i - k) / n);
      c.strokeStyle = color; c.globalAlpha = Math.max(0, al) * .85; c.beginPath(); c.moveTo(SX(a.x), SY(a.y + ROCKET_H * .3)); c.lineTo(SX(b.x), SY(b.y + ROCKET_H * .3)); c.stroke();
    }
    c.setLineDash([]); c.globalAlpha = 1;
  };
  if (base) trailOf(S.base, Math.min(S.f, S.base.frames.length - 1), '#a58bff', [4, 4], 140);
  trailOf(ep, S.f, '#ffb347', [], 140);

  // exhaust particles
  if (!REDUCED) {
    if (S.playing && cur.action === 1 && dt > 0) {
      const n = Math.max(1, Math.round(dt * 90 * Math.min(S.speed, 2)));
      for (let k = 0; k < n; k++) {
        const a = cur.angle, ex = cur.x + Math.sin(a) * .4, ey = cur.y + Math.cos(a) * .4 * 0;
        const sp = 18 + Math.random() * 22, sx = (Math.random() - .5) * 7;
        S.particles.push({ x: ex, y: ey, vx: sx + Math.sin(a) * sp, vy: -Math.cos(a) * sp, life: 0, max: .5 + Math.random() * .5 });
      }
      if (S.particles.length > 220) S.particles.splice(0, S.particles.length - 220);
    }
    if (S.playing) for (const p of S.particles) { const d = dt * S.speed; p.life += d; p.x += p.vx * d; p.y += p.vy * d; p.vy -= 4 * d; if (p.y < 0) { p.y = 0; p.vy *= -.25; p.vx *= .8; } }
    S.particles = S.particles.filter(p => p.life < p.max);
    for (const p of S.particles) {
      const k = p.life / p.max; c.globalAlpha = (1 - k) * .55; c.fillStyle = k < .35 ? '#ffd27a' : '#ff7a3c';
      c.beginPath(); c.arc(SX(p.x), SY(p.y), Math.max(.8, (1 - k * .6) * 1.5 * Math.max(1, s * .25)), 0, 7); c.fill();
    }
    c.globalAlpha = 1;
  }

  // ghost + rocket
  const drawAt = (st, opts) => {
    c.save(); c.translate(SX(st.x), SY(st.y)); c.rotate(-st.angle);
    const rs = s * RSCALE;
    if (!opts.dead) drawFlames(c, rs, st.fuel > 0 ? st.action : 0, ts, 1);
    drawRocket(c, rs, opts.alpha, opts); c.restore();
  };
  const atEnd = S.f >= ep.frames.length - 1;
  if (base) drawAt(base, { ghost: true, alpha: .55, dead: S.f >= S.base.frames.length - 1 });
  if (!(atEnd && ep.outcome === 'crashed')) drawAt(cur, { alpha: 1, dead: atEnd });
  else drawWreck(c, SX(cur.x), SY(cur.y), s, cur.angle);

  // end markers
  if (atEnd) {
    const t = S.endHold, e = REDUCED ? 1 : clamp(t / 0.9, 0, 1), last = ep.frames[ep.frames.length - 1];
    const mx = SX(last.x), my = SY(last.y);
    if (ep.outcome === 'landed') {
      c.strokeStyle = '#45e0a0'; for (let r = 0; r < 3; r++) { const k = clamp(e * 1.2 - r * .25, 0, 1); c.globalAlpha = (1 - k) * .8; c.lineWidth = 2; c.beginPath(); c.ellipse(mx, gy - 2, 10 + k * 8 * s, 3 + k * 2 * s, 0, 0, 7); c.stroke(); }
      c.globalAlpha = 1;
    } else if (ep.outcome === 'crashed') {
      const grd = c.createRadialGradient(mx, my, 0, mx, my, 9 * s * (.4 + e)); grd.addColorStop(0, '#fff0c0' + (e < 1 ? 'ff' : 'aa')); grd.addColorStop(.35, '#ff8a2ccc'); grd.addColorStop(1, '#ff3b3000');
      c.globalAlpha = 1 - e * .45; c.fillStyle = grd; c.beginPath(); c.arc(mx, my, 9 * s * (.4 + e), 0, 7); c.fill(); c.globalAlpha = 1;
      c.strokeStyle = '#ff5d6c'; c.lineWidth = 2.5; const q = 3 + 1.5 * s * .5; c.beginPath(); c.moveTo(mx - q, my - q - 14); c.lineTo(mx + q, my + q - 14); c.moveTo(mx + q, my - q - 14); c.lineTo(mx - q, my + q - 14); c.stroke();
    } else {
      c.strokeStyle = ep.outcome === 'timeout' ? '#ffb347' : '#ff5d6c'; c.lineWidth = 2; c.beginPath(); c.arc(mx, my, 14 + 4 * Math.sin(t * 4), 0, 7); c.stroke();
    }
    // marker label
    const lbl = { landed: '✓ touchdown', crashed: '✕ impact', out_of_bounds: '! out of bounds', timeout: '⏱ timeout' }[ep.outcome];
    c.fillStyle = ep.outcome === 'landed' ? '#45e0a0' : ep.outcome === 'timeout' ? '#ffb347' : '#ff5d6c'; c.font = '600 11px ' + getComputedStyle(document.body).getPropertyValue('--mono'); c.textAlign = 'center';
    c.fillText(lbl, clamp(mx, 60, w - 60), clamp(my - 34, 20, gy - 40));
  }
}
function drawWreck(c, x, y, s, a) {
  c.save(); c.translate(x, y); c.rotate(-a * 0.5); c.fillStyle = '#2b3138'; c.strokeStyle = '#59616b'; c.lineWidth = 1;
  c.beginPath(); c.moveTo(-1.4 * s, 0); c.lineTo(-.6 * s, -1.4 * s); c.lineTo(.3 * s, -.7 * s); c.lineTo(1.5 * s, -1.9 * s); c.lineTo(2 * s, 0); c.closePath(); c.fill(); c.stroke(); c.restore();
}

/* ---------- controls ---------- */
function setPlaying(p) {
  S.playing = p; const b = $('btnPlay');
  b.innerHTML = p ? '&#10074;&#10074;' : '&#9654;'; b.setAttribute('aria-pressed', String(p));
}
function seek(f, snap) {
  if (!S.ep) return; S.f = clamp(f, 0, S.ep.frames.length - 1); S.particles = []; S.endHold = 0;
  S.lastTeleIdx = -1; $('scrub').value = S.f; if (snap) S.cam = null;
}
function newSeed(val) {
  S.seed = val; $('seedInput').value = val; $('seedSelect').value = val; S.base = null;
  fetchEpisode(true); if (S.compare) fetchBaseline();
}
function wire() {
  $('btnPlay').onclick = () => { if (S.ep && S.f >= S.ep.frames.length - 1 && !S.playing) seek(0, true); setPlaying(!S.playing); };
  $('btnBack').onclick = () => { setPlaying(false); seek(Math.ceil(S.f) - 1); };
  $('btnFwd').onclick = () => { setPlaying(false); seek(Math.floor(S.f) + 1); };
  $('scrub').addEventListener('input', (e) => { seek(+e.target.value); });
  $('scrub').addEventListener('pointerdown', () => { S._wasPlaying = S.playing; setPlaying(false); });
  $('scrub').addEventListener('pointerup', () => { if (S._wasPlaying) setPlaying(true); });
  $('speedSeg').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return; S.speed = +b.dataset.speed;
    for (const x of $('speedSeg').children) x.classList.toggle('on', x === b);
  });
  $('btnSeed').onclick = () => {
    if (STATIC) { const sd = MANIFEST.seeds; stopTrainingReplay(); newSeed(sd[(sd.indexOf(S.seed) + 1) % sd.length]); }
    else newSeed(Math.floor(Math.random() * 100000));
  };
  $('seedSelect').addEventListener('change', (e) => { stopTrainingReplay(); newSeed(+e.target.value); });
  $('btnTrain').onclick = () => startTrainingReplay();
  $('seedInput').addEventListener('change', (e) => { const v = Math.max(0, Math.floor(+e.target.value || 0)); newSeed(v); });
  $('ckptSelect').addEventListener('change', (e) => { stopTrainingReplay(); S.ck = e.target.value; S.lastLatestUpdate = undefined; fetchEpisode(true); });
  $('cmpToggle').addEventListener('change', (e) => { S.compare = e.target.checked; if (S.compare) fetchBaseline(); else { S.base = null; renderChips(); } renderChips(); });
  $('runSelect').addEventListener('change', (e) => selectRun(e.target.value, true));
  document.addEventListener('keydown', (e) => {
    if (/INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName) && document.activeElement.type !== 'range') return;
    if (e.code === 'Space') { e.preventDefault(); $('btnPlay').click(); }
    else if (e.code === 'ArrowRight') { $('btnFwd').click(); }
    else if (e.code === 'ArrowLeft') { $('btnBack').click(); }
    else if (e.key === 'n' || e.key === 'N') $('btnSeed').click();
  });
  for (const id of ['chRates', 'chReward', 'chEnt', 'chVal']) wireHover($(id));
  new ResizeObserver(() => renderTraining()).observe(document.querySelector('.side'));
  window.__viewer = S; // handy for debugging in devtools
}

/* ---------- static (recorded) mode ---------- */
function staticStages() {
  return S.checkpoints.filter(c => c.id !== 'latest' && c.id !== 'best').slice().sort((a, b) => (a.update || 0) - (b.update || 0));
}
function renderTpBar() {
  const bar = $('tpBar');
  if (!S.tp) { bar.hidden = true; return; }
  const st = S.tp.stages, i = S.tp.i, c = st[i], tot = (st[st.length - 1].update || 1);
  bar.hidden = false;
  bar.innerHTML = '<span><b>RECORDED TRAINING RUN - NOT LIVE.</b> Replaying saved checkpoints of ' + S.run + ' on seed ' + S.seed + '.</span>' +
    '<span>stage <b>' + (i + 1) + '/' + st.length + '</b> · ' + c.id + ' · update ' + c.update + ' (' + Math.round(100 * (c.update || 0) / tot) + '% of training)</span>' +
    '<button class="tpstop" id="tpStop">stop</button>';
  $('tpStop').onclick = stopTrainingReplay;
}
function gotoStage(i) {
  const c = S.tp.stages[i]; S.tp.i = i; S.ck = c.id; $('ckptSelect').value = c.id;
  fetchEpisode(true); renderTpBar();
}
function startTrainingReplay() {
  const stages = staticStages(); if (!stages.length) return;
  S.tp = { stages, i: 0 }; setPlaying(true);
  if (S.speed < 4) $('speedSeg').querySelector('[data-speed="4"]').click(); // early checkpoints flail for ~40 s at 1x
  gotoStage(0);
}
function stopTrainingReplay() { if (S.tp) { S.tp = null; renderTpBar(); } }
function advanceTraining() {
  const tp = S.tp; if (!tp) return;
  S.endHold = 0;
  if (tp.i + 1 < tp.stages.length) gotoStage(tp.i + 1);
  else { S.tp = null; renderTpBar(); }  // finished: keep the final policy on screen
}
function setupStatic() {
  const m = MANIFEST, n = m.seeds.length;
  $('seedInput').hidden = true; $('seedSelect').hidden = false;
  $('seedSelect').innerHTML = m.seeds.map(s => '<option value="' + s + '">' + s + '</option>').join('');
  S.seed = m.seeds[0]; $('seedSelect').value = S.seed; $('seedInput').value = S.seed;
  $('btnSeed').textContent = 'Next seed'; $('btnSeed').title = 'Next exported seed (N)';
  $('btnTrain').hidden = false;
  const note = $('staticNote'); note.hidden = false;
  note.textContent = 'Recorded demo: this page plays back saved rollouts of a finished training run (' + m.run + '), not a live model. ' +
    'Only ' + n + ' seeds (' + m.seeds[0] + '-' + m.seeds[n - 1] + ') and ' + m.checkpoints.length + ' checkpoints (' + m.checkpoints.map(c => c.id).join(', ') + ') were pre-computed, so seed and checkpoint choices are limited to those. ' +
    'Training curves are thinned to ' + m.train_rows_exported + ' of ' + m.train_rows_total + ' updates.';
  const b = $('banner'); b.hidden = false; b.className = 'banner info';
  b.textContent = 'This is a recorded training run, not live. Use "Replay training progress" to watch the policy go from crashing to landing across saved checkpoints.';
}

/* ---------- boot ---------- */
buildTele(); wire(); setPlaying(S.playing); renderChips(); renderTelemetry(true); renderTraining();
showOverlay('Connecting', FIXTURE ? 'loading fixtures' : 'contacting /api/runs');
detectStatic().then(() => {
  if (STATIC) { setupStatic(); showOverlay('Loading recorded demo', 'demo/'); }
  pollRuns();
  setInterval(() => { if (!(STATIC && S.lastOK)) pollRuns(); }, 2000);
});
setInterval(() => { if (S.run && S.ck === 'latest' && S.checkpoints.length && !document.hidden) fetchEpisode(false); }, 5000);
requestAnimationFrame(frame);
})();
