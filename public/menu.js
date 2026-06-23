/**
 * Event Horizon — the main menu, recreated in the 2D composite pipeline.
 *
 * The original mockup was a standalone Three.js scene. Here the same look — a
 * pixel black hole with a spinning accretion disk, infalling tetromino debris,
 * a glitchy EVENT HORIZON title and a ring of NES HUD panels — is rebuilt with
 * the very same canvas primitives the game uses (drawText / drawBlock /
 * frameBox), so it composites into #source and goes through the CRT shader like
 * everything else. No extra render loop, no three.js.
 *
 * The scene is pure decoration; the interactive controls (player name, start
 * level, START) are overlaid by app.js on top of it. createMenu() returns a
 * renderer plus the toolkit it was handed, so app.js stays the single owner of
 * the font, palette and hit-testing.
 */

import { NTSC_FPS, framesPerGridcell } from "./nes-engine.js";
import { PuzzleEnv, Agent, GRID, GRID_H } from "./rl.js";
import { argmax } from "./nn.js";

// The real NES frame period — the atomic tick the menu quantises to, so debris
// rotation and consumption happen on authentic Tetris timing, not free-running.
const FRAME_MS = 1000 / NTSC_FPS;
// Level-0 gravity (48 frames/cell) is the slowest drop; treat it as the longest
// rotation interval, scaling shorter for faster orbits but never below 1 frame.
const SLOWEST_DROP_FRAMES = framesPerGridcell(0); // 48

// NES-ish palette, mirroring the mockup's COL table (hex strings for fillStyle).
const COL = {
  black: "#000000",
  white: "#fcfcfc",
  lightGray: "#bcbcbc",
  darkGray: "#7c7c7c",
  deepBlue: "#0020bc",
  blue: "#0058f8",
  lightBlue: "#3cbcfc",
  cyan: "#00a8f8",
  purple: "#6844fc",
  magenta: "#d800cc",
  red: "#f83800",
  darkRed: "#a81000",
  orange: "#fca044",
  yellow: "#f8b800",
  green: "#58d854",
  darkGreen: "#248400",
};

// Identity colour per tetromino, used to tint the debris blocks.
const PIECE_COLORS = {
  I: { main: COL.cyan, light: COL.lightBlue, dark: COL.blue },
  O: { main: COL.yellow, light: COL.white, dark: COL.orange },
  T: { main: COL.magenta, light: COL.purple, dark: COL.deepBlue },
  S: { main: COL.green, light: COL.yellow, dark: COL.darkGreen },
  Z: { main: COL.red, light: COL.orange, dark: COL.darkRed },
  J: { main: COL.blue, light: COL.lightBlue, dark: COL.deepBlue },
  L: { main: COL.orange, light: COL.yellow, dark: COL.darkRed },
};

const TETROMINOES = {
  I: [[1, 1, 1, 1]],
  O: [[1, 1], [1, 1]],
  T: [[0, 1, 0], [1, 1, 1]],
  S: [[0, 1, 1], [1, 1, 0]],
  Z: [[1, 1, 0], [0, 1, 1]],
  J: [[1, 0, 0], [1, 1, 1]],
  L: [[0, 0, 1], [1, 1, 1]],
};
const PIECE_TYPES = Object.keys(TETROMINOES);

const clamp01 = (v) => Math.max(0, Math.min(1, v));
const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3);

/* -------------------------------------------------------------------------- */
/* STATUS panel demo — a live neural agent solving a 4x4 pieces-fill puzzle.    */
/*                                                                            */
/* This is NOT scripted: it runs the policy network trained by                 */
/* `npm run train` (loaded from status-policy.json). Each turn it samples a     */
/* placement from the net (stochastic, so it usually solves but can dead-end),  */
/* highlights the chosen cells with a selection outline for a beat, commits     */
/* the placement, then advances. On a finished episode it flashes SOLVED or     */
/* FAILED and restarts. Until the policy loads it falls back to greedy-random    */
/* legal moves so the panel is never empty.                                    */
/* -------------------------------------------------------------------------- */
function createStatusDemo() {
  const env = new PuzzleEnv();
  let agent = null; // set once the trained policy loads
  let usingRandom = true;

  // Try to load the trained policy; degrade gracefully if it's missing or if
  // fetch isn't available (e.g. running this module under Node for tests).
  if (typeof fetch === "function") {
    fetch("/status-policy.json")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error("no policy"))))
      .then((json) => { agent = Agent.fromJSON(json); usingRandom = false; })
      .catch(() => { /* keep the random fallback */ });
  }

  // phase machine timings (seconds)
  const SELECT_S = 0.55; // outline the chosen placement
  const PLACE_S = 0.2; // brief settle after committing
  const END_S = 1.4; // hold SOLVED / FAILED before restarting
  const EPSILON = 0.18; // chance per move the agent explores (lets it fail)

  let phase = "select"; // select -> place -> (next select | end) -> select
  let timer = 0;
  let chosen = -1; // anchor cell currently outlined
  let chosenCells = null; // [{x,y}] of the outlined placement
  let outcome = null; // "solved" | "failed" while in the end phase
  let lastResult = "—"; // sticky label of the previous episode
  let exploring = false; // true when this move was a random exploration

  env.reset(Math.random);
  pickMove();

  // Choose the next action: sample from the policy (so it can fail), or a
  // random legal move while the policy is still loading.
  function pickMove() {
    const mask = env.actionMask();
    let any = false;
    for (const m of mask) if (m) { any = true; break; }
    if (!any) { chosen = -1; chosenCells = null; return false; }

    if (agent) {
      // ε-exploration: occasionally the agent "explores" a random legal move
      // instead of its best one. The trained policy is near-perfect, so this is
      // what makes the demo genuinely able to FAIL sometimes (dead-end), not
      // just always solve — the panel header flags it as EXPLORING.
      exploring = Math.random() < EPSILON;
      if (exploring) {
        const legal = [];
        for (let i = 0; i < mask.length; i++) if (mask[i]) legal.push(i);
        chosen = legal[(Math.random() * legal.length) | 0];
      } else {
        const { probs } = agent.policy(env.observe(), mask);
        const r = Math.random();
        let acc = 0;
        chosen = probs.length - 1;
        for (let i = 0; i < probs.length; i++) { acc += probs[i]; if (r <= acc) { chosen = i; break; } }
        if (!mask[chosen]) chosen = argmax(probs); // safety: never pick illegal
      }
    } else {
      // random legal fallback
      const legal = [];
      for (let i = 0; i < mask.length; i++) if (mask[i]) legal.push(i);
      chosen = legal[(Math.random() * legal.length) | 0];
    }

    const p = env.currentPiece;
    const ax = chosen % GRID;
    const ay = (chosen / GRID) | 0;
    chosenCells = p.cells.map(([dx, dy]) => ({ x: ax + dx, y: ay + dy }));
    return true;
  }

  function update(dt) {
    timer += dt;
    if (phase === "select") {
      if (chosen < 0) { // no legal move available -> failed episode
        outcome = "failed"; phase = "end"; timer = 0; return;
      }
      if (timer >= SELECT_S) { phase = "place"; timer = 0; }
    } else if (phase === "place") {
      if (timer >= PLACE_S) {
        const res = env.step(chosen);
        if (res.done) {
          outcome = env.success ? "solved" : "failed";
          phase = "end"; timer = 0;
        } else {
          phase = "select"; timer = 0; pickMove();
        }
      }
    } else if (phase === "end") {
      if (timer >= END_S) {
        lastResult = outcome === "solved" ? "SOLVED" : "FAILED";
        env.reset(Math.random);
        outcome = null; phase = "select"; timer = 0; pickMove();
      }
    }
  }

  return { env, update,
    get phase() { return phase; },
    get chosen() { return chosen; },
    get chosenCells() { return chosenCells; },
    get outcome() { return outcome; },
    get lastResult() { return lastResult; },
    get usingRandom() { return usingRandom; },
    get exploring() { return exploring; },
    get selectPulse() { return timer / SELECT_S; },
  };
}

/**
 * Build the menu renderer.
 *
 * @param {object} opts
 * @param {number} opts.w  composite width  (VW)
 * @param {number} opts.h  composite height (VH)
 * @param {object} opts.toolkit  shared 2D primitives from app.js:
 *        { drawText(c,text,x,y,scale,color), textWidth(text,scale),
 *          drawBlock(c,x,y,cell,color,dim), GLYPH_H }
 */
export function createMenu({ w, h, toolkit }) {
  const { drawText, textWidth, drawBlock, GLYPH_H } = toolkit;

  // -- layout, scaled from the 320x240 mockup into the WxH composite ----------
  // The title sits at the TOP centre (over the well's horizontal span); the
  // black-hole well runs beneath it down to the SAME bottom as the side HUD
  // columns, so the centre canvas matches the panels' full height.
  const PAD = Math.round(w * 0.025);
  const SIDE_W = Math.round(w * 0.18); // width of each HUD column
  const MAIN_X = PAD + SIDE_W + Math.round(w * 0.02);
  const MAIN_W = w - 2 * (MAIN_X - 0); // symmetric well between the columns
  const wellW = w - 2 * MAIN_X;
  const RIGHT_X = w - PAD - SIDE_W;

  // Vertical extent shared by the side columns: their top panel starts at
  // ph(0.035) and the bottom leaderboard ends at ph(0.925). The well matches
  // this bottom; its top is pushed down to leave a title band above it.
  const COL_TOP = Math.round(h * 0.035);
  const COL_BOTTOM = Math.round(h * 0.925);
  const TITLE_BAND_H = Math.round(h * 0.085); // top band reserved for the title
  const MAIN_Y = COL_TOP + TITLE_BAND_H;
  const MAIN_H = COL_BOTTOM - MAIN_Y;

  // Black-hole centre. Anchored in the UPPER part of the well so the controls
  // console can sit beneath it without ever covering the anomaly — the hole is
  // the showpiece and must read clearly above the console.
  const BH = { x: w / 2, y: MAIN_Y + MAIN_H * 0.27 };

  // Scene clip region (inside the main well), so debris/stars never spill into
  // the HUD columns.
  const clip = {
    minX: MAIN_X + 4,
    maxX: MAIN_X + wellW - 4,
    minY: MAIN_Y + 4,
    maxY: MAIN_Y + MAIN_H - 4,
  };

  // -- particle systems, seeded once -----------------------------------------
  const rand = (a, b) => a + Math.random() * (b - a);

  const stars = [];
  for (let i = 0; i < 110; i++) {
    const bx = rand(clip.minX, clip.maxX);
    const by = rand(clip.minY, clip.maxY);
    stars.push({
      baseX: bx,
      baseY: by,
      size: Math.random() > 0.8 ? 2 : 1,
      color: [COL.white, COL.cyan, COL.purple, COL.orange][(Math.random() * 4) | 0],
      tw: rand(2, 9),
      phase: rand(0, 100),
      drift: rand(1, 4),
    });
  }

  // Scale radii up from the mockup (which used ~r 24..58 around an 18px core in
  // a 152px-wide well) by the ratio of our well to the mockup well.
  const RS = wellW / 152; // radius scale

  const ringPixels = [];
  for (let i = 0; i < 520; i++) {
    const roll = Math.random();
    const color =
      roll > 0.93 ? COL.white :
      roll > 0.80 ? COL.yellow :
      roll > 0.58 ? COL.orange :
      roll > 0.35 ? COL.red :
      roll > 0.18 ? COL.magenta :
                    COL.deepBlue;
    ringPixels.push({
      color,
      size: roll > 0.75 ? 2 : 1,
      a: rand(0, Math.PI * 2),
      r: (24 + Math.random() * 34) * RS,
      speed: rand(0.6, 2.4),
      squash: rand(0.38, 0.46),
      wobble: rand(0, 100),
    });
  }

  const rimPixels = [];
  for (let i = 0; i < 72; i++) {
    rimPixels.push({
      color: i % 7 === 0 ? COL.white : COL.orange,
      a: (i / 72) * Math.PI * 2,
      phase: rand(0, 100),
    });
  }

  // Cap orbital radii so debris/streaks stay inside the well rather than flying
  // far off the clip rect. Half the well width is the natural ceiling; leave a
  // little margin so the squashed (vertical) ellipse fits too.
  const maxOrbit = (wellW / 2) * 0.9;

  const streaks = [];
  for (let i = 0; i < 30; i++) {
    streaks.push({
      color: Math.random() > 0.5 ? COL.purple : COL.orange,
      a: rand(0, Math.PI * 2),
      r: Math.min(maxOrbit, (50 + Math.random() * 58) * RS),
      speed: rand(0.2, 0.9),
      len: (4 + ((Math.random() * 8) | 0)) * RS,
      phase: rand(0, 100),
    });
  }

  // (Re)seed one debris piece at the outer edge of the disk. Pieces that fall
  // into the hole are recycled through here so the disk never empties.
  function spawnDebris(d = {}) {
    const speed = rand(0.18, 0.83);
    d.type = PIECE_TYPES[(Math.random() * PIECE_TYPES.length) | 0];
    d.a = rand(0, Math.PI * 2);
    // start near the outer edge; consumption pulls r inward over its lifetime
    d.r = Math.min(maxOrbit, (60 + Math.random() * 50) * RS);
    d.speed = speed;
    d.squash = rand(0.5, 0.6);
    d.cell = Math.random() > 0.75 ? 7 : 5;
    d.pulse = rand(0, 100);
    // Rotation, quantised to the NES tick. Faster orbit = fewer ticks between
    // 90° steps, but never below 1 tick (the real Tetris rotation minimum).
    d.rot = (Math.random() * 4) | 0; // 0..3 quarter-turns
    d.rotTickAccum = 0;
    d.rotIntervalTicks = Math.max(1, Math.round(SLOWEST_DROP_FRAMES * (0.35 / speed)));
    // Consumption: each piece drifts inward at its own slow rate; when it
    // reaches the horizon it is "consumed" (counted) and respawned.
    d.consumeRate = rand(0.6, 1.8) * RS; // px of radius lost per second
    return d;
  }

  const debris = [];
  for (let i = 0; i < 36; i++) {
    const d = spawnDebris();
    // spread initial radii across the whole disk so it doesn't all fall at once
    d.r = Math.min(maxOrbit, (38 + Math.random() * 72) * RS);
    debris.push(d);
  }

  // Running tally of pieces the hole has swallowed — surfaced in the HUD as the
  // VOID "consumed" count and used to occasionally clear a "line".
  let consumed = 0;
  let linesCleared = 0;

  // Latest combined leaderboard, fed in via render(...data). Persisted records
  // have live:false; in-progress players have live:true.
  let entries = [];

  // The STATUS panel's live neural puzzle-solver (loads the trained policy).
  const statusDemo = createStatusDemo();

  // ---- attract mode -------------------------------------------------------
  // The menu runs as a self-cycling attract loop: a boot sequence types out
  // diagnostics, flashes, then hands over to the live scene; from then on a
  // "camera flash" (a brief white bloom + glitch jolt) periodically punctuates
  // the scene, as if a probe were photographing the anomaly. The interactive
  // controls remain live throughout — none of this blocks input.
  const BOOT_LINES = [
    "EVENT HORIZON OS  V1.13",
    "WAKE FROM CRYO ......... OK",
    "MEMORY CHECK 65536K .... OK",
    "GRAVIMETRIC ARRAY ...... OK",
    "TETRO-DRIVE SPIN-UP .... OK",
    "ANOMALY LOCK .......... HOT",
    "ENGAGING ATTRACT MODE",
  ];
  const BOOT_LINE_S = 0.32; // seconds between revealed boot lines
  const BOOT_HOLD_S = 0.6; // pause after the last line before the entry flash
  const bootDuration = BOOT_LINES.length * BOOT_LINE_S + BOOT_HOLD_S;

  // Camera-flash cadence in the attract loop (seconds), and the flash envelope.
  const FLASH_PERIOD_S = 8.5;
  const FLASH_DECAY_S = 0.55; // how long one flash takes to fade
  let flashClock = 0; // seconds accumulated toward the next flash
  let flashLevel = 0; // 0..1 current flash brightness
  let booted = false; // latched true once the current boot sequence finishes

  // Boot/scene time is measured from bootEpoch, not page load, so the boot can
  // be replayed: reboot() just moves the epoch to "now" and clears booted. The
  // idle watchdog reboots after IDLE_TIMEOUT_S without user activity, arcade-
  // style; any interaction (handled in app.js) calls pokeActivity().
  let bootEpoch = 0; // page-seconds at which the current boot started
  let lastActivity = 0; // page-seconds of the last user interaction
  const IDLE_TIMEOUT_S = 30;

  // HUD column panel rects, laid out down each side as fractions of the well.
  const ph = (frac) => Math.round(h * frac);
  // Left column has one fewer panel than the right, so its panels are spaced a
  // little taller to end at the SAME bottom as the right's PHASE (ph 0.70) —
  // this removes the dead gap that used to sit above RANKING.
  const leftPanels = [
    { x: PAD, y: ph(0.035), w: SIDE_W, h: ph(0.13), label: "VOID" },
    { x: PAD, y: ph(0.18), w: SIDE_W, h: ph(0.135), label: "STATUS" },
    { x: PAD, y: ph(0.33), w: SIDE_W, h: ph(0.16), label: "SIGNAL" },
    { x: PAD, y: ph(0.505), w: SIDE_W, h: ph(0.195), label: "SECTOR" },
  ];
  const rightPanels = [
    { x: RIGHT_X, y: ph(0.035), w: SIDE_W, h: ph(0.13), label: "NEXT" },
    { x: RIGHT_X, y: ph(0.18), w: SIDE_W, h: ph(0.055), label: "" },
    { x: RIGHT_X, y: ph(0.25), w: SIDE_W, h: ph(0.10), label: "GRAVITY" },
    { x: RIGHT_X, y: ph(0.365), w: SIDE_W, h: ph(0.16), label: "THREAT" },
    { x: RIGHT_X, y: ph(0.54), w: SIDE_W, h: ph(0.16), label: "PHASE" },
  ];

  // Leaderboard panels fill the empty lower corners under each HUD column:
  // RANKING (all persisted records) on the left, LIVE (in-progress players
  // only) on the right. They share the column x/width and run to near the
  // bottom safe margin, giving room for a header + several rows.
  const LB_Y = ph(0.715);
  const LB_H = ph(0.21); // ends ~ph(0.925), inside the bottom safe margin
  const rankPanel = { x: PAD, y: LB_Y, w: SIDE_W, h: LB_H, label: "RANKING" };
  const livePanel = { x: RIGHT_X, y: LB_Y, w: SIDE_W, h: LB_H, label: "LIVE" };
  const LB_ROWS = 6; // single-line rows per board

  // NEXT preview: a short queue of upcoming pieces that advances on a steady
  // tick, like the real game feeding pieces. Seeded random, cycled over time.
  const nextQueue = [];
  for (let i = 0; i < 16; i++) nextQueue.push(PIECE_TYPES[(Math.random() * PIECE_TYPES.length) | 0]);
  let nextHead = 0;
  let nextTimer = 0;
  const NEXT_PERIOD_S = 1.4; // how often the preview advances

  // Title block sits in the top band, centred over the well. The subtitle is
  // the upper line; the wordmark is drawn `+12` below it (see drawTitle).
  const titleY = COL_TOP + 2;

  // signal jitter, refreshed on its own slow timer
  let signalValue = 88;
  let signalTimer = 0;

  // -- small drawing helpers --------------------------------------------------
  // A panel's label sits at y+4 (7px tall); content lives in the "body" below
  // it. These give every panel a single, consistent reference frame so content
  // can be centred instead of hand-offset, fixing per-panel misalignment.
  const LABEL_BTM = 13; // first usable y below the label
  const PAD_IN = 6; // inner horizontal padding
  const body = (p) => ({
    x: p.x + PAD_IN,
    top: p.y + LABEL_BTM,
    bottom: p.y + p.h - 4,
    w: p.w - 2 * PAD_IN,
    get cy() { return (this.top + this.bottom) / 2; },
    get h() { return this.bottom - this.top; },
  });
  // Baseline-Y so a `scale`-sized line of text is vertically centred at `cy`.
  const textTop = (cy, scale = 1) => Math.round(cy - (GLYPH_H * scale) / 2);

  function panelFrame(c, x, y, pw, phh, label) {
    c.fillStyle = COL.black;
    c.fillRect(x, y, pw, phh);
    c.strokeStyle = COL.white;
    c.lineWidth = 2;
    c.strokeRect(x + 1, y + 1, pw - 2, phh - 2);
    if (label) {
      drawText(c, label, x + Math.round((pw - textWidth(label, 1)) / 2), y + 4, 1, COL.lightBlue);
    }
  }


  // A flat 1px square at (x,y) — the scene's pixel primitive.
  function px(c, x, y, size, color, alpha) {
    c.globalAlpha = alpha;
    c.fillStyle = color;
    c.fillRect(Math.round(x), Math.round(y), size, size);
  }

  // A small tetromino glyph, used in HUD panels (STATUS / PHASE).
  function miniTet(c, type, x, y, bs) {
    const shape = TETROMINOES[type];
    const colors = PIECE_COLORS[type];
    c.fillStyle = colors.main;
    for (let gy = 0; gy < shape.length; gy++) {
      for (let gx = 0; gx < shape[gy].length; gx++) {
        if (shape[gy][gx]) c.fillRect(x + gx * bs, y + gy * bs, bs - 1, bs - 1);
      }
    }
  }

  // A bevelled debris tetromino: main fill, light top edge, dark right edge.
  // `rot` is a quarter-turn count (0..3); the glyph is rotated about its centre.
  function debrisTet(c, type, cx, cy, cell, scale, alpha, rot = 0) {
    const shape = TETROMINOES[type];
    const colors = PIECE_COLORS[type];
    const cw = shape[0].length;
    const chh = shape.length;
    const bs = Math.max(2, Math.round(cell * scale));
    const edge = Math.max(1, Math.floor(bs / 3));
    c.save();
    c.globalAlpha = alpha;
    c.translate(Math.round(cx), Math.round(cy));
    if (rot) c.rotate((rot & 3) * (Math.PI / 2));
    const ox = -(cw * bs) / 2;
    const oy = -(chh * bs) / 2;
    for (let gy = 0; gy < chh; gy++) {
      for (let gx = 0; gx < cw; gx++) {
        if (!shape[gy][gx]) continue;
        const bx = Math.round(ox + gx * bs);
        const by = Math.round(oy + gy * bs);
        c.fillStyle = colors.main;
        c.fillRect(bx, by, bs, bs);
        c.fillStyle = colors.light;
        c.fillRect(bx, by, bs, edge);
        c.fillStyle = colors.dark;
        c.fillRect(bx + bs - edge, by, edge, bs);
      }
    }
    c.restore();
  }

  // ---------------------------------------------------------------------------
  // The whole menu for one frame. `t` is seconds; `dt` is delta seconds.
  // `data.entries` is the combined top-N leaderboard (persisted + live), each
  // { name, score, lines, level, live }, used by the two leaderboard panels.
  // ---------------------------------------------------------------------------
  function render(c, t, dt, data = {}) {
    entries = data.entries || entries;
    // Idle watchdog: once the scene is live, a long stretch with no activity
    // cuts back to a fresh boot + attract cycle (arcade attract-loop feel).
    if (booted && t - lastActivity > IDLE_TIMEOUT_S) reboot(t);

    // Local clock measured from the current boot, so a replay restarts timing.
    const lt = t - bootEpoch;

    // backdrop
    c.fillStyle = COL.black;
    c.fillRect(0, 0, w, h);

    // --- boot sequence: diagnostics type out, then an entry flash hands over
    // to the live scene. Replays on idle via reboot().
    if (!booted && lt < bootDuration) {
      drawBoot(c, lt);
      // a rising flash in the final fraction of boot, peaking at hand-over
      const intoScene = clamp01((lt - (bootDuration - 0.4)) / 0.4);
      if (intoScene > 0) drawFlashOverlay(c, intoScene);
      c.globalAlpha = 1;
      return;
    }
    booted = true;

    // The scene's own time starts at hand-over so the intro ease and orbital
    // phases begin from 0 when the boot finishes, not at the boot epoch.
    const st = lt - bootDuration;
    const intro = easeOutCubic(clamp01(st / 2.2));

    // schedule + decay camera flashes
    flashClock += dt;
    if (flashClock >= FLASH_PERIOD_S) {
      flashClock -= FLASH_PERIOD_S;
      flashLevel = 1;
    }
    if (flashLevel > 0) flashLevel = Math.max(0, flashLevel - dt / FLASH_DECAY_S);

    // a flash jolts the camera: a tiny shake offset applied to the whole scene
    const shake = flashLevel * 2;
    c.save();
    if (shake > 0.1) {
      c.translate(
        Math.round(Math.sin(st * 80) * shake),
        Math.round(Math.cos(st * 71) * shake),
      );
    }

    drawScene(c, st, dt, intro);
    drawMainBorder(c);
    drawHud(c, st, dt);
    drawLeaderboards(c);
    drawTitle(c, st, intro);

    c.restore();

    // the flash bloom sits over everything and fades out
    if (flashLevel > 0.01) drawFlashOverlay(c, flashLevel);

    c.globalAlpha = 1;
  }

  // Diagnostics screen: reveal one boot line at a time with a blinking cursor.
  function drawBoot(c, t) {
    const shown = Math.min(BOOT_LINES.length, Math.floor(t / BOOT_LINE_S) + 1);
    const x = Math.round(w * 0.16);
    let y = Math.round(h * 0.22);
    for (let i = 0; i < shown; i++) {
      const ok = BOOT_LINES[i].endsWith("OK") || BOOT_LINES[i].endsWith("HOT");
      drawText(c, BOOT_LINES[i], x, y, 1, i === BOOT_LINES.length - 1 ? COL.cyan
        : ok ? COL.green : COL.white);
      y += 16;
    }
    // blinking cursor block on the line currently being "typed"
    if (shown < BOOT_LINES.length && Math.floor(t * 3) % 2 === 0) {
      c.fillStyle = COL.green;
      c.fillRect(x + textWidth(BOOT_LINES[shown - 1], 1) + 2, y - 16, 5, GLYPH_H);
    }
  }

  // A white screen-bloom used for the boot hand-over and the attract flashes.
  function drawFlashOverlay(c, level) {
    c.globalAlpha = clamp01(level) * 0.85;
    c.fillStyle = COL.white;
    c.fillRect(0, 0, w, h);
    c.globalAlpha = 1;
  }

  // Restart the boot sequence from `t`. Resets the per-cycle attract state but
  // keeps the consume/lines tally — the anomaly has been running the whole time.
  function reboot(t) {
    bootEpoch = t;
    lastActivity = t;
    booted = false;
    flashClock = 0;
    flashLevel = 0;
  }

  // Called by app.js on any user interaction to defer the idle reboot.
  function pokeActivity(t) {
    lastActivity = t;
  }

  function drawMainBorder(c) {
    c.strokeStyle = COL.white;
    c.lineWidth = 2;
    c.strokeRect(MAIN_X + 1, MAIN_Y + 1, wellW - 2, MAIN_H - 2);
  }

  function drawScene(c, t, dt, intro) {
    // clip the anomaly to the main well
    c.save();
    c.beginPath();
    c.rect(clip.minX, clip.minY, clip.maxX - clip.minX, clip.maxY - clip.minY);
    c.clip();

    // stars, pulled gently toward the hole
    for (const s of stars) {
      const dx = BH.x - s.baseX;
      const dy = BH.y - s.baseY;
      const dist = Math.hypot(dx, dy) || 1;
      const pull = (Math.sin(t * 0.45 + s.phase) * 0.5 + 0.5) * s.drift;
      const x = s.baseX + (dx / dist) * pull;
      const y = s.baseY + (dy / dist) * pull * 0.6;
      const alpha = Math.sin(t * s.tw + s.phase) > -0.1 ? 1 : 0.3;
      px(c, x, y, s.size, s.color, alpha);
    }

    // infall streaks
    for (const s of streaks) {
      const a = s.a + t * s.speed;
      const r = s.r - (Math.sin(t * 0.8 + s.phase) * 0.5 + 0.5) * 10 * RS;
      const x = BH.x + Math.cos(a) * r;
      const y = BH.y + Math.sin(a) * r * 0.62;
      const alpha = 0.35 + 0.65 * Math.sin(t * 6 + s.phase) ** 2;
      c.save();
      c.globalAlpha = alpha;
      c.translate(Math.round(x), Math.round(y));
      c.rotate(a + Math.PI / 2);
      c.fillStyle = s.color;
      c.fillRect(0, 0, Math.max(2, Math.round(s.len)), 1);
      c.restore();
    }

    // accretion disk: pixels behind the hole drawn first (dimmer), in front later
    const core = 18 * RS;
    for (const p of ringPixels) {
      const spin = p.a + t * p.speed;
      const r = p.r + Math.sin(t * 1.8 + p.wobble) * 2 * RS;
      const x = BH.x + Math.cos(spin) * r;
      const y = BH.y + Math.sin(spin) * r * p.squash;
      const front = Math.sin(spin) > 0;
      if (front) continue; // back half now
      px(c, x, y, p.size, p.color, 0.45);
    }

    // event horizon core
    const coreR = core + Math.sin(t * 3) * 0.6 * RS;
    c.globalAlpha = 1;
    c.fillStyle = COL.black;
    c.beginPath();
    c.arc(BH.x, BH.y, coreR, 0, Math.PI * 2);
    c.fill();

    // bright rim around the core
    for (const p of rimPixels) {
      const a = p.a + t * 1.5;
      const r = coreR * 1.07 + Math.sin(t * 10 + p.phase) * 1.1 * RS;
      const alpha = 0.45 + Math.sin(t * 13 + p.phase) * 0.35;
      px(c, BH.x + Math.cos(a) * r, BH.y + Math.sin(a) * r, 2, p.color, alpha);
    }

    // accretion disk front half (brighter, drawn over the core)
    for (const p of ringPixels) {
      const spin = p.a + t * p.speed;
      const r = p.r + Math.sin(t * 1.8 + p.wobble) * 2 * RS;
      const x = BH.x + Math.cos(spin) * r;
      const y = BH.y + Math.sin(spin) * r * p.squash;
      if (Math.sin(spin) <= 0) continue; // front only
      px(c, x, y, p.size, p.color, 0.95);
    }

    // tetromino debris: orbits the hole, rotates on the real Tetris tick, and
    // is slowly consumed (drawn inward) until it crosses the horizon.
    const horizon = coreR + 2 * RS;
    for (const d of debris) {
      // --- quantised rotation: advance whole NES ticks, step 90° when the
      // piece's interval (ticks) elapses. Interval scales with orbit speed but
      // is floored at one tick, the real rotation minimum. ---
      d.rotTickAccum += (dt * 1000) / FRAME_MS; // dt(s) -> ticks
      while (d.rotTickAccum >= d.rotIntervalTicks) {
        d.rotTickAccum -= d.rotIntervalTicks;
        d.rot = (d.rot + 1) & 3;
      }

      // --- consumption: shrink the orbit radius toward the horizon. A gentle
      // breathing wobble rides on top so it still looks alive. ---
      d.r -= d.consumeRate * dt;
      const wobble = (Math.sin(t * 0.5 + d.pulse) * 0.5 + 0.5) * 5 * RS;
      const r = d.r - wobble;

      if (r <= horizon) {
        // swallowed: count it, occasionally clear a "line", respawn at the rim
        consumed++;
        if (consumed % 4 === 0) linesCleared++;
        spawnDebris(d);
        continue;
      }

      const a = d.a + t * d.speed;
      const front = Math.sin(a) > 0;
      const x = BH.x + Math.cos(a) * r;
      const y = BH.y + Math.sin(a) * r * d.squash;
      // fade + shrink as it nears the horizon, so it dissolves into the hole
      const nearness = clamp01((r - horizon) / (40 * RS));
      const sc = (front ? 1.1 : 0.72) * intro * (0.4 + 0.6 * nearness);
      if (sc <= 0.01) continue;
      debrisTet(c, d.type, x, y, d.cell, sc, (front ? 1 : 0.55) * (0.3 + 0.7 * nearness),
        d.rot);
    }

    c.globalAlpha = 1;
    c.restore();
  }

  function drawHud(c, t, dt) {
    // ---- left column --------------------------------------------------------
    for (const p of leftPanels) panelFrame(c, p.x, p.y, p.w, p.h, p.label);

    // VOID: a miniature swirling vortex — several spiral arms of particles
    // falling inward toward a bright pulsing core, with the swallowed-pieces
    // count as the header. Echoes the main black hole, panel-sized.
    const v = leftPanels[0];
    const vb = body(v);
    drawText(c, "VOID", vb.x, vb.top, 1, "#8a8a9a");
    const cnt = String(consumed).padStart(4, "0");
    drawText(c, cnt, vb.x + vb.w - textWidth(cnt, 1), vb.top, 1, COL.cyan);
    const swirlTop = vb.top + GLYPH_H + 2;
    const vcx = v.x + v.w / 2;
    const vcy = (swirlTop + vb.bottom) / 2;
    const vmaxR = Math.min(v.w / 2 - 6, (vb.bottom - swirlTop) / 2);
    const ARMS = 3, PER_ARM = 9;
    for (let arm = 0; arm < ARMS; arm++) {
      for (let i = 0; i < PER_ARM; i++) {
        // each particle spirals in: radius shrinks with i, angle winds with t
        const frac = i / PER_ARM;
        const rr = vmaxR * (1 - frac) + 1;
        const a = (arm / ARMS) * Math.PI * 2 + t * 1.8 + frac * 3.2;
        const col = arm === 0 ? COL.cyan : arm === 1 ? COL.purple : COL.magenta;
        const alpha = 0.25 + 0.75 * frac; // brighter near the core
        px(c, vcx + Math.cos(a) * rr, vcy + Math.sin(a) * rr * 0.7,
          frac > 0.6 ? 2 : 1, col, alpha);
      }
    }
    // bright pulsing core
    c.globalAlpha = 0.7 + 0.3 * Math.sin(t * 6) ** 2;
    c.fillStyle = COL.white;
    c.fillRect(Math.round(vcx) - 1, Math.round(vcy) - 1, 3, 3);
    c.globalAlpha = 1;

    // STATUS: a live neural agent solving a 4x4 pieces-fill puzzle (see
    // createStatusDemo). Header line shows the running state + last result; the
    // body draws the grid, the current placement's selection outline, and a
    // SOLVED/FAILED flash at the end of each episode.
    const st = leftPanels[1];
    statusDemo.update(dt);
    drawStatusDemo(c, st);

    // SIGNAL: a header row ("SIG" + right-aligned %) on its OWN reserved strip,
    // then a spectrum bar graph strictly below it — the bars never reach up into
    // the header, so the readout stays clear of the bars.
    const sg = leftPanels[2];
    const gb = body(sg);
    signalTimer += dt;
    if (signalTimer > 0.12) {
      signalTimer = 0;
      signalValue = 84 + ((Math.random() * 15) | 0);
    }
    drawText(c, "SIG", gb.x, gb.top, 1, "#8a8a9a");
    const pctTxt = `${signalValue}%`;
    drawText(c, pctTxt, gb.x + gb.w - textWidth(pctTxt, 1), gb.top, 1, COL.magenta);
    const barAreaTop = gb.top + GLYPH_H + 3; // a clear gap under the header
    const baseline = gb.bottom;
    const barAreaH = baseline - barAreaTop;
    // Derive the bar count from the width and a fixed pitch so the graph fills
    // edge to edge (a fixed count + floor(width/count) left a gap on the right).
    const pitch = 5;
    const bw = pitch;
    const bars = Math.floor(gb.w / pitch);
    for (let i = 0; i < bars; i++) {
      // a moving spectrum: two sine components so neighbours differ smoothly
      const env = (Math.sin(t * 4 + i * 0.7) * 0.5 + 0.5) * 0.7 +
                  (Math.sin(t * 9 + i * 1.9) * 0.5 + 0.5) * 0.3;
      const bh = Math.max(2, Math.round(env * barAreaH));
      const x = gb.x + i * bw;
      // brighter tip, dimmer body for a little CRT-EQ depth
      c.globalAlpha = 0.85;
      c.fillStyle = COL.magenta;
      c.fillRect(x, baseline - bh, bw - 1, bh);
      c.globalAlpha = 1;
      c.fillStyle = "#ff7af8";
      c.fillRect(x, baseline - bh, bw - 1, 1);
    }
    c.globalAlpha = 1;

    // SECTOR: a code line on top, then a blinking grid that FILLS the rest of
    // the body (sized from the available space, not a fixed small block) with a
    // roaming cursor box.
    const se = leftPanels[3];
    const eb = body(se);
    drawText(c, "07-13", eb.x, eb.top, 1, COL.purple);
    const gridCols = 4, gridRows = 3;
    const gridTop = eb.top + GLYPH_H + 4;
    const gridAreaH = eb.bottom - gridTop;
    // cell pitch derived to fill the body width/height, with a 2px gap
    const pitchX = Math.floor(eb.w / gridCols);
    const pitchY = Math.floor(gridAreaH / gridRows);
    const cellSz = Math.max(4, Math.min(pitchX, pitchY) - 2);
    const gridW = pitchX * gridCols;
    const gridH = pitchY * gridRows;
    const gx0 = Math.round(se.x + (se.w - gridW) / 2);
    const gy0 = Math.round(gridTop + (gridAreaH - gridH) / 2);
    for (let gy = 0; gy < gridRows; gy++) {
      for (let gx = 0; gx < gridCols; gx++) {
        const on = Math.sin(t * 3 + gx * 1.1 + gy * 0.9) > 0.2;
        c.fillStyle = on ? COL.purple : COL.deepBlue;
        c.fillRect(gx0 + gx * pitchX, gy0 + gy * pitchY, cellSz, cellSz);
      }
    }
    if (Math.floor(t * 5) % 2 === 0) {
      const cx = gx0 + (Math.floor(t * 1.8) % gridCols) * pitchX - 1;
      const cy = gy0 + (Math.floor(t * 1.2) % gridRows) * pitchY - 1;
      c.strokeStyle = COL.white;
      c.lineWidth = 1;
      c.strokeRect(cx, cy, cellSz + 2, cellSz + 2);
    }

    // ---- right column -------------------------------------------------------
    for (const p of rightPanels) panelFrame(c, p.x, p.y, p.w, p.h, p.label);

    // NEXT: a preview that advances on a steady tick. The head piece is drawn
    // large and centred; the following two are queued small to its right.
    const nx = rightPanels[0];
    nextTimer += dt;
    if (nextTimer >= NEXT_PERIOD_S) {
      nextTimer -= NEXT_PERIOD_S;
      nextHead = (nextHead + 1) % nextQueue.length;
    }
    const previewType = nextQueue[nextHead];
    const nxb = body(nx);
    // The body splits into a head zone (top) and a queue row (bottom). Reserve
    // ~13px for the queue row; centre the big head piece in what remains.
    const queueRowH = 13;
    const headCY = (nxb.top + (nxb.bottom - queueRowH)) / 2;
    const headCell = 6;
    debrisTet(c, previewType, nx.x + nx.w / 2, headCY, headCell, 1, 1, 0);
    // up-next queue: two small pieces centred as a pair on the bottom row
    const queueY = nxb.bottom - queueRowH / 2;
    for (let q = 1; q <= 2; q++) {
      const qt = nextQueue[(nextHead + q) % nextQueue.length];
      const qx = nx.x + nx.w / 2 + (q === 1 ? -13 : 13);
      debrisTet(c, qt, qx, queueY, 3, 1, 0.6, 0);
    }

    // drift / distance readout — centred in the short label-less strip.
    const dr = rightPanels[1];
    const drb = body(dr);
    const distValue = 11345 + Math.floor(Math.sin(t * 0.8) * 20);
    const lyText = `1.${distValue}LY`;
    c.globalAlpha = 0.7 + 0.3 * Math.sin(t * 5) ** 2;
    // this panel has no label, so centre vertically in its FULL height
    drawText(c, lyText, dr.x + Math.round((dr.w - textWidth(lyText, 1)) / 2),
      textTop(dr.y + dr.h / 2), 1, COL.purple);
    c.globalAlpha = 1;

    // GRAVITY: value line on top, a bar meter centred below it.
    const gr = rightPanels[2];
    const grb = body(gr);
    drawText(c, "9.81 G", grb.x, grb.top, 1, COL.purple);
    const gBars = 5, gBarW = 5, gBarGap = 3;
    const gTotalW = gBars * gBarW + (gBars - 1) * gBarGap;
    const gBarX = Math.round(gr.x + (gr.w - gTotalW) / 2);
    const gBarBase = grb.bottom;
    for (let i = 0; i < gBars; i++) {
      const bh = 6 + Math.sin(t * 4 + i) * 1.5;
      const last = i === gBars - 1;
      c.globalAlpha = last
        ? 0.5 + 0.5 * Math.sin(t * 10) ** 2
        : 0.65 + 0.35 * Math.sin(t * 2 + i) ** 2;
      c.fillStyle = last ? COL.red : COL.purple;
      c.fillRect(gBarX + i * (gBarW + gBarGap), gBarBase - bh, gBarW, bh);
    }
    c.globalAlpha = 1;

    // THREAT: a live radar — concentric rings, a rotating sweep beam that
    // leaves a fading trail, and a blip that flashes as the beam passes it.
    // The "CRITICAL" header pulses. Far more alive than the old static arrow.
    const th = rightPanels[3];
    const thb = body(th);
    c.globalAlpha = 0.55 + 0.45 * Math.sin(t * 6) ** 2; // pulsing header
    drawText(c, "CRITICAL", th.x + Math.round((th.w - textWidth("CRITICAL", 1)) / 2),
      thb.top, 1, COL.red);
    c.globalAlpha = 1;

    const radarTop = thb.top + GLYPH_H + 2;
    const rcx = th.x + th.w / 2;
    const rcy = (radarTop + thb.bottom) / 2;
    const rr = Math.min((thb.bottom - radarTop) / 2, th.w / 2 - 6) - 1;

    // concentric range rings + crosshairs (dim red)
    c.strokeStyle = "#5a1408";
    c.lineWidth = 1;
    for (const f of [1, 0.6, 0.3]) {
      c.beginPath();
      c.arc(rcx, rcy, rr * f, 0, Math.PI * 2);
      c.stroke();
    }
    c.beginPath();
    c.moveTo(rcx - rr, rcy); c.lineTo(rcx + rr, rcy);
    c.moveTo(rcx, rcy - rr); c.lineTo(rcx, rcy + rr);
    c.stroke();

    // sweep beam: a short fan of fading radii behind the leading edge
    const sweep = (t * 2.2) % (Math.PI * 2);
    for (let k = 0; k < 10; k++) {
      const a = sweep - k * 0.13;
      c.globalAlpha = (1 - k / 10) * 0.8;
      c.strokeStyle = COL.red;
      c.beginPath();
      c.moveTo(rcx, rcy);
      c.lineTo(rcx + Math.cos(a) * rr, rcy + Math.sin(a) * rr);
      c.stroke();
    }
    c.globalAlpha = 1;

    // a blip on the scope that flashes brightest as the beam sweeps over it
    const blipA = 2.1; // fixed bearing
    const blipR = rr * 0.62;
    const bx = rcx + Math.cos(blipA) * blipR;
    const by = rcy + Math.sin(blipA) * blipR;
    let da = ((sweep - blipA) % (Math.PI * 2) + Math.PI * 2) % (Math.PI * 2);
    const lit = da < 0.6 ? 1 - da / 0.6 : 0.15; // bright just after the beam
    c.globalAlpha = 0.3 + 0.7 * lit;
    c.fillStyle = lit > 0.5 ? COL.white : COL.red;
    c.fillRect(Math.round(bx) - 1, Math.round(by) - 1, 3, 3);
    c.globalAlpha = 1;

    // PHASE: a big number + rotating mini-L on top, a 4-segment phase progress
    // meter through the middle, and the lines tally at the bottom — three rows
    // spread to fill the body so there's no dead space.
    const pa = rightPanels[4];
    const pb = body(pa);
    const phaseF = (t * 0.65) % 4;
    const phase = 1 + Math.floor(phaseF);
    // row 1: big "0N" + mini-L
    const numCY = pb.top + GLYPH_H + 1;
    drawText(c, `0${phase}`, pb.x, textTop(numCY, 2), 2, COL.red);
    miniTet(c, "L", pa.x + pa.w - 22, Math.round(numCY) - 4, 4);
    // row 2: 4-segment progress meter, current segment pulsing
    const segGap = 3;
    const segW = Math.floor((pb.w - 3 * segGap) / 4);
    const segY = Math.round(pb.cy + 2);
    for (let i = 0; i < 4; i++) {
      const done = i < phase - 1;
      const cur = i === phase - 1;
      c.globalAlpha = done ? 0.9 : cur ? 0.5 + 0.5 * Math.sin(t * 8) ** 2 : 0.25;
      c.fillStyle = done || cur ? COL.red : COL.darkRed;
      c.fillRect(pb.x + i * (segW + segGap), segY, segW, 4);
    }
    c.globalAlpha = 1;
    // row 3: lines tally at the bottom
    drawText(c, `LN ${String(linesCleared).padStart(3, "0")}`, pb.x,
      pb.bottom - GLYPH_H, 1, COL.green);
    c.globalAlpha = 1;
  }

  // Render the STATUS panel's neural puzzle demo: a header line, the 4x4 grid
  // with placed pieces, the current placement's pulsing selection outline, and
  // a SOLVED/FAILED flash at episode end.
  const STATUS_PIECE_COL = [COL.cyan, COL.magenta, COL.green, COL.orange];
  function drawStatusDemo(c, panel) {
    panelFrame(c, panel.x, panel.y, panel.w, panel.h, "STATUS");
    const eb = body(panel);
    const demo = statusDemo;
    const env = demo.env;

    // header: live state on the left, last result on the right
    const stateLabel =
      demo.outcome === "solved" ? "SOLVED" :
      demo.outcome === "failed" ? "FAILED" :
      demo.usingRandom ? "BOOT" :
      demo.exploring ? "EXPLORE" : "SOLVING";
    const stateCol =
      demo.outcome === "solved" ? COL.green :
      demo.outcome === "failed" ? COL.red :
      demo.exploring ? COL.yellow : COL.lightBlue;
    drawText(c, stateLabel, eb.x, eb.top, 1, stateCol);
    drawText(c, demo.lastResult, eb.x + eb.w - textWidth(demo.lastResult, 1), eb.top,
      1, demo.lastResult === "SOLVED" ? COL.green : demo.lastResult === "FAILED" ? COL.red : "#6a6a7a");

    // grid geometry: GRID(=10) wide x GRID_H(=4) tall, cells sized to FIT the
    // panel body (the limiting axis wins so it never overflows), centred.
    const gridTop = eb.top + GLYPH_H + 3;
    const gridAreaH = eb.bottom - gridTop;
    const cell = Math.max(2, Math.min(Math.floor(eb.w / GRID), Math.floor(gridAreaH / GRID_H)));
    const gw = cell * GRID;
    const gh = cell * GRID_H;
    const gx0 = Math.round(panel.x + (panel.w - gw) / 2);
    const gy0 = Math.round(gridTop + (gridAreaH - gh) / 2);

    // grid background + placed cells
    for (let y = 0; y < GRID_H; y++) {
      for (let x = 0; x < GRID; x++) {
        const px0 = gx0 + x * cell;
        const py0 = gy0 + y * cell;
        const v = env.grid[y * GRID + x];
        c.fillStyle = v ? STATUS_PIECE_COL[(v - 1) % STATUS_PIECE_COL.length] : "#161622";
        c.fillRect(px0, py0, cell - 1, cell - 1);
      }
    }

    // selection outline on the chosen placement, pulsing while in the select
    // phase, then a solid commit flash during place.
    if ((demo.phase === "select" || demo.phase === "place") && demo.chosenCells) {
      const pulse = demo.phase === "select"
        ? 0.4 + 0.6 * Math.sin(demo.selectPulse * Math.PI * 3) ** 2
        : 1;
      c.globalAlpha = pulse;
      c.strokeStyle = COL.white;
      c.lineWidth = 1;
      for (const cc of demo.chosenCells) {
        c.strokeRect(gx0 + cc.x * cell + 0.5, gy0 + cc.y * cell + 0.5, cell - 1, cell - 1);
      }
      c.globalAlpha = 1;
    }

    // end-of-episode flash overlay tinting the grid
    if (demo.outcome) {
      c.globalAlpha = 0.18 + 0.12 * Math.sin(performance.now() / 90);
      c.fillStyle = demo.outcome === "solved" ? COL.green : COL.red;
      c.fillRect(gx0, gy0, gw, gh);
      c.globalAlpha = 1;
    }
  }

  // The two leaderboard panels in the lower corners. RANKING shows persisted
  // records; LIVE shows only in-progress players. Rows are "RR NAME  SCORE":
  // rank+name left-aligned, score right-aligned, score green for live entries.
  function drawLeaderboards(c) {
    const persisted = entries.filter((e) => e && !e.live);
    const live = entries.filter((e) => e && e.live);

    function board(panel, rows, emptyMsg) {
      panelFrame(c, panel.x, panel.y, panel.w, panel.h, panel.label);
      const b = body(panel); // shared reference frame (area below the label)
      const x = b.x;
      const right = b.x + b.w;

      // Empty state: a single centred message instead of a column of dashes,
      // so a board with no entries (common for LIVE in the lobby) reads as
      // intentional rather than broken/empty.
      if (rows.length === 0) {
        for (let i = 0; i < emptyMsg.length; i++) {
          const line = emptyMsg[i];
          const ly = Math.round(b.cy - (emptyMsg.length * 9) / 2 + i * 9);
          drawText(c, line, panel.x + Math.round((panel.w - textWidth(line, 1)) / 2),
            ly, 1, "#5a5a6a");
        }
        return;
      }

      // Distribute LB_ROWS evenly across the body. Each row's baseline is the
      // centre of its slot minus half a glyph, rounded independently so the
      // rounding error never accumulates down the column (which is what made
      // the spacing drift). rank+name and score share that exact baseline.
      const slotH = b.h / LB_ROWS;
      for (let i = 0; i < LB_ROWS; i++) {
        const y = Math.round(b.top + slotH * (i + 0.5) - GLYPH_H / 2);
        const e = rows[i];
        const rank = String(i + 1).padStart(2, "0");
        if (!e) {
          drawText(c, `${rank} ---`, x, y, 1, "#3a3a3a");
          continue;
        }
        const nameColor = e.live ? COL.green : COL.cyan;
        const scColor = e.live ? COL.green : COL.white;
        drawText(c, `${rank} ${String(e.name).slice(0, 4)}`, x, y, 1, nameColor);
        const sc = String(Math.max(0, e.score | 0)).padStart(6, "0");
        drawText(c, sc, right - textWidth(sc, 1), y, 1, scColor);
      }
      c.globalAlpha = 1;
    }

    board(rankPanel, persisted, ["NO", "RECORDS", "YET"]);
    board(livePanel, live, ["NO LIVE", "PLAYERS"]);
  }

  function drawTitle(c, t, intro) {
    const cx = w / 2;

    // subtitle above the wordmark
    c.globalAlpha = (0.75 + Math.sin(t * 4) * 0.15) * intro;
    const sub = "DEEP SPACE ANOMALY";
    drawText(c, sub, cx - textWidth(sub, 1) / 2, titleY, 1, COL.purple);

    // EVENT HORIZON wordmark with red/cyan chromatic-aberration ghosts
    const title = "EVENT HORIZON";
    const ts = 2;
    const tw = textWidth(title, ts);
    const tx = cx - tw / 2;
    const ty = titleY + 12;

    if (Math.sin(t * 25) > 0.88) {
      c.globalAlpha = 0.6 * intro;
      drawText(c, title, tx + 2, ty, ts, COL.red);
    }
    if (Math.sin(t * 21) > 0.92) {
      c.globalAlpha = 0.5 * intro;
      drawText(c, title, tx - 2, ty, ts, COL.cyan);
    }
    c.globalAlpha = (0.95 + Math.sin(t * 11) * 0.05) * intro;
    drawText(c, title, tx, ty, ts, COL.white);
    c.globalAlpha = 1;
  }

  return {
    render,
    pokeActivity,
    // expose layout so app.js can place the controls overlay relative to the scene
    layout: { MAIN_X, MAIN_Y, wellW, MAIN_H, BH, titleY, w, h },
    // 0 while the boot sequence is on screen, ramping to 1 just after hand-over.
    // app.js uses this to fade the controls console in once the scene is live.
    // Measured from the current boot epoch, so an idle reboot hides it again.
    consoleReveal(t) {
      return easeOutCubic(clamp01((t - bootEpoch - bootDuration) / 0.6));
    },
  };
}
