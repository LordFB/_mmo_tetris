/**
 * MMO Tetris — vanilla client (full-screen NES CRT).
 *
 * The ENTIRE screen is one CRT. Every element from the layout — your board in
 * the centre, the two neighbour boards in the lower corners, the TOP-10
 * leaderboard upper-left, the NEXT box upper-right, and the score/level/lines
 * and player panels along the bottom — is drawn as crisp NES pixel art onto a
 * single low-res composite canvas (#source), which the CRT shader then displays
 * on #game with scanlines, curvature, mask and glow.
 *
 * Gameplay is the faithful NES engine at a fixed 60.0988 Hz timestep. Inputs are
 * recorded per frame; on game-over we submit { seed, startLevel, inputs } and the
 * server re-simulates to derive the authoritative score (replay anti-cheat).
 */

import {
  createInitialState,
  step,
  WIDTH,
  HEIGHT,
  cellsOf,
  tileColor,
  levelColors,
  NES_WHITE,
  NTSC_FPS,
  INPUT_LEFT,
  INPUT_RIGHT,
  INPUT_DOWN,
  INPUT_ROTATE_CW,
  INPUT_ROTATE_CCW,
  MAX_REPLAY_FRAMES,
} from "./nes-engine.js";
import { createCrt } from "./crt.js";

// ---------------------------------------------------------------------------
// Virtual composite screen. Low-res so it reads as NES pixel art; the CRT and
// the browser scale it up. 4:3-ish to suit a tube. All panel coordinates below
// are in these virtual pixels.
// ---------------------------------------------------------------------------
const VW = 512;
const VH = 448;

const source = document.querySelector("#source");
source.width = VW;
source.height = VH;
const ctx = source.getContext("2d");
ctx.imageSmoothingEnabled = false;

const glCanvas = document.querySelector("#game");
let crt = null;

// ---- lobby DOM (text entry can't live on a canvas) ----
const lobby = document.querySelector("#lobby");
const lobbyMsg = document.querySelector("#lobby-msg");
const statusEl = document.querySelector("#status");
const nameInput = document.querySelector("#player-name");
const startLevelInput = document.querySelector("#start-level");
const startBtn = document.querySelector("#start");

nameInput.value = localStorage.getItem("mmo-tetris.player-name") || "ANON";
startLevelInput.value = localStorage.getItem("mmo-tetris.start-level") || "0";

// ---------------------------------------------------------------------------
// Layout (virtual pixels). Mirrors the mockup.
//
// The CRT barrel curvature pushes the outer edges outward, so the composite
// keeps a safe margin top and bottom; nothing is drawn past these or it gets
// clipped by the tube. The deepest elements (the neighbour player strips and
// the score panel) are anchored to BOTTOM so they always sit inside the margin.
// ---------------------------------------------------------------------------
const TOP_MARGIN = 16;
const BOTTOM_MARGIN = 30; // generous: the bottom curve is the worst offender
const STRIP_H = 24; // player-name strip under each neighbour board

const MAIN_CELL = 16; // your board (10x20 -> 160x320)
const NB_CELL = 8; // neighbour boards (10x20 -> 80x160)

const nbBoardH = HEIGHT * NB_CELL;
// Neighbour board top so that board + 6px gap + strip ends at the bottom margin.
const nbTop = VH - BOTTOM_MARGIN - STRIP_H - 6 - nbBoardH;

const mainBoardH = HEIGHT * MAIN_CELL;
const scorePanelH = 38;
// Main board top so that board + 6px gap + score panel ends at the bottom margin.
const mainTop = VH - BOTTOM_MARGIN - scorePanelH - 8 - mainBoardH;

const L = {
  main: { x: (VW - WIDTH * MAIN_CELL) / 2, y: Math.max(TOP_MARGIN + 24, mainTop), cell: MAIN_CELL },
  left: { x: 16, y: nbTop, cell: NB_CELL },
  right: { x: VW - 16 - WIDTH * NB_CELL, y: nbTop, cell: NB_CELL },
  board: { /* derived */ },
  leaderboard: { x: 14, y: TOP_MARGIN, w: 150 },
  next: { x: VW - 14 - 96, y: TOP_MARGIN + 14, w: 96, h: 72 },
  scorePanelH,
  stripH: STRIP_H,
};
L.board.w = WIDTH * MAIN_CELL;
L.board.h = HEIGHT * MAIN_CELL;

// ---------------------------------------------------------------------------
// game / loop state
// ---------------------------------------------------------------------------
const KEY_BITS = {
  ArrowLeft: INPUT_LEFT,
  ArrowRight: INPUT_RIGHT,
  ArrowDown: INPUT_DOWN,
  z: INPUT_ROTATE_CCW,
  Z: INPUT_ROTATE_CCW,
  x: INPUT_ROTATE_CW,
  X: INPUT_ROTATE_CW,
  ArrowUp: INPUT_ROTATE_CW,
};
const heldBits = new Set();

let state = null;
let seed = 0;
let startLevel = 0;
let running = false;
let replay = null;
let replayLen = 0;
let submitted = false;

// --- precise fixed-timestep clock ---
// The simulation advances in exact NES frames. We accumulate real elapsed time
// (from performance.now(), the monotonic high-resolution clock) and consume it
// one FRAME_MS at a time, so the tick rate is independent of the display's
// refresh rate. The accumulator is double precision and never rounded, so there
// is no long-term drift: over an hour the error stays below one frame.
const FRAME_MS = 1000 / NTSC_FPS; // 16.639... ms, the true NTSC frame period
const MAX_CATCHUP_FRAMES = 8; // cap per render to avoid a spiral of death
let accumulator = 0; // unconsumed real time, in ms (fractional)
let simClock = 0; // performance.now() of the last consumed frame boundary
let paused = false; // true while the tab is hidden

const SNAPSHOT_EVERY = 6; // stream a frame-stamped snapshot ~10x/sec
let snapshotCounter = 0;

// shared-room data pushed by server
let players = 0;
let myPosition = 0; // 1-based rank, 0 = unranked
let leaderboard = [];
let leftSnap = null;
let rightSnap = null;
let leftName = "—";
let rightName = "—";
let myName = "ANON";
// Frame of the snapshot currently shown per side, for ordered updates: a
// snapshot stamped with an older frame than what we already show is dropped
// (out-of-order delivery), unless the neighbouring player changed.
const shownFrame = { left: -1, right: -1 };

function updateNeighbour(side, info) {
  const isLeft = side === "left";
  const prevName = isLeft ? leftName : rightName;
  const name = info?.name ?? "—";
  const snap = info?.snap ?? null;
  const changedPlayer = name !== prevName;

  // Drop a stale frame for the SAME player; always accept a player change.
  if (!changedPlayer && snap && snap.frame !== undefined &&
      snap.frame < shownFrame[side]) {
    return;
  }
  shownFrame[side] = snap?.frame ?? -1;

  if (isLeft) { leftName = name; leftSnap = snap; }
  else { rightName = name; rightSnap = snap; }
}

// networking
let socket;
let reconnectTimer;

// ---------------------------------------------------------------------------
// NES text — a compact 5x7 pixel font so labels read as console text, not as
// the browser's font (which would break the illusion under the CRT).
// ---------------------------------------------------------------------------
const FONT = buildFont();
const GLYPH_W = 5;
const GLYPH_H = 7;

function drawText(c, text, x, y, scale, color) {
  c.fillStyle = color;
  let cx = x;
  const s = scale;
  for (const ch of String(text).toUpperCase()) {
    if (ch === " ") { cx += (GLYPH_W + 1) * s; continue; }
    const glyph = FONT[ch];
    if (glyph) {
      for (let gy = 0; gy < GLYPH_H; gy++) {
        const row = glyph[gy];
        for (let gx = 0; gx < GLYPH_W; gx++) {
          if (row & (1 << (GLYPH_W - 1 - gx))) {
            c.fillRect(cx + gx * s, y + gy * s, s, s);
          }
        }
      }
    }
    cx += (GLYPH_W + 1) * s;
  }
  return cx;
}

function textWidth(text, scale) {
  return String(text).length * (GLYPH_W + 1) * scale;
}

// ---------------------------------------------------------------------------
// block rendering — the NES tile: solid colour fill with a white highlight in
// the upper-left and a 1px dark inset, cells flush together.
// ---------------------------------------------------------------------------
function drawBlock(c, x, y, cell, color, dim) {
  // The authentic NES block: 1px black gridline border, solid colour fill, and
  // a white corner accent (upper-left square + thin top/left edges) that gives
  // the block its glossy, recognisable look.
  c.fillStyle = "#000";
  c.fillRect(x, y, cell, cell);
  c.fillStyle = color;
  c.fillRect(x + 1, y + 1, cell - 2, cell - 2);

  c.fillStyle = dim ? "rgba(252,252,252,0.5)" : NES_WHITE;
  // upper-left highlight square
  const hl = Math.max(2, Math.floor(cell / 3.5));
  c.fillRect(x + 2, y + 2, hl, hl);
  // thin glossy edges along the top and left inside the fill
  if (cell >= 12) {
    c.fillRect(x + 2, y + 2, cell - 5, 1);
    c.fillRect(x + 2, y + 2, 1, cell - 5);
  }
}

// ---------------------------------------------------------------------------
// panels
// ---------------------------------------------------------------------------
function frameBox(c, x, y, w, h) {
  c.fillStyle = "#000";
  c.fillRect(x, y, w, h);
  c.strokeStyle = NES_WHITE;
  c.lineWidth = 2;
  c.strokeRect(x + 1, y + 1, w - 2, h - 2);
}

function renderMainBoard() {
  const { x, y, cell } = L.main;
  // bordered well
  frameBox(c0(), x - 4, y - 4, L.board.w + 8, L.board.h + 8);
  const level = state ? state.level : 0;
  const flash = state && state.phase === "line_clear" &&
    (Math.floor(performance.now() / 60) % 2 === 0);

  for (let by = 0; by < HEIGHT; by++) {
    const rowClearing = state && state.phase === "line_clear" &&
      state.clearingRows.includes(by);
    for (let bx = 0; bx < WIDTH; bx++) {
      const v = state ? state.board[by * WIDTH + bx] : 0;
      const px = x + bx * cell;
      const py = y + by * cell;
      if (rowClearing && flash) {
        ctx.fillStyle = NES_WHITE;
        ctx.fillRect(px, py, cell, cell);
      } else if (v) {
        drawBlock(ctx, px, py, cell, tileColor(v - 1, level), false);
      }
    }
  }

  if (state && state.phase === "active") {
    const p = state.activePiece;
    for (const cc of cellsOf(p.id, p.rotation)) {
      const bx = p.x + cc.x;
      const by = p.y + cc.y;
      if (by >= 0 && by < HEIGHT && bx >= 0 && bx < WIDTH) {
        drawBlock(ctx, x + bx * cell, y + by * cell, cell, tileColor(p.id, level), false);
      }
    }
  }
}

function renderNeighbour(panel, snap, name, label) {
  const { x, y, cell } = panel;
  frameBox(c0(), x - 3, y - 3, WIDTH * cell + 6, HEIGHT * cell + 6);
  if (!snap || !snap.board) {
    drawText(ctx, "WAITING", x + 6, y + HEIGHT * cell / 2 - 4, 1, "#5a5a5a");
  } else {
    const level = snap.level || 0;
    for (let by = 0; by < HEIGHT; by++) {
      for (let bx = 0; bx < WIDTH; bx++) {
        const v = snap.board[by * WIDTH + bx];
        if (v) drawBlock(ctx, x + bx * cell, y + by * cell, cell, tileColor(v - 1, level), true);
      }
    }
    if (snap.gameOver) {
      ctx.fillStyle = "rgba(0,0,0,0.6)";
      ctx.fillRect(x, y, WIDTH * cell, HEIGHT * cell);
      drawText(ctx, "TOPPED", x + 6, y + HEIGHT * cell / 2 - 8, 1, NES_RED);
      drawText(ctx, "OUT", x + 6, y + HEIGHT * cell / 2 + 2, 1, NES_RED);
    }
  }
  // player strip under the board
  const stripY = y + HEIGHT * cell + 6;
  frameBox(c0(), x - 3, stripY, WIDTH * cell + 6, L.stripH);
  drawText(ctx, name.slice(0, 12), x + 2, stripY + 3, 1, NES_CYAN);
  drawText(
    ctx,
    snap ? `${pad(snap.score, 6)}` : "------",
    x + 2,
    stripY + 13,
    1,
    NES_WHITE,
  );
  drawText(ctx, snap ? `LV${snap.level ?? 0}` : "", x + WIDTH * cell - 28, stripY + 13, 1, "#9a9a9a");
}

const NES_GREEN = "#58f898";

function renderLeaderboard() {
  const { x, y } = L.leaderboard;
  drawText(ctx, "TOP 10", x, y, 2, NES_WHITE);
  // blink phase for the live indicator dot
  const blink = Math.floor(performance.now() / 400) % 2 === 0;
  let ly = y + 22;
  for (let i = 0; i < 10; i++) {
    const e = leaderboard[i];
    const rank = pad2(i + 1);
    if (e) {
      // live (in-progress) players are green; persisted records are cyan
      const nameColor = e.live ? NES_GREEN : NES_CYAN;
      drawText(ctx, `${rank} ${e.name.slice(0, 8)}`, x, ly, 1, nameColor);
      const sc = pad(e.score, 6);
      drawText(ctx, sc, x + 140 - textWidth(sc, 1), ly, 1, e.live ? NES_GREEN : NES_WHITE);
      // small blinking dot marks a currently-playing entry
      if (e.live && blink) {
        ctx.fillStyle = NES_GREEN;
        ctx.fillRect(x + 142, ly + 1, 3, 5);
      }
    } else {
      drawText(ctx, `${rank} ---`, x, ly, 1, "#3a3a3a");
    }
    ly += 12;
  }
}

function renderNext() {
  const { x, y, w, h } = L.next;
  drawText(ctx, "NEXT", x + w / 2 - textWidth("NEXT", 2) / 2, y - 14, 2, NES_WHITE);
  frameBox(c0(), x, y, w, h);
  if (!state) return;
  const level = state.level;
  const cells = cellsOf(state.nextPiece, 0);
  const cell = 14;
  let minX = 9, minY = 9, maxX = -9, maxY = -9;
  for (const cc of cells) {
    minX = Math.min(minX, cc.x); maxX = Math.max(maxX, cc.x);
    minY = Math.min(minY, cc.y); maxY = Math.max(maxY, cc.y);
  }
  const pw = (maxX - minX + 1) * cell;
  const ph = (maxY - minY + 1) * cell;
  const ox = x + (w - pw) / 2 - minX * cell;
  const oy = y + (h - ph) / 2 - minY * cell;
  for (const cc of cells) {
    drawBlock(ctx, ox + cc.x * cell, oy + cc.y * cell, cell, tileColor(state.nextPiece, level), false);
  }
}

function renderScorePanel() {
  // Bottom-centre panel: POSITION | LEVEL SCORE LINES
  const panelW = L.board.w + 8;
  const x = L.main.x - 4;
  const y = L.main.y + L.board.h + 8;
  const h = L.scorePanelH;
  frameBox(c0(), x, y, panelW, h);

  // POSITION big on the left
  const posText = myPosition > 0 ? `#${myPosition}` : "#-";
  drawText(ctx, "POSITION", x + 6, y + 6, 1, NES_CYAN);
  drawText(ctx, posText, x + 6, y + 17, 2, NES_WHITE);

  // LEVEL / SCORE / LINES stacked on the right
  const rx = x + 74;
  const level = state ? state.level : 0;
  const sc = state ? pad(state.score, 6) : "000000";
  const ln = state ? pad(state.lines, 3) : "000";
  drawText(ctx, `LEVEL ${level}`, rx, y + 4, 1, NES_WHITE);
  drawText(ctx, `SCORE ${sc}`, rx, y + 15, 1, NES_CYAN);
  drawText(ctx, `LINES ${ln}`, rx, y + 26, 1, NES_WHITE);

  // ONLINE count, in the gap between the leaderboard and the left neighbour,
  // well inside the safe margins so nothing is clipped or collides.
  const onlineTxt = `ONLINE ${players}`;
  drawText(ctx, onlineTxt, L.leaderboard.x, L.left.y - 16, 1, "#9a9a9a");
}

// small colour constants used above (defined after to keep imports tidy)
const NES_CYAN = "#3cbcfc";
const NES_RED = "#f83800";

// helper so panel fns read clearly; the 2D context for frames
function c0() { return ctx; }

function pad(n, len) { return String(Math.max(0, n | 0)).padStart(len, "0"); }
function pad2(n) { return String(n).padStart(2, "0"); }

// ---------------------------------------------------------------------------
// compose the whole frame
// ---------------------------------------------------------------------------
function renderScene() {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, VW, VH);

  renderLeaderboard();
  renderNext();
  renderNeighbour(L.left, leftSnap, leftName, "");
  renderNeighbour(L.right, rightSnap, rightName, "");
  renderMainBoard();
  renderScorePanel();

  if (!running && state && state.phase === "game_over") {
    // subtle game-over banner on the board itself
    const bx = L.main.x;
    const by = L.main.y + L.board.h / 2 - 14;
    ctx.fillStyle = "rgba(0,0,0,0.7)";
    ctx.fillRect(bx, by - 4, L.board.w, 36);
    const t = "GAME OVER";
    drawText(ctx, t, bx + (L.board.w - textWidth(t, 2)) / 2, by, 2, NES_RED);
  }
}

// ---------------------------------------------------------------------------
// game loop — fixed timestep
// ---------------------------------------------------------------------------
function currentInputBits() {
  let bits = 0;
  for (const b of heldBits) bits |= b;
  return bits;
}

function startGame() {
  seed = Math.floor(Math.random() * 0xffff) & 0xffff;
  startLevel = clampLevel(parseInt(startLevelInput.value, 10) || 0);
  myName = nameInput.value.trim().slice(0, 16) || "ANON";
  localStorage.setItem("mmo-tetris.player-name", myName);
  localStorage.setItem("mmo-tetris.start-level", String(startLevel));
  state = createInitialState(seed, startLevel);
  replay = new Uint8Array(MAX_REPLAY_FRAMES);
  replayLen = 0;
  accumulator = 0;
  simClock = performance.now();
  paused = false;
  submitted = false;
  running = true;
  snapshotCounter = 0;
  lobby.classList.add("hidden");
  send({ type: "start" });
}

function clampLevel(v) { return Math.max(0, Math.min(29, v | 0)); }

function endGame() {
  running = false;
  submitPlay();
  lobbyMsg.textContent = `GAME OVER — ${pad(state.score, 6)}`;
  statusEl.textContent = "VERIFYING…";
  lobby.classList.remove("hidden");
}

function tickSimulation() {
  const bits = currentInputBits();
  if (replayLen < replay.length) replay[replayLen++] = bits;
  state = step(state, bits);

  if (++snapshotCounter >= SNAPSHOT_EVERY) {
    snapshotCounter = 0;
    sendSnapshot();
  }
  if (state.phase === "game_over") endGame();
}

function frame(time) {
  requestAnimationFrame(frame);

  if (running && !paused) {
    // Advance the simulation by exact NES frames using the high-resolution
    // monotonic clock. rAF's own timestamp is ignored for the sim so the tick
    // rate never inherits the display refresh rate.
    const now = performance.now();
    accumulator += now - simClock;
    simClock = now;

    // Spiral-of-death guard: if we somehow owe a huge backlog (a long stall
    // that wasn't a visibility pause), drop the excess but keep the fractional
    // remainder so we don't introduce sub-frame drift.
    const maxBacklog = FRAME_MS * MAX_CATCHUP_FRAMES;
    if (accumulator > maxBacklog) accumulator = maxBacklog;

    let steps = 0;
    while (accumulator >= FRAME_MS && running && steps < MAX_CATCHUP_FRAMES) {
      accumulator -= FRAME_MS;
      steps++;
      tickSimulation();
    }
  }

  renderScene();
  if (crt) crt.render(time / 1000);
}

// Pause cleanly when the tab is hidden: rAF stops firing anyway, so we freeze
// the clock and resume from exactly where we left off — no real time is folded
// into the accumulator, so the replay stays frame-exact.
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    paused = true;
  } else {
    paused = false;
    simClock = performance.now(); // discard the hidden interval entirely
    accumulator = 0;
  }
});

// ---------------------------------------------------------------------------
// input
// ---------------------------------------------------------------------------
window.addEventListener("keydown", (event) => {
  if (event.repeat) return;
  if (event.key === "Enter") {
    if (!running && document.activeElement?.tagName !== "INPUT") startGame();
    return;
  }
  const bit = KEY_BITS[event.key];
  if (bit !== undefined && running) {
    event.preventDefault();
    heldBits.add(bit);
  }
});
window.addEventListener("keyup", (event) => {
  const bit = KEY_BITS[event.key];
  if (bit !== undefined) { event.preventDefault(); heldBits.delete(bit); }
});
window.addEventListener("blur", () => heldBits.clear());
startBtn.addEventListener("click", () => { if (!running) startGame(); });

// ---------------------------------------------------------------------------
// canvas sizing — keep the CRT crisp and full-viewport
// ---------------------------------------------------------------------------
// The CRT fills the WHOLE viewport so the entire screen is the tube. The game
// content is kept at a classic 4:3 aspect INSIDE the shader (the composite is
// mapped into a centred 4:3 region), so blocks are never stretched no matter the
// monitor's aspect, while scanlines / mask / curvature still cover the full
// screen — the area around the picture reads as the dark inner glass of the set.
const TUBE_ASPECT = 4 / 3;

function resize() {
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  glCanvas.style.position = "fixed";
  glCanvas.style.left = "0";
  glCanvas.style.top = "0";
  glCanvas.style.width = vw + "px";
  glCanvas.style.height = vh + "px";

  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  glCanvas.width = Math.max(1, Math.floor(vw * dpr));
  glCanvas.height = Math.max(1, Math.floor(vh * dpr));
}
window.addEventListener("resize", resize);

// ---------------------------------------------------------------------------
// networking
// ---------------------------------------------------------------------------
function send(message) {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
}

function sendSnapshot() {
  if (!state || socket?.readyState !== WebSocket.OPEN) return;
  const flat = Array.from(state.board);
  if (state.phase === "active") {
    for (const cc of cellsOf(state.activePiece.id, state.activePiece.rotation)) {
      const bx = state.activePiece.x + cc.x;
      const by = state.activePiece.y + cc.y;
      if (by >= 0 && by < HEIGHT && bx >= 0 && bx < WIDTH) {
        flat[by * WIDTH + bx] = state.activePiece.id + 1;
      }
    }
  }
  send({
    type: "snapshot",
    frame: state.frame, // frame stamp: lets receivers order/drop stale snapshots
    board: flat,
    score: state.score,
    lines: state.lines,
    level: state.level,
    gameOver: state.phase === "game_over",
  });
}

function submitPlay() {
  if (submitted || !state) return;
  submitted = true;
  send({
    type: "play_complete",
    seed,
    startLevel,
    inputs: Array.from(replay.subarray(0, replayLen)),
  });
}

function connect() {
  clearTimeout(reconnectTimer);
  if (socket) socket.close();
  const endpoint = new URL(window.location.href);
  endpoint.protocol = endpoint.protocol === "https:" ? "wss:" : "ws:";
  endpoint.pathname = "/ws";
  endpoint.search = "";
  socket = new WebSocket(endpoint);
  statusEl.textContent = "CONNECTING…";

  socket.addEventListener("open", () => {
    const name = nameInput.value.trim().slice(0, 16) || "ANON";
    send({ type: "hello", name });
    statusEl.textContent = "● ONLINE";
  });
  socket.addEventListener("message", (event) => {
    let m; try { m = JSON.parse(event.data); } catch { return; }
    handleMessage(m);
  });
  socket.addEventListener("close", () => {
    statusEl.textContent = "RECONNECTING…";
    reconnectTimer = setTimeout(connect, 1500);
  });
}

function handleMessage(m) {
  switch (m.type) {
    case "welcome":
    case "presence":
      if (m.players !== undefined) players = m.players;
      if (m.entries) leaderboard = m.entries;
      break;
    case "leaderboard":
      leaderboard = m.entries || [];
      break;
    case "neighbours":
      updateNeighbour("left", m.left);
      updateNeighbour("right", m.right);
      if (m.position !== undefined) myPosition = m.position;
      break;
    case "play_accepted":
      statusEl.textContent = `✓ VERIFIED ${pad(m.score, 6)}`;
      break;
    case "play_rejected":
      statusEl.textContent = `✕ ${String(m.reason || "rejected").toUpperCase()}`;
      break;
    case "error":
      statusEl.textContent = String(m.error || "error").toUpperCase();
      break;
  }
}

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------
function buildFont() {
  // 5x7 uppercase pixel font. Each glyph is 7 rows of 5-bit masks.
  const F = {};
  const def = (ch, rows) => { F[ch] = rows; };
  def("A", [0x0e,0x11,0x11,0x1f,0x11,0x11,0x11]);
  def("B", [0x1e,0x11,0x11,0x1e,0x11,0x11,0x1e]);
  def("C", [0x0e,0x11,0x10,0x10,0x10,0x11,0x0e]);
  def("D", [0x1e,0x11,0x11,0x11,0x11,0x11,0x1e]);
  def("E", [0x1f,0x10,0x10,0x1e,0x10,0x10,0x1f]);
  def("F", [0x1f,0x10,0x10,0x1e,0x10,0x10,0x10]);
  def("G", [0x0e,0x11,0x10,0x17,0x11,0x11,0x0f]);
  def("H", [0x11,0x11,0x11,0x1f,0x11,0x11,0x11]);
  def("I", [0x0e,0x04,0x04,0x04,0x04,0x04,0x0e]);
  def("J", [0x07,0x02,0x02,0x02,0x02,0x12,0x0c]);
  def("K", [0x11,0x12,0x14,0x18,0x14,0x12,0x11]);
  def("L", [0x10,0x10,0x10,0x10,0x10,0x10,0x1f]);
  def("M", [0x11,0x1b,0x15,0x15,0x11,0x11,0x11]);
  def("N", [0x11,0x19,0x15,0x13,0x11,0x11,0x11]);
  def("O", [0x0e,0x11,0x11,0x11,0x11,0x11,0x0e]);
  def("P", [0x1e,0x11,0x11,0x1e,0x10,0x10,0x10]);
  def("Q", [0x0e,0x11,0x11,0x11,0x15,0x12,0x0d]);
  def("R", [0x1e,0x11,0x11,0x1e,0x14,0x12,0x11]);
  def("S", [0x0f,0x10,0x10,0x0e,0x01,0x01,0x1e]);
  def("T", [0x1f,0x04,0x04,0x04,0x04,0x04,0x04]);
  def("U", [0x11,0x11,0x11,0x11,0x11,0x11,0x0e]);
  def("V", [0x11,0x11,0x11,0x11,0x11,0x0a,0x04]);
  def("W", [0x11,0x11,0x11,0x15,0x15,0x1b,0x11]);
  def("X", [0x11,0x11,0x0a,0x04,0x0a,0x11,0x11]);
  def("Y", [0x11,0x11,0x0a,0x04,0x04,0x04,0x04]);
  def("Z", [0x1f,0x01,0x02,0x04,0x08,0x10,0x1f]);
  def("0", [0x0e,0x11,0x13,0x15,0x19,0x11,0x0e]);
  def("1", [0x04,0x0c,0x04,0x04,0x04,0x04,0x0e]);
  def("2", [0x0e,0x11,0x01,0x02,0x04,0x08,0x1f]);
  def("3", [0x1f,0x02,0x04,0x02,0x01,0x11,0x0e]);
  def("4", [0x02,0x06,0x0a,0x12,0x1f,0x02,0x02]);
  def("5", [0x1f,0x10,0x1e,0x01,0x01,0x11,0x0e]);
  def("6", [0x06,0x08,0x10,0x1e,0x11,0x11,0x0e]);
  def("7", [0x1f,0x01,0x02,0x04,0x08,0x08,0x08]);
  def("8", [0x0e,0x11,0x11,0x0e,0x11,0x11,0x0e]);
  def("9", [0x0e,0x11,0x11,0x0f,0x01,0x02,0x0c]);
  def("#", [0x0a,0x0a,0x1f,0x0a,0x1f,0x0a,0x0a]);
  def("-", [0x00,0x00,0x00,0x1f,0x00,0x00,0x00]);
  def(".", [0x00,0x00,0x00,0x00,0x00,0x0c,0x0c]);
  def("·", [0x00,0x00,0x00,0x04,0x00,0x00,0x00]);
  def("✓", [0x00,0x01,0x02,0x14,0x08,0x00,0x00]);
  def("✕", [0x00,0x11,0x0a,0x04,0x0a,0x11,0x00]);
  return F;
}

resize();
crt = createCrt(glCanvas, source);
if (!crt) source.classList.add("visible-fallback");
lobbyMsg.textContent = "DROP INTO THE ROOM";
connect();
requestAnimationFrame(frame);
