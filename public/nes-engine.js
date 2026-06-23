/**
 * Faithful NES (NTSC) Tetris engine — a single dependency-free ESM module.
 *
 * This is a 1:1 port of packages/nes-engine (TypeScript) into plain ESM so the
 * EXACT SAME CODE runs in the browser (the player's client) and in Node (the
 * server's replay verifier). That shared determinism is the whole anti-cheat
 * model: the client sends only { seed, startLevel, inputs[] }, and the server
 * re-runs this engine over those inputs to derive the authoritative score. A
 * forged score cannot survive re-simulation.
 *
 * Determinism rules (do not break): integer-only on every path that affects
 * state, no Date, no Math.random, no floating point in state transitions.
 */

// ---------------------------------------------------------------------------
// pieces — Nintendo Rotation System, right-handed, no kicks
// ---------------------------------------------------------------------------

export const PIECE_T = 0;
export const PIECE_J = 1;
export const PIECE_Z = 2;
export const PIECE_O = 3;
export const PIECE_S = 4;
export const PIECE_L = 5;
export const PIECE_I = 6;
export const PIECE_COUNT = 7;

/** Canonical NES spawn-table order: T J Z O S L I. */
export const PIECE_NAMES = ["T", "J", "Z", "O", "S", "L", "I"];

/** Renderer colours, indexed by PieceId. NES uses 3 colours; we tint by piece. */
export const PIECE_COLORS = [
  "#d365ff", // T
  "#53aeff", // J
  "#ff5757", // Z
  "#ffe000", // O
  "#43f611", // S
  "#fa9e00", // L
  "#2cd5f6", // I
];

/**
 * Authentic NES (NTSC) per-level block colours.
 *
 * The original game stores four PPU palette indices per level at ROM $984C and
 * reuses them every 10 levels. Index 0 is the (ignored) transparent slot and
 * index 1 is always white; indices 2 and 3 are the two coloured tiles. Below
 * are those two coloured slots per level, converted from the 2C02 PPU indices
 * to sRGB using the canonical Nintendo palette. White is shared and constant.
 *
 * Source: meatfighter NES Tetris disassembly ($984C table) + NESdev 2C02 palette.
 */
export const NES_WHITE = "#fcfcfc";
export const NES_BLACK = "#000000";

/** [color1 (PPU "blue" slot), color2 (PPU "red" slot)] for levels 0..9. */
export const LEVEL_PALETTES = [
  ["#3cbcfc", "#0058f8"], // 0: cyan / blue
  ["#b8f818", "#00b800"], // 1: yellow-green / green
  ["#f878f8", "#d800cc"], // 2: pink / purple
  ["#58d854", "#6844fc"], // 3: green / indigo
  ["#58f898", "#e40058"], // 4: mint / red-pink
  ["#6888fc", "#58f898"], // 5: periwinkle / mint
  ["#7c7c7c", "#f83800"], // 6: gray / red-orange
  ["#a81000", "#6844fc"], // 7: dark red / indigo
  ["#f83800", "#0058f8"], // 8: red-orange / blue
  ["#fca044", "#f83800"], // 9: orange / red-orange
];

/**
 * The three block tile graphics in the NES use white, color1 and color2. Each
 * tetromino is always drawn with one fixed tile graphic. This is the canonical
 * per-piece assignment, taken directly from the NES orientation table tile ids
 * (T,O,I -> 7B white; J,S -> 7D color1; Z,L -> 7C color2):
 *   0 = white, 1 = color1, 2 = color2
 * Piece order is T,J,Z,O,S,L,I.
 */
export const PIECE_TILE = [0, 1, 2, 0, 1, 2, 0];

/** Return { white, c1, c2 } for a level (cycles every 10). */
export function levelColors(level) {
  const p = LEVEL_PALETTES[((level % 10) + 10) % 10];
  return { white: NES_WHITE, c1: p[0], c2: p[1] };
}

/** The fill colour for a given piece id at a given level. */
export function tileColor(pieceId, level) {
  const { white, c1, c2 } = levelColors(level);
  const t = PIECE_TILE[pieceId];
  return t === 0 ? white : t === 1 ? c1 : c2;
}

/** Rotation states per piece, clockwise from spawn (index 0). [{x,y}*4] each. */
export const PIECE_STATES = [
  // T (4)
  [
    [{ x: -1, y: 0 }, { x: 0, y: 0 }, { x: 1, y: 0 }, { x: 0, y: 1 }],
    [{ x: 0, y: -1 }, { x: -1, y: 0 }, { x: 0, y: 0 }, { x: 0, y: 1 }],
    [{ x: 0, y: -1 }, { x: -1, y: 0 }, { x: 0, y: 0 }, { x: 1, y: 0 }],
    [{ x: 0, y: -1 }, { x: 0, y: 0 }, { x: 1, y: 0 }, { x: 0, y: 1 }],
  ],
  // J (4)
  [
    [{ x: -1, y: 0 }, { x: 0, y: 0 }, { x: 1, y: 0 }, { x: 1, y: 1 }],
    [{ x: 0, y: -1 }, { x: 0, y: 0 }, { x: -1, y: 1 }, { x: 0, y: 1 }],
    [{ x: -1, y: -1 }, { x: -1, y: 0 }, { x: 0, y: 0 }, { x: 1, y: 0 }],
    [{ x: 0, y: -1 }, { x: 1, y: -1 }, { x: 0, y: 0 }, { x: 0, y: 1 }],
  ],
  // Z (2)
  [
    [{ x: -1, y: 0 }, { x: 0, y: 0 }, { x: 0, y: 1 }, { x: 1, y: 1 }],
    [{ x: 1, y: -1 }, { x: 0, y: 0 }, { x: 1, y: 0 }, { x: 0, y: 1 }],
  ],
  // O (1)
  [
    [{ x: -1, y: 0 }, { x: 0, y: 0 }, { x: -1, y: 1 }, { x: 0, y: 1 }],
  ],
  // S (2)
  [
    [{ x: 0, y: 0 }, { x: 1, y: 0 }, { x: -1, y: 1 }, { x: 0, y: 1 }],
    [{ x: 0, y: -1 }, { x: 0, y: 0 }, { x: 1, y: 0 }, { x: 1, y: 1 }],
  ],
  // L (4)
  [
    [{ x: -1, y: 0 }, { x: 0, y: 0 }, { x: 1, y: 0 }, { x: -1, y: 1 }],
    [{ x: -1, y: -1 }, { x: 0, y: -1 }, { x: 0, y: 0 }, { x: 0, y: 1 }],
    [{ x: 1, y: -1 }, { x: -1, y: 0 }, { x: 0, y: 0 }, { x: 1, y: 0 }],
    [{ x: 0, y: -1 }, { x: 0, y: 0 }, { x: 0, y: 1 }, { x: 1, y: 1 }],
  ],
  // I (2)
  [
    [{ x: -2, y: 0 }, { x: -1, y: 0 }, { x: 0, y: 0 }, { x: 1, y: 0 }],
    [{ x: 0, y: -2 }, { x: 0, y: -1 }, { x: 0, y: 0 }, { x: 0, y: 1 }],
  ],
];

export const SPAWN_X = 5;
export const SPAWN_Y = 0;

export function stateCount(piece) {
  return PIECE_STATES[piece].length;
}

export function cellsOf(piece, rotation) {
  const states = PIECE_STATES[piece];
  const idx = ((rotation % states.length) + states.length) % states.length;
  return states[idx];
}

export function rotateRight(piece, rotation) {
  const n = stateCount(piece);
  return (rotation + 1) % n;
}

export function rotateLeft(piece, rotation) {
  const n = stateCount(piece);
  return (rotation - 1 + n) % n;
}

// ---------------------------------------------------------------------------
// rng — 16-bit Fibonacci LFSR + NES two-stage selection with repeat protection
// ---------------------------------------------------------------------------

export const DEFAULT_SEED = 0x8988;
const SPAWN_TABLE = [0, 1, 2, 3, 4, 5, 6];

export function stepLfsr(value) {
  const bit = ((value >> 1) ^ (value >> 9)) & 1;
  return ((value >> 1) | (bit << 15)) & 0xffff;
}

function highByte(value) {
  return (value >> 8) & 0xff;
}

export function createRng(seed = DEFAULT_SEED) {
  return { value: seed & 0xffff, spawnCount: 0, prevPiece: 6 };
}

export function spawnPiece(rng) {
  const spawnCount = (rng.spawnCount + 1) & 0xff;
  let value = rng.value;

  let index = (highByte(value) + spawnCount) & 7;
  const prevIndex = rng.prevPiece;
  const firstPick = index < PIECE_COUNT ? SPAWN_TABLE[index] : 7;

  if (index === 7 || firstPick === rng.prevPiece) {
    value = stepLfsr(value);
    index = (highByte(value) + prevIndex) % PIECE_COUNT;
  }

  const piece = SPAWN_TABLE[index];
  return { rng: { value, spawnCount, prevPiece: piece }, piece };
}

export function advanceFrameRng(rng) {
  return { value: stepLfsr(rng.value), spawnCount: rng.spawnCount, prevPiece: rng.prevPiece };
}

// ---------------------------------------------------------------------------
// gravity & level progression (NTSC)
// ---------------------------------------------------------------------------

export const FRAMES_PER_GRIDCELL_BY_LEVEL = [
  48, 43, 38, 33, 28, 23, 18, 13, 8, 6,
  5, 5, 5, 4, 4, 4, 3, 3, 3,
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  1,
];

export function framesPerGridcell(level) {
  if (level < 0) return FRAMES_PER_GRIDCELL_BY_LEVEL[0];
  if (level >= FRAMES_PER_GRIDCELL_BY_LEVEL.length) {
    return FRAMES_PER_GRIDCELL_BY_LEVEL[FRAMES_PER_GRIDCELL_BY_LEVEL.length - 1];
  }
  return FRAMES_PER_GRIDCELL_BY_LEVEL[level];
}

export const SOFT_DROP_FRAMES_PER_GRIDCELL = 2;

export function firstTransitionLines(startLevel) {
  return Math.min(startLevel * 10 + 10, Math.max(100, startLevel * 10 - 50));
}

export function levelFor(startLevel, totalLines) {
  const first = firstTransitionLines(startLevel);
  if (totalLines < first) return startLevel;
  return startLevel + 1 + Math.floor((totalLines - first) / 10);
}

// ---------------------------------------------------------------------------
// DAS
// ---------------------------------------------------------------------------

export const DAS_INITIAL = 16;
export const DAS_REPEAT = 6;

export function createDas() {
  return { charge: 0, dir: 0 };
}
export function fullCharge() {
  return DAS_INITIAL;
}
export function postShiftCharge() {
  return DAS_INITIAL - DAS_REPEAT;
}

// ---------------------------------------------------------------------------
// scoring
// ---------------------------------------------------------------------------

export const LINE_CLEAR_BASE = { 0: 0, 1: 40, 2: 100, 3: 300, 4: 1200 };

export function lineClearScore(lines, level) {
  const base = LINE_CLEAR_BASE[lines] ?? 0;
  return base * (level + 1);
}

// ---------------------------------------------------------------------------
// input bitfield (no hard drop, no hold — only a real controller's inputs)
// ---------------------------------------------------------------------------

export const INPUT_LEFT = 1 << 0;
export const INPUT_RIGHT = 1 << 1;
export const INPUT_DOWN = 1 << 2; // soft drop
export const INPUT_ROTATE_CW = 1 << 3; // A
export const INPUT_ROTATE_CCW = 1 << 4; // B
export const INPUT_MASK = 0x1f;

export function has(bits, mask) {
  return (bits & mask) !== 0;
}
export function pressed(bits, prev, mask) {
  return (bits & mask) !== 0 && (prev & mask) === 0;
}

// ---------------------------------------------------------------------------
// board
// ---------------------------------------------------------------------------

export const WIDTH = 10;
export const HEIGHT = 20;
export const CELL_COUNT = WIDTH * HEIGHT;

export function createBoard() {
  return new Uint8Array(CELL_COUNT);
}
export function indexOf(x, y) {
  return y * WIDTH + x;
}
export function getCell(board, x, y) {
  if (x < 0 || x >= WIDTH || y < 0 || y >= HEIGHT) return 0;
  return board[indexOf(x, y)];
}

export function collides(board, piece, rotation, originX, originY) {
  const cells = cellsOf(piece, rotation);
  for (let i = 0; i < 4; i++) {
    const c = cells[i];
    const x = originX + c.x;
    const y = originY + c.y;
    if (x < 0 || x >= WIDTH) return true;
    if (y >= HEIGHT) return true;
    if (y >= 0 && board[indexOf(x, y)] !== 0) return true;
  }
  return false;
}

export function lockPiece(board, piece, rotation, originX, originY) {
  const cells = cellsOf(piece, rotation);
  const value = piece + 1;
  for (let i = 0; i < 4; i++) {
    const c = cells[i];
    const x = originX + c.x;
    const y = originY + c.y;
    if (y >= 0 && y < HEIGHT && x >= 0 && x < WIDTH) {
      board[indexOf(x, y)] = value;
    }
  }
}

export function fullRows(board) {
  const rows = [];
  for (let y = 0; y < HEIGHT; y++) {
    let full = true;
    for (let x = 0; x < WIDTH; x++) {
      if (board[indexOf(x, y)] === 0) {
        full = false;
        break;
      }
    }
    if (full) rows.push(y);
  }
  return rows;
}

export function clearRows(board, rows) {
  if (rows.length === 0) return 0;
  const removed = new Set(rows);
  const out = new Uint8Array(CELL_COUNT);
  let writeY = HEIGHT - 1;
  for (let y = HEIGHT - 1; y >= 0; y--) {
    if (removed.has(y)) continue;
    for (let x = 0; x < WIDTH; x++) {
      out[indexOf(x, writeY)] = board[indexOf(x, y)];
    }
    writeY--;
  }
  board.set(out);
  return rows.length;
}

// ---------------------------------------------------------------------------
// rotation (NRS, no kicks)
// ---------------------------------------------------------------------------

export function tryRotateRight(board, piece, rotation, x, y) {
  const next = rotateRight(piece, rotation);
  if (next === rotation) return { rotation, moved: false };
  if (collides(board, piece, next, x, y)) return { rotation, moved: false };
  return { rotation: next, moved: true };
}

export function tryRotateLeft(board, piece, rotation, x, y) {
  const next = rotateLeft(piece, rotation);
  if (next === rotation) return { rotation, moved: false };
  if (collides(board, piece, next, x, y)) return { rotation, moved: false };
  return { rotation: next, moved: true };
}

// ---------------------------------------------------------------------------
// reducer — the deterministic step function (single source of game truth)
// ---------------------------------------------------------------------------

const LINE_CLEAR_STEPS = 5;

export function createInitialState(seed, startLevel) {
  const board = createBoard();
  let rng = createRng(seed);

  const first = spawnPiece(rng);
  rng = first.rng;
  const second = spawnPiece(rng);
  rng = second.rng;

  const activePiece = { id: first.piece, rotation: 0, x: SPAWN_X, y: SPAWN_Y };

  return {
    frame: 0,
    level: startLevel,
    startLevel,
    lines: 0,
    score: 0,
    board,
    activePiece,
    nextPiece: second.piece,
    rng,
    das: createDas(),
    phase: "active",
    prevInput: 0,
    gravityCounter: 0,
    softDropCells: 0,
    softDropping: false,
    areTimer: 0,
    lineClearTimer: 0,
    clearingRows: [],
    pendingClearCount: 0,
  };
}

export function areForLockRow(lowestY) {
  const rowsFromBottom = HEIGHT - 1 - lowestY;
  const group = Math.floor(rowsFromBottom / 4);
  return Math.min(10 + group * 2, 18);
}

function clone(s) {
  return {
    ...s,
    board: s.board.slice(),
    activePiece: { ...s.activePiece },
    rng: { ...s.rng },
    das: { ...s.das },
    clearingRows: s.clearingRows.slice(),
  };
}

export function step(state, input) {
  if (state.phase === "game_over") {
    const s = clone(state);
    s.frame += 1;
    s.prevInput = input;
    return s;
  }

  const s = clone(state);
  s.rng = advanceFrameRng(s.rng);

  switch (s.phase) {
    case "active":
      stepActive(s, input);
      break;
    case "line_clear":
      stepLineClear(s);
      break;
    case "are":
      stepAre(s);
      break;
  }

  s.frame += 1;
  s.prevInput = input;
  return s;
}

function stepActive(s, input) {
  const prev = s.prevInput;
  const p = s.activePiece;

  if (pressed(input, prev, INPUT_ROTATE_CW)) {
    const r = tryRotateRight(s.board, p.id, p.rotation, p.x, p.y);
    p.rotation = r.rotation;
  } else if (pressed(input, prev, INPUT_ROTATE_CCW)) {
    const r = tryRotateLeft(s.board, p.id, p.rotation, p.x, p.y);
    p.rotation = r.rotation;
  }

  applyDas(s, input, prev);
  applyGravity(s, input);
}

function applyDas(s, input, prev) {
  const left = has(input, INPUT_LEFT);
  const right = has(input, INPUT_RIGHT);

  let dir = 0;
  if (left && !right) dir = -1;
  else if (right && !left) dir = 1;

  const das = s.das;

  if (dir === 0) {
    das.charge = 0;
    das.dir = 0;
    return;
  }

  const freshPress =
    das.dir !== dir ||
    (dir === -1 && pressed(input, prev, INPUT_LEFT)) ||
    (dir === 1 && pressed(input, prev, INPUT_RIGHT));

  if (freshPress) {
    das.dir = dir;
    das.charge = 0;
    tryShift(s, dir);
    return;
  }

  das.charge += 1;
  if (das.charge >= DAS_INITIAL) {
    const moved = tryShift(s, dir);
    das.charge = moved ? postShiftCharge() : fullCharge();
  }
}

function tryShift(s, dx) {
  const p = s.activePiece;
  if (!collides(s.board, p.id, p.rotation, p.x + dx, p.y)) {
    p.x += dx;
    return true;
  }
  return false;
}

function applyGravity(s, input) {
  const softDrop = has(input, INPUT_DOWN);

  if (softDrop) {
    if (!s.softDropping) {
      s.softDropping = true;
      s.softDropCells = 0;
    }
  } else {
    s.softDropping = false;
    s.softDropCells = 0;
  }

  const gravityInterval = framesPerGridcell(s.level);
  const interval =
    softDrop && SOFT_DROP_FRAMES_PER_GRIDCELL < gravityInterval
      ? SOFT_DROP_FRAMES_PER_GRIDCELL
      : gravityInterval;

  s.gravityCounter += 1;
  if (s.gravityCounter < interval) return;
  s.gravityCounter = 0;

  const p = s.activePiece;
  if (!collides(s.board, p.id, p.rotation, p.x, p.y + 1)) {
    p.y += 1;
    if (softDrop) s.softDropCells += 1;
  } else {
    lockAndAdvance(s);
  }
}

function lockAndAdvance(s) {
  const p = s.activePiece;

  if (s.softDropping && s.softDropCells > 0) {
    s.score += s.softDropCells;
  }
  s.softDropping = false;
  s.softDropCells = 0;

  lockPiece(s.board, p.id, p.rotation, p.x, p.y);

  const lowestY = lowestLockedRow(p);
  const rows = fullRows(s.board);
  if (rows.length > 0) {
    s.phase = "line_clear";
    s.clearingRows = rows;
    s.pendingClearCount = rows.length;
    s.lineClearTimer = LINE_CLEAR_STEPS;
    s.areTimer = areForLockRow(lowestY);
  } else {
    s.phase = "are";
    s.areTimer = areForLockRow(lowestY);
    s.pendingClearCount = 0;
  }
}

function lowestLockedRow(p) {
  let lowest = -Infinity;
  const cells = cellsOf(p.id, p.rotation);
  for (const c of cells) {
    const y = p.y + c.y;
    if (y > lowest) lowest = y;
  }
  return Math.min(lowest, HEIGHT - 1);
}

function stepLineClear(s) {
  if (s.frame % 4 === 0) {
    s.lineClearTimer -= 1;
  }
  if (s.lineClearTimer > 0) return;

  const cleared = clearRows(s.board, s.clearingRows);
  s.lines += cleared;
  s.level = levelFor(s.startLevel, s.lines);
  s.score += lineClearScore(cleared, s.level);
  s.clearingRows = [];
  s.pendingClearCount = 0;

  s.phase = "are";
}

function stepAre(s) {
  s.areTimer -= 1;
  if (s.areTimer > 0) return;
  spawnNext(s);
}

function spawnNext(s) {
  const id = s.nextPiece;
  const spawn = { id, rotation: 0, x: SPAWN_X, y: SPAWN_Y };

  if (collides(s.board, spawn.id, spawn.rotation, spawn.x, spawn.y)) {
    s.phase = "game_over";
    s.activePiece = spawn;
    return;
  }

  const next = spawnPiece(s.rng);
  s.rng = next.rng;

  s.activePiece = spawn;
  s.nextPiece = next.piece;
  s.phase = "active";
  s.gravityCounter = 0;
  s.das = createDas();
}

// ---------------------------------------------------------------------------
// replay — the anti-cheat core. Re-run a game from { seed, startLevel, inputs }
// and return the derived final state. Both client and server call this.
// ---------------------------------------------------------------------------

export const NTSC_FPS = 60.0988;

/** Maximum frames a replay may contain (~30 minutes at NTSC). */
export const MAX_REPLAY_FRAMES = 110_000;

/**
 * Re-simulate a full game. `inputs` is a sequence of per-frame InputBits (one
 * u8 per frame). Returns the final EngineState. Throws on malformed input.
 */
export function runReplay({ seed, startLevel, inputs }) {
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffff) {
    throw new Error("bad seed");
  }
  if (!Number.isInteger(startLevel) || startLevel < 0 || startLevel > 29) {
    throw new Error("bad startLevel");
  }
  if (!(inputs instanceof Uint8Array) && !Array.isArray(inputs)) {
    throw new Error("bad inputs");
  }
  if (inputs.length > MAX_REPLAY_FRAMES) {
    throw new Error("replay too long");
  }

  let state = createInitialState(seed, startLevel);
  for (let i = 0; i < inputs.length; i++) {
    const bits = inputs[i] & INPUT_MASK;
    state = step(state, bits);
    if (state.phase === "game_over") break;
  }
  return state;
}
