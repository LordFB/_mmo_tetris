/**
 * rl.js — a small, coherent reinforcement-learning layer on top of nn.js.
 *
 * It defines a generic contract so the SAME training loop and agent serve the
 * 4x4 pieces-fill puzzle today and a Tetris-playing agent later:
 *
 *   Env {
 *     reset(rng) -> void
 *     observe()  -> Float64Array      // fixed-length feature vector (net input)
 *     actionMask()-> Uint8Array       // 1 = legal action, 0 = illegal
 *     step(action)-> { reward, done }  // advance one action
 *     get actionSize / obsSize         // fixed dimensions
 *     // optional: render hooks for the UI (the puzzle exposes its grid/piece)
 *   }
 *
 *   Agent (policy net): observe -> masked softmax over actions -> sample/argmax.
 *
 * The trainer (scripts/train.mjs) runs REINFORCE with a moving-average baseline.
 * Everything is seedable via an injected rng for reproducible runs.
 */

import { Net, softmax, sample, argmax } from "./nn.js";

/* -------------------------------------------------------------------------- */
/* PuzzleEnv — fill a 4x4 grid with a bag of small pieces, no overlaps.        */
/*                                                                            */
/* The agent is fed: the 16 cell occupancies, plus a one-hot of the current   */
/* piece. Each turn it chooses an anchor cell (0..15) to drop the piece's      */
/* top-left at; illegal anchors (off-board / overlapping) are masked out.      */
/* Placing the last piece to fully tile the board = success; reaching a state  */
/* with no legal move before the board is full = failure.                     */
/*                                                                            */
/* The piece set is chosen so the 4x4 is always tileable in principle, making  */
/* "did the agent actually solve it" a meaningful signal.                      */
/* -------------------------------------------------------------------------- */

// Board is Tetris-width (10) and 4 tall — a wide well that fills the STATUS
// panel and, not coincidentally, matches Tetris's column count so the same env
// shape carries over to a real Tetris agent later. GRID stays exported as the
// WIDTH (most callers index columns); GRID_H is the height.
export const GRID = 10; // width (columns)
export const GRID_H = 4; // height (rows)
export const CELLS = GRID * GRID_H; // 40

// Pieces as lists of {x,y} offsets from the anchor (top-left of bounding box).
export const PIECES = [
  { name: "O", cells: [[0, 0], [1, 0], [0, 1], [1, 1]] }, // 2x2 square
  { name: "I", cells: [[0, 0], [0, 1], [0, 2], [0, 3]] }, // vertical 1x4 (fills a column)
];

// A reference tileable cover of 10x4 (6 vertical lines + 4 squares = 40 cells),
// used as the default when no rng is supplied. Per-episode the env draws a
// RANDOM tileable bag (see makeBag) so the demo and training see varied puzzles.
export const BAG = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0];

/**
 * Build a random, guaranteed-tileable bag for a 10x4 board from {square, line}.
 * A square-pair fills one 2-column block (2 cols x 4 rows = 8 cells); a vertical
 * line fills one column. So choosing `pairs` square-pairs and filling the
 * remaining `GRID - 2*pairs` columns with lines always tiles. The piece list is
 * then shuffled so the agent can't rely on a fixed order.
 */
export function makeBag(rng = Math.random) {
  const maxPairs = Math.floor(GRID / 2); // 5
  const pairs = Math.floor(rng() * (maxPairs + 1)); // 0..5 square-pairs
  const lines = GRID - 2 * pairs; // remaining columns get vertical lines
  const bag = [];
  for (let i = 0; i < pairs * 2; i++) bag.push(0); // squares (2 per pair)
  for (let i = 0; i < lines; i++) bag.push(1); // vertical lines
  // Fisher–Yates shuffle with the injected rng
  for (let i = bag.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [bag[i], bag[j]] = [bag[j], bag[i]];
  }
  return bag;
}

export class PuzzleEnv {
  constructor() {
    this.obsSize = CELLS + PIECES.length; // grid + current-piece one-hot
    this.actionSize = CELLS; // anchor cell 0..CELLS-1
    this.reset(Math.random);
  }

  reset(rng = Math.random) {
    this.grid = new Uint8Array(CELLS);
    this.bag = makeBag(rng); // a fresh random tileable bag each episode
    this.bagIndex = 0; // which piece in this.bag we're placing
    this.placed = 0;
    this.done = false;
    this.success = false;
    return this;
  }

  get currentPiece() {
    return this.bagIndex < this.bag.length ? PIECES[this.bag[this.bagIndex]] : null;
  }

  // Feature vector: occupancy then a one-hot of the current piece type.
  observe() {
    const v = new Float64Array(this.obsSize);
    for (let i = 0; i < CELLS; i++) v[i] = this.grid[i];
    const p = this.currentPiece;
    if (p) v[CELLS + PIECES.indexOf(p)] = 1;
    return v;
  }

  // Can `piece` sit with its anchor at cell index `anchor` without going
  // off-board or overlapping?
  fits(piece, anchor) {
    const ax = anchor % GRID;
    const ay = (anchor / GRID) | 0;
    for (const [dx, dy] of piece.cells) {
      const x = ax + dx;
      const y = ay + dy;
      if (x < 0 || x >= GRID || y < 0 || y >= GRID_H) return false;
      if (this.grid[y * GRID + x]) return false;
    }
    return true;
  }

  actionMask() {
    const mask = new Uint8Array(this.actionSize);
    const p = this.currentPiece;
    if (!p) return mask;
    for (let a = 0; a < this.actionSize; a++) mask[a] = this.fits(p, a) ? 1 : 0;
    return mask;
  }

  hasAnyMove() {
    const m = this.actionMask();
    for (const v of m) if (v) return true;
    return false;
  }

  // Place the current piece at anchor (assumed legal). Returns {reward, done}.
  step(anchor) {
    const p = this.currentPiece;
    if (!p || this.done) return { reward: 0, done: true };
    if (!this.fits(p, anchor)) {
      // An illegal action ends the episode in failure (shouldn't happen when
      // the agent respects the mask, but keeps step() total).
      this.done = true;
      return { reward: -1, done: true };
    }
    const ax = anchor % GRID;
    const ay = (anchor / GRID) | 0;
    const filled = p.cells.length;
    for (const [dx, dy] of p.cells) this.grid[(ay + dy) * GRID + (ax + dx)] = 1 + this.bagIndex;
    this.placed++;
    this.bagIndex++;

    // Potential-based shaping: reward the AREA just covered (so placing more
    // total cells before stopping yields a strictly higher return — this gives
    // REINFORCE a smooth gradient instead of a bimodal one, which otherwise
    // collapses to a deterministic 2-piece dead-end). A full valid tiling earns
    // a strong terminal bonus; a dead-end with pieces left is penalized.
    let reward = filled / CELLS; // in (0,1], proportional to cells placed
    if (this.bagIndex >= this.bag.length) {
      this.done = true;
      this.success = this.grid.every((c) => c !== 0);
      if (this.success) reward += 1; // strong terminal reward for a full solve
    } else if (!this.hasAnyMove()) {
      // dead-end: pieces remain but nothing fits -> failure
      this.done = true;
      reward -= 0.5;
    }
    return { reward, done: this.done };
  }
}

/* -------------------------------------------------------------------------- */
/* Agent — a policy net wrapper. Masks illegal actions BEFORE softmax so the    */
/* distribution only ever puts probability on legal moves, then samples (train) */
/* or takes argmax (greedy / demo).                                            */
/* -------------------------------------------------------------------------- */
export class Agent {
  constructor(net) {
    this.net = net;
  }

  static fresh(obsSize, actionSize, hidden = [32, 32], seed = 1) {
    const sizes = [obsSize, ...hidden, actionSize];
    const net = new Net(sizes, { hidden: "relu", output: "linear", softmax: false, seed });
    return new Agent(net);
  }

  // Returns { probs, logits } with illegal actions zeroed in probs.
  policy(obs, mask) {
    const logits = this.net.forward(obs); // raw (softmax:false on the net)
    const masked = logits.slice();
    for (let i = 0; i < masked.length; i++) {
      if (!mask[i]) masked[i] = -1e9; // push illegal logits to ~0 prob
    }
    return { probs: softmax(masked), logits };
  }

  act(obs, mask, rng, greedy = false) {
    const { probs } = this.policy(obs, mask);
    return greedy ? argmax(probs) : sample(probs, rng);
  }

  toJSON() {
    return { format: "agent.v1", net: this.net.toJSON() };
  }

  static fromJSON(o) {
    return new Agent(Net.fromJSON(o.net));
  }
}

/* -------------------------------------------------------------------------- */
/* Episode rollout — generic over any Env/Agent. Records the per-step          */
/* (obs, mask, action, reward) needed for REINFORCE. Greedy mode is used for   */
/* evaluation and the UI demo.                                                 */
/* -------------------------------------------------------------------------- */
export function rollout(env, agent, rng, { greedy = false, record = true } = {}) {
  env.reset(rng);
  const steps = [];
  let total = 0;
  while (!env.done) {
    const obs = env.observe();
    const mask = env.actionMask();
    const { probs } = agent.policy(obs, mask);
    const action = greedy ? argmax(probs) : sample(probs, rng);
    const { reward, done } = env.step(action);
    total += reward;
    if (record) steps.push({ obs, mask, action, reward });
    if (done) break;
  }
  return { steps, total, success: env.success };
}
