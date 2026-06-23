import assert from "node:assert/strict";
import test from "node:test";
import {
  createInitialState,
  step,
  runReplay,
  INPUT_DOWN,
  INPUT_ROTATE_CW,
  framesPerGridcell,
  levelFor,
  lineClearScore,
  spawnPiece,
  createRng,
} from "../public/nes-engine.js";

test("NTSC gravity table matches known values", () => {
  assert.equal(framesPerGridcell(0), 48);
  assert.equal(framesPerGridcell(9), 6);
  assert.equal(framesPerGridcell(18), 3);
  assert.equal(framesPerGridcell(29), 1);
  assert.equal(framesPerGridcell(100), 1); // clamps
});

test("NES line-clear scoring uses post-clear level multiplier", () => {
  assert.equal(lineClearScore(1, 0), 40);
  assert.equal(lineClearScore(4, 0), 1200);
  assert.equal(lineClearScore(4, 9), 12000);
});

test("level progression: low start transitions at (start+1)*10 lines", () => {
  assert.equal(levelFor(0, 9), 0);
  assert.equal(levelFor(0, 10), 1);
  assert.equal(levelFor(5, 59), 5);
  assert.equal(levelFor(5, 60), 6);
});

test("randomizer is deterministic for a fixed seed", () => {
  const a = createRng(0x8988);
  const b = createRng(0x8988);
  const seqA = [];
  const seqB = [];
  let ra = a, rb = b;
  for (let i = 0; i < 20; i++) {
    const pa = spawnPiece(ra); ra = pa.rng; seqA.push(pa.piece);
    const pb = spawnPiece(rb); rb = pb.rng; seqB.push(pb.piece);
  }
  assert.deepEqual(seqA, seqB);
});

test("step is pure: same inputs from same state give same result", () => {
  const s0 = createInitialState(0x1234, 0);
  const a = step(s0, INPUT_DOWN);
  const b = step(s0, INPUT_DOWN);
  assert.equal(a.frame, b.frame);
  assert.equal(a.score, b.score);
  assert.deepEqual(Array.from(a.board), Array.from(b.board));
  // s0 not mutated
  assert.equal(s0.frame, 0);
});

test("runReplay reproduces the same final state every time", () => {
  const inputs = [];
  for (let i = 0; i < 2000; i++) {
    inputs.push(i % 30 === 0 ? INPUT_ROTATE_CW : INPUT_DOWN);
  }
  const r1 = runReplay({ seed: 0x4321, startLevel: 0, inputs });
  const r2 = runReplay({ seed: 0x4321, startLevel: 0, inputs });
  assert.equal(r1.score, r2.score);
  assert.equal(r1.lines, r2.lines);
  assert.equal(r1.frame, r2.frame);
  assert.equal(r1.phase, "game_over"); // soft-dropping in place tops out
});

test("runReplay rejects malformed parameters", () => {
  assert.throws(() => runReplay({ seed: -1, startLevel: 0, inputs: [0] }));
  assert.throws(() => runReplay({ seed: 0, startLevel: 99, inputs: [0] }));
  assert.throws(() => runReplay({ seed: 0, startLevel: 0, inputs: new Array(200000).fill(0) }));
});
