import { test } from "node:test";
import assert from "node:assert/strict";

import {
  Net,
  Dense,
  softmax,
  argmax,
  sample,
  mulberry32,
  SGD,
  Adam,
} from "../public/nn.js";
import { PuzzleEnv, Agent, rollout, BAG, makeBag, CELLS, GRID, GRID_H } from "../public/rl.js";

const anchor = (x, y) => y * GRID + x;

// Force a deterministic bag for tests that hand-play a specific tiling.
function withBag(env, bag) {
  env.reset();
  env.bag = bag.slice();
  env.bagIndex = 0;
  return env;
}

/* --------------------------------------------------------------------------
 * Gradient check — the load-bearing test. We verify the analytic gradient from
 * backwardPolicy() matches a central finite-difference estimate of the
 * cross-entropy loss w.r.t. every weight and bias. If backprop is wrong, every
 * downstream thing (training, the demo) is silently broken; this catches it.
 * ------------------------------------------------------------------------ */
test("backprop gradient matches finite differences (softmax CE)", () => {
  const rng = mulberry32(7);
  const net = new Net([5, 6, 4], { hidden: "tanh", output: "linear", seed: 7 });
  const input = Float64Array.from([0.3, -0.7, 0.1, 0.9, -0.2]);
  const target = 2;

  // analytic gradient
  const probs = softmax(net.forward(input));
  net.zeroGrad();
  net.backwardPolicy(probs, target, 1); // weight 1 = plain cross-entropy

  const loss = (n) => {
    const p = softmax(n.forward(input));
    return -Math.log(p[target] + 1e-12);
  };

  const eps = 1e-5;
  for (const layer of net.layers) {
    for (const [params, grads] of [
      [layer.W, layer.gW],
      [layer.b, layer.gb],
    ]) {
      for (let i = 0; i < params.length; i++) {
        const orig = params[i];
        params[i] = orig + eps;
        const lp = loss(net);
        params[i] = orig - eps;
        const lm = loss(net);
        params[i] = orig;
        const numeric = (lp - lm) / (2 * eps);
        const analytic = grads[i];
        const denom = Math.max(1, Math.abs(numeric) + Math.abs(analytic));
        assert.ok(
          Math.abs(numeric - analytic) / denom < 1e-4,
          `grad mismatch: numeric=${numeric} analytic=${analytic}`,
        );
      }
    }
  }
});

test("a single SGD step reduces cross-entropy loss on a fixed example", () => {
  const net = new Net([4, 8, 3], { hidden: "relu", seed: 3 });
  const opt = new SGD(net, { lr: 0.1 });
  const x = Float64Array.from([0.5, -0.5, 0.25, -0.25]);
  const target = 1;
  const ce = () => -Math.log(softmax(net.forward(x))[target] + 1e-12);

  const before = ce();
  for (let k = 0; k < 20; k++) {
    const p = softmax(net.forward(x));
    net.zeroGrad();
    net.backwardPolicy(p, target, 1);
    opt.step();
  }
  assert.ok(ce() < before, "loss should decrease after gradient steps");
});

test("Adam optimizer also drives the loss down", () => {
  const net = new Net([4, 8, 3], { hidden: "relu", seed: 9 });
  const opt = new Adam(net, { lr: 0.05 });
  const x = Float64Array.from([0.1, 0.9, -0.3, 0.4]);
  const target = 2;
  const ce = () => -Math.log(softmax(net.forward(x))[target] + 1e-12);
  const before = ce();
  for (let k = 0; k < 30; k++) {
    const p = softmax(net.forward(x));
    net.zeroGrad();
    net.backwardPolicy(p, target, 1);
    opt.step();
  }
  assert.ok(ce() < before);
});

/* --------------------------------------------------------------------------
 * Serialization — the trained net must round-trip exactly so the browser's
 * inference equals the trainer's. We check identical outputs after toJSON ->
 * JSON string -> fromJSON.
 * ------------------------------------------------------------------------ */
test("Net serialize round-trips to identical outputs", () => {
  const net = new Net([6, 10, 4], { hidden: "relu", seed: 11 });
  const x = Float64Array.from([0.2, -0.1, 0.7, 0.3, -0.4, 0.9]);
  const before = Array.from(net.forward(x));

  const restored = Net.fromJSON(JSON.parse(JSON.stringify(net.toJSON())));
  const after = Array.from(restored.forward(x));

  assert.deepEqual(after, before);
});

test("Dense serialize round-trips weights and shape", () => {
  const d = new Dense(3, 2, "tanh").init(mulberry32(1));
  const r = Dense.fromJSON(JSON.parse(JSON.stringify(d.toJSON())));
  assert.equal(r.inSize, 3);
  assert.equal(r.outSize, 2);
  assert.equal(r.activation, "tanh");
  assert.deepEqual(Array.from(r.W), Array.from(d.W));
  assert.deepEqual(Array.from(r.b), Array.from(d.b));
});

test("softmax is a normalized distribution; argmax/sample agree on a spike", () => {
  const p = softmax(Float64Array.from([1, 5, 2, 0]));
  const sum = p.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sum - 1) < 1e-12);
  assert.equal(argmax(p), 1);
  // a near-deterministic distribution should sample its mode
  const spike = softmax(Float64Array.from([0, 50, 0]));
  assert.equal(sample(spike, () => 0.99), 1);
});

/* --------------------------------------------------------------------------
 * PuzzleEnv — deterministic mechanics, legal-move masking, and a hand-played
 * perfect solve actually registering success. This anchors the RL signal.
 * ------------------------------------------------------------------------ */
test("PuzzleEnv masks illegal placements and bounds the action space", () => {
  const env = new PuzzleEnv();
  assert.equal(env.actionSize, CELLS);
  assert.equal(CELLS, GRID * GRID_H);
  // force a line-first bag so the assertions are deterministic. A vertical 1x4
  // (height 4 = full column) only fits anchored on the TOP row (y=0).
  withBag(env, BAG);
  const mask = env.actionMask();
  assert.equal(mask[anchor(0, 0)], 1, "vertical line fits anchored at top of col 0");
  assert.equal(mask[anchor(0, 1)], 0, "vertical line cannot start at row 1 (would overflow)");
});

test("makeBag always produces a tileable 40-cell bag (area + parity)", () => {
  const rng = mulberry32(123);
  for (let i = 0; i < 200; i++) {
    const bag = makeBag(rng);
    // area must be exactly CELLS (squares=4, lines=4 cells each)
    const area = bag.reduce((s, p) => s + 4, 0);
    assert.equal(area, CELLS);
    // squares come in even counts (full 2-col blocks) for a clean tiling
    const squares = bag.filter((p) => p === 0).length;
    assert.equal(squares % 2, 0, "squares must pair up");
  }
});

test("PuzzleEnv: a constructed tiling solves the 10x4 (success + reward)", () => {
  // force the reference bag: 6 vertical lines + 4 squares.
  const env = withBag(new PuzzleEnv(), [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]);

  let totalReward = 0;
  const place = (a) => { const r = env.step(a); totalReward += r.reward; return r; };

  for (let col = 0; col < 6; col++) place(anchor(col, 0)); // 6 vertical lines
  place(anchor(6, 0)); // square cols 6-7 rows 0-1
  place(anchor(6, 2)); // square cols 6-7 rows 2-3
  place(anchor(8, 0)); // square cols 8-9 rows 0-1
  const last = place(anchor(8, 2)); // square cols 8-9 rows 2-3

  assert.ok(last.done && env.success, "board should be fully solved");
  assert.ok(env.grid.every((c) => c !== 0), "every cell filled");
  // area covered sums to CELLS (-> shaped reward 1) plus the +1 solve bonus
  assert.ok(Math.abs(totalReward - 2) < 1e-9, `unexpected reward ${totalReward}`);
});

test("PuzzleEnv: a blocking sequence dead-ends in failure", () => {
  // 6 vertical lines fill cols 0-5; then a square at cols 6-7 (r0-1) and a
  // square STRADDLING cols 7-8 (r2-3) fragment the right side so the remaining
  // squares can't all be placed — the episode ends without a full solve.
  const env = withBag(new PuzzleEnv(), [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]);
  for (let col = 0; col < 6; col++) env.step(anchor(col, 0));
  env.step(anchor(6, 0));
  env.step(anchor(7, 2));
  // play out any remaining legal moves; it cannot reach a full fill
  while (!env.done) {
    const m = env.actionMask();
    let a = -1;
    for (let i = 0; i < m.length; i++) if (m[i]) { a = i; break; }
    if (a < 0) break;
    env.step(a);
  }
  assert.ok(env.done, "episode ends");
  assert.ok(!env.success, "fragmented board cannot be fully solved");
  assert.ok(!env.grid.every((c) => c !== 0), "some cells remain empty");
});

/* --------------------------------------------------------------------------
 * Agent + rollout — the policy only ever acts on legal moves, and a rollout
 * produces a consistent step record for training.
 * ------------------------------------------------------------------------ */
test("Agent never selects a masked (illegal) action", () => {
  const env = new PuzzleEnv();
  const agent = Agent.fresh(env.obsSize, env.actionSize, [16], 5);
  const rng = mulberry32(42);
  for (let trial = 0; trial < 50; trial++) {
    env.reset();
    while (!env.done) {
      const mask = env.actionMask();
      const a = agent.act(env.observe(), mask, rng);
      assert.equal(mask[a], 1, "agent picked an illegal action");
      env.step(a);
    }
  }
});

test("Agent serialize round-trips and rollout records steps", () => {
  const env = new PuzzleEnv();
  const agent = Agent.fresh(env.obsSize, env.actionSize, [16], 2);
  const restored = Agent.fromJSON(JSON.parse(JSON.stringify(agent.toJSON())));
  const obs = env.reset().observe();
  const mask = env.actionMask();
  assert.deepEqual(
    Array.from(restored.policy(obs, mask).probs),
    Array.from(agent.policy(obs, mask).probs),
  );

  const { steps } = rollout(env, agent, mulberry32(1));
  assert.ok(steps.length >= 1 && steps.length <= BAG.length);
  for (const s of steps) assert.equal(s.mask[s.action], 1);
});
