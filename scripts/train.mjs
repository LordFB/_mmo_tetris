/**
 * train.mjs — REINFORCE trainer for the 4x4 puzzle policy (run via `npm run
 * train`). Trains the same Agent/Net the browser menu loads for its STATUS
 * demo, and serializes the result to public/status-policy.json.
 *
 * The loop is deliberately generic (env + agent + policy gradient), so pointing
 * it at a Tetris env later is a matter of swapping the Env, not rewriting the
 * trainer.
 *
 *   node scripts/train.mjs [--episodes N] [--lr X] [--seed S] [--out path]
 *
 * Algorithm: vanilla REINFORCE with discounted returns and a moving-average
 * reward baseline (variance reduction). For each episode we roll out under the
 * current stochastic policy, compute per-step discounted returns G_t, subtract
 * the baseline, and accumulate the policy-gradient ∇ log π(a_t|s_t) · (G_t − b).
 * We update once per episode (batch = the episode's steps).
 */

import { writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { Adam, softmax } from "../public/nn.js";
import { PuzzleEnv, Agent, rollout, GRID, GRID_H } from "../public/rl.js";
import { mulberry32 } from "../public/nn.js";

const here = path.dirname(fileURLToPath(import.meta.url));

/* ----------------------------- arg parsing ------------------------------- */
function arg(name, def) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : def;
}
const EPISODES = parseInt(arg("episodes", "30000"), 10);
const LR = parseFloat(arg("lr", "0.005"));
const GAMMA = parseFloat(arg("gamma", "0.99"));
const ENTROPY = parseFloat(arg("entropy", "0.02")); // exploration bonus weight
const SEED = parseInt(arg("seed", "12345"), 10);
const OUT = arg("out", path.join(here, "..", "public", "status-policy.json"));
const HIDDEN = arg("hidden", "32,32").split(",").map(Number);

/* ------------------------------- setup ----------------------------------- */
const rng = mulberry32(SEED);
const env = new PuzzleEnv();
const agent = Agent.fresh(env.obsSize, env.actionSize, HIDDEN, SEED);
const opt = new Adam(agent.net, { lr: LR });

console.log(
  `training puzzle policy: episodes=${EPISODES} lr=${LR} gamma=${GAMMA} ` +
    `hidden=[${HIDDEN}] obs=${env.obsSize} actions=${env.actionSize}`,
);

/* --------------------------- training loop ------------------------------- */
let baseline = 0; // moving-average return baseline
const BASE_DECAY = 0.99;
let recentSuccess = 0; // EMA of solve rate for logging

const t0 = Date.now();
for (let ep = 0; ep < EPISODES; ep++) {
  const { steps, total, success } = rollout(env, agent, rng, { greedy: false });

  // discounted returns G_t (reward-to-go)
  const returns = new Array(steps.length);
  let g = 0;
  for (let i = steps.length - 1; i >= 0; i--) {
    g = steps[i].reward + GAMMA * g;
    returns[i] = g;
  }

  // baseline = moving average of episode return (a single scalar baseline is
  // enough for this tiny task and keeps the trainer dependency-free of a critic)
  baseline = BASE_DECAY * baseline + (1 - BASE_DECAY) * total;

  // accumulate the combined (policy-gradient + entropy-bonus) logit gradient
  // over the episode, then one optimizer step.
  agent.net.zeroGrad();
  for (let i = 0; i < steps.length; i++) {
    const { obs, mask, action } = steps[i];
    // recompute the masked policy at this state (forward fills caches)
    const logits = agent.net.forward(obs);
    const masked = logits.slice();
    for (let k = 0; k < masked.length; k++) if (!mask[k]) masked[k] = -1e9;
    const probs = softmax(masked);
    const advantage = returns[i] - baseline;

    // Entropy H of the (masked) policy, for the exploration bonus.
    let H = 0;
    for (let k = 0; k < probs.length; k++) {
      if (mask[k]) H -= probs[k] * Math.log(probs[k] + 1e-12);
    }

    // d(loss)/d(logit_k), descending L = -A·log π(a) - ENTROPY·H:
    //   policy-gradient term : A·(p_k − 1[k=a])
    //   entropy-bonus term   : ENTROPY·p_k·(log p_k + H)
    // Illegal actions carry ~0 prob, so their gradient is naturally ~0.
    const dLogits = new Float64Array(probs.length);
    for (let k = 0; k < probs.length; k++) {
      if (!mask[k]) continue;
      const pg = advantage * (probs[k] - (k === action ? 1 : 0));
      const ent = ENTROPY * probs[k] * (Math.log(probs[k] + 1e-12) + H);
      dLogits[k] = pg + ent;
    }
    agent.net.backwardFromLogits(dLogits);
  }
  // average the accumulated grads over the episode's steps
  opt.step(steps.length > 0 ? 1 / steps.length : 0);

  recentSuccess = 0.995 * recentSuccess + 0.005 * (success ? 1 : 0);
  if ((ep + 1) % 1000 === 0) {
    console.log(
      `ep ${String(ep + 1).padStart(6)} | return ${total.toFixed(1).padStart(6)} ` +
        `| baseline ${baseline.toFixed(2).padStart(6)} | solve~ ${(recentSuccess * 100).toFixed(1)}%`,
    );
  }
}

/* --------------------------- evaluation ---------------------------------- */
// Greedy eval over many fresh episodes to report the deterministic solve rate
// the menu demo will actually exhibit.
let solved = 0;
const EVAL = 2000;
const evalRng = mulberry32(SEED ^ 0x9e3779b9);
for (let i = 0; i < EVAL; i++) {
  const { success } = rollout(env, agent, evalRng, { greedy: true, record: false });
  if (success) solved++;
}
const greedyRate = (solved / EVAL) * 100;
console.log(`\ndone in ${((Date.now() - t0) / 1000).toFixed(1)}s — greedy solve rate: ${greedyRate.toFixed(1)}%`);

/* --------------------------- serialize ----------------------------------- */
const payload = {
  ...agent.toJSON(),
  meta: {
    task: `puzzle-${GRID}x${GRID_H}-fill-random-bag`,
    episodes: EPISODES,
    lr: LR,
    gamma: GAMMA,
    hidden: HIDDEN,
    seed: SEED,
    greedySolveRate: Number(greedyRate.toFixed(2)),
    trainedAt: new Date().toISOString(),
  },
};
writeFileSync(OUT, JSON.stringify(payload));
console.log(`saved policy -> ${path.relative(path.join(here, ".."), OUT)} (${greedyRate.toFixed(1)}% greedy solve)`);
