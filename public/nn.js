/**
 * nn.js — a tiny, dependency-free neural network library.
 *
 * Pure ES module: the SAME file runs in the browser (the menu's STATUS demo
 * loads a serialized net for inference) and in Node (scripts/train.mjs trains
 * it). It is intentionally small but *real*: explicit forward/backward through
 * dense layers with selectable activations, He/Xavier init, an SGD/Adam
 * optimizer, and JSON (de)serialization.
 *
 * The design goal is reuse: this is the substrate for the 4x4 puzzle policy
 * today and for a Tetris-playing agent later. Nothing here knows about either
 * task — it only knows vectors in and vectors out. The RL glue (environments,
 * agents, the REINFORCE loop) lives in rl.js and builds on this.
 *
 * Conventions:
 *  - Vectors are plain Float64Array (or number[]); we operate one sample at a
 *    time (no batching tensor machinery) to keep the code legible. Training
 *    accumulates gradients across a batch manually.
 *  - A "net" is a stack of Dense layers each followed by an activation.
 *  - All randomness flows through an injectable rng() so training is
 *    reproducible (see mulberry32).
 */

/* -------------------------------------------------------------------------- */
/* RNG — small, seedable, deterministic. Shared by init and any stochastic    */
/* sampling so a given seed reproduces a run exactly.                         */
/* -------------------------------------------------------------------------- */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function rng() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Gaussian via Box–Muller, driven by the injected uniform rng.
function gaussian(rng) {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/* -------------------------------------------------------------------------- */
/* Activations. Each is { fn, dfn } where dfn takes the *pre-activation* z and  */
/* the activation output a, returning d(out)/d(z). Softmax is handled inline    */
/* in the net because its Jacobian couples outputs.                            */
/* -------------------------------------------------------------------------- */
export const ACTIVATIONS = {
  relu: {
    fn: (z) => (z > 0 ? z : 0),
    dfn: (z) => (z > 0 ? 1 : 0),
  },
  tanh: {
    fn: (z) => Math.tanh(z),
    dfn: (z, a) => 1 - a * a,
  },
  sigmoid: {
    fn: (z) => 1 / (1 + Math.exp(-z)),
    dfn: (z, a) => a * (1 - a),
  },
  linear: {
    fn: (z) => z,
    dfn: () => 1,
  },
};

/* -------------------------------------------------------------------------- */
/* Dense layer: out = act(W·in + b). Stores per-sample caches for backprop and  */
/* accumulates gradients (gW, gb) across a batch until the optimizer applies.   */
/* -------------------------------------------------------------------------- */
export class Dense {
  constructor(inSize, outSize, activation = "relu") {
    this.inSize = inSize;
    this.outSize = outSize;
    this.activation = activation;
    this.W = new Float64Array(outSize * inSize); // row-major [out][in]
    this.b = new Float64Array(outSize);
    this.gW = new Float64Array(outSize * inSize);
    this.gb = new Float64Array(outSize);
    // last forward caches (one sample)
    this._in = null;
    this._z = new Float64Array(outSize);
    this._a = new Float64Array(outSize);
  }

  /** He init for relu, Xavier otherwise. */
  init(rng) {
    const fanIn = this.inSize;
    const scale =
      this.activation === "relu"
        ? Math.sqrt(2 / fanIn)
        : Math.sqrt(1 / fanIn);
    for (let i = 0; i < this.W.length; i++) this.W[i] = gaussian(rng) * scale;
    this.b.fill(0);
    return this;
  }

  forward(input) {
    this._in = input;
    const { inSize, outSize, W, b, _z, _a } = this;
    const act = ACTIVATIONS[this.activation];
    for (let o = 0; o < outSize; o++) {
      let s = b[o];
      const base = o * inSize;
      for (let i = 0; i < inSize; i++) s += W[base + i] * input[i];
      _z[o] = s;
      _a[o] = act.fn(s);
    }
    return _a;
  }

  /**
   * Backprop. `dOut` is d(loss)/d(activation-output) for this layer. Returns
   * d(loss)/d(input) to feed the previous layer, and accumulates gW/gb.
   * If `actAlreadyApplied` is true, dOut is treated as d(loss)/d(z) directly
   * (used when the net folds softmax+cross-entropy into the output gradient).
   */
  backward(dOut, actAlreadyApplied = false) {
    const { inSize, outSize, W, _in, _z, _a, gW, gb } = this;
    const act = ACTIVATIONS[this.activation];
    const dIn = new Float64Array(inSize);
    for (let o = 0; o < outSize; o++) {
      const dz = actAlreadyApplied ? dOut[o] : dOut[o] * act.dfn(_z[o], _a[o]);
      gb[o] += dz;
      const base = o * inSize;
      for (let i = 0; i < inSize; i++) {
        gW[base + i] += dz * _in[i];
        dIn[i] += dz * W[base + i];
      }
    }
    return dIn;
  }

  zeroGrad() {
    this.gW.fill(0);
    this.gb.fill(0);
  }

  toJSON() {
    return {
      inSize: this.inSize,
      outSize: this.outSize,
      activation: this.activation,
      W: Array.from(this.W),
      b: Array.from(this.b),
    };
  }

  static fromJSON(o) {
    const d = new Dense(o.inSize, o.outSize, o.activation);
    d.W.set(o.W);
    d.b.set(o.b);
    return d;
  }
}

/* -------------------------------------------------------------------------- */
/* Net: a stack of Dense layers. Supports a `softmaxOutput` flag so the final   */
/* layer's `linear` logits are turned into a probability distribution for       */
/* policy/classification, with the cross-entropy gradient folded in.            */
/* -------------------------------------------------------------------------- */
export class Net {
  /**
   * @param {number[]} sizes  layer sizes incl. input, e.g. [16, 32, 16]
   * @param {object}   opts   { hidden:"relu", output:"linear", softmax:true, seed }
   */
  constructor(sizes, opts = {}) {
    const hidden = opts.hidden ?? "relu";
    const output = opts.output ?? "linear";
    this.softmax = opts.softmax ?? false;
    this.layers = [];
    for (let i = 0; i < sizes.length - 1; i++) {
      const isLast = i === sizes.length - 2;
      this.layers.push(new Dense(sizes[i], sizes[i + 1], isLast ? output : hidden));
    }
    if (opts.seed !== undefined) this.init(mulberry32(opts.seed));
  }

  init(rng) {
    for (const l of this.layers) l.init(rng);
    return this;
  }

  /** Forward pass. Returns the output vector (softmax-normalized if enabled). */
  forward(input) {
    let x = input;
    for (const l of this.layers) x = l.forward(x);
    return this.softmax ? softmax(x) : x.slice();
  }

  /**
   * Policy/classification backward: given the chosen/target index and the
   * upstream scalar (advantage for RL, or 1 for plain cross-entropy), folds the
   * softmax+cross-entropy gradient and backprops it. Caller must have just run
   * forward(input). Accumulates gradients; call optimizer.step() to apply.
   *
   * For softmax outputs the gradient of -log p[target] w.r.t. logits is
   * (p - onehot(target)); we scale it by `weight` (e.g. -advantage) so a single
   * code path serves supervised CE and REINFORCE.
   */
  backwardPolicy(probs, targetIndex, weight = 1) {
    const out = this.layers[this.layers.length - 1];
    const dz = new Float64Array(out.outSize);
    for (let k = 0; k < out.outSize; k++) {
      dz[k] = weight * (probs[k] - (k === targetIndex ? 1 : 0));
    }
    let d = out.backward(dz, true); // dz is already d(loss)/d(logits)
    for (let i = this.layers.length - 2; i >= 0; i--) d = this.layers[i].backward(d);
    return d;
  }

  /**
   * Backward from an explicit gradient on the OUTPUT LAYER LOGITS (pre-softmax).
   * This is the general entry point for policy-gradient style updates where the
   * caller has already assembled d(loss)/d(logits) (e.g. REINFORCE + an entropy
   * bonus). The output activation must be linear (logits).
   */
  backwardFromLogits(dLogits) {
    const out = this.layers[this.layers.length - 1];
    let d = out.backward(dLogits, true);
    for (let i = this.layers.length - 2; i >= 0; i--) d = this.layers[i].backward(d);
    return d;
  }

  /**
   * Regression backward from an explicit d(loss)/d(output) vector (e.g. the
   * value-head MSE gradient when this net is later used as a critic).
   */
  backward(dOut) {
    let d = dOut;
    for (let i = this.layers.length - 1; i >= 0; i--) d = this.layers[i].backward(d);
    return d;
  }

  zeroGrad() {
    for (const l of this.layers) l.zeroGrad();
  }

  toJSON() {
    return {
      format: "nn.v1",
      softmax: this.softmax,
      layers: this.layers.map((l) => l.toJSON()),
    };
  }

  static fromJSON(o) {
    const net = Object.create(Net.prototype);
    net.softmax = !!o.softmax;
    net.layers = o.layers.map(Dense.fromJSON);
    return net;
  }
}

/* Numerically stable softmax. */
export function softmax(logits) {
  let max = -Infinity;
  for (const v of logits) if (v > max) max = v;
  const out = new Float64Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    out[i] = Math.exp(logits[i] - max);
    sum += out[i];
  }
  for (let i = 0; i < out.length; i++) out[i] /= sum || 1;
  return out;
}

/** Sample an index from a probability vector using the injected rng. */
export function sample(probs, rng) {
  const r = rng();
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i];
    if (r <= acc) return i;
  }
  return probs.length - 1;
}

/** Argmax index (greedy action / prediction). */
export function argmax(v) {
  let best = 0;
  for (let i = 1; i < v.length; i++) if (v[i] > v[best]) best = i;
  return best;
}

/* -------------------------------------------------------------------------- */
/* Optimizers. Both consume the per-layer accumulated gradients. Adam keeps     */
/* moment estimates keyed per parameter for stable RL training.                 */
/* -------------------------------------------------------------------------- */
export class SGD {
  constructor(net, { lr = 0.01, momentum = 0 } = {}) {
    this.net = net;
    this.lr = lr;
    this.momentum = momentum;
    this.vW = net.layers.map((l) => new Float64Array(l.W.length));
    this.vb = net.layers.map((l) => new Float64Array(l.b.length));
  }
  step(scale = 1) {
    const { lr, momentum } = this;
    this.net.layers.forEach((l, li) => {
      const vW = this.vW[li];
      const vb = this.vb[li];
      for (let i = 0; i < l.W.length; i++) {
        const g = l.gW[i] * scale;
        vW[i] = momentum * vW[i] - lr * g;
        l.W[i] += vW[i];
      }
      for (let i = 0; i < l.b.length; i++) {
        const g = l.gb[i] * scale;
        vb[i] = momentum * vb[i] - lr * g;
        l.b[i] += vb[i];
      }
    });
    this.net.zeroGrad();
  }
}

export class Adam {
  constructor(net, { lr = 0.005, beta1 = 0.9, beta2 = 0.999, eps = 1e-8 } = {}) {
    this.net = net;
    this.lr = lr;
    this.b1 = beta1;
    this.b2 = beta2;
    this.eps = eps;
    this.t = 0;
    this.mW = net.layers.map((l) => new Float64Array(l.W.length));
    this.vW = net.layers.map((l) => new Float64Array(l.W.length));
    this.mb = net.layers.map((l) => new Float64Array(l.b.length));
    this.vb = net.layers.map((l) => new Float64Array(l.b.length));
  }
  /** Apply accumulated grads. `scale` lets you average a batch (e.g. 1/N). */
  step(scale = 1) {
    this.t++;
    const { b1, b2, eps, lr } = this;
    const bc1 = 1 - Math.pow(b1, this.t);
    const bc2 = 1 - Math.pow(b2, this.t);
    const upd = (params, grads, m, v) => {
      for (let i = 0; i < params.length; i++) {
        const g = grads[i] * scale;
        m[i] = b1 * m[i] + (1 - b1) * g;
        v[i] = b2 * v[i] + (1 - b2) * g * g;
        const mh = m[i] / bc1;
        const vh = v[i] / bc2;
        params[i] -= (lr * mh) / (Math.sqrt(vh) + eps);
      }
    };
    this.net.layers.forEach((l, li) => {
      upd(l.W, l.gW, this.mW[li], this.vW[li]);
      upd(l.b, l.gb, this.mb[li], this.vb[li]);
    });
    this.net.zeroGrad();
  }
}
