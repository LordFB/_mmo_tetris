import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import WebSocket from "ws";
import { createServer } from "../server.mjs";
import { runReplay, INPUT_DOWN, INPUT_ROTATE_CW } from "../public/nes-engine.js";

/** Resolve with the next message whose `type` matches, ignoring others. */
function waitFor(socket, type, timeoutMs = 4000) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      socket.off("message", onMessage);
      reject(new Error(`timed out waiting for ${type}`));
    }, timeoutMs);
    function onMessage(data) {
      const msg = JSON.parse(data.toString());
      if (msg.type === type) {
        clearTimeout(timer);
        socket.off("message", onMessage);
        resolve(msg);
      }
    }
    socket.on("message", onMessage);
  });
}

function open(port) {
  const socket = new WebSocket(`ws://127.0.0.1:${port}/ws`);
  return new Promise((resolve, reject) => {
    socket.once("open", () => resolve(socket));
    socket.once("error", reject);
  });
}

/** A real, finished game replay the server can verify. */
function buildReplay() {
  const seed = 0x4321;
  const startLevel = 0;
  const inputs = [];
  for (let i = 0; i < 3000; i++) {
    inputs.push(i % 30 === 0 ? INPUT_ROTATE_CW : INPUT_DOWN);
  }
  const final = runReplay({ seed, startLevel, inputs });
  return { seed, startLevel, inputs, expected: final };
}

async function withServer(context, fn) {
  const directory = await mkdtemp(path.join(os.tmpdir(), "mmo-tetris-"));
  const server = createServer({ databasePath: path.join(directory, "plays.sqlite") });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  context.after(async () => {
    await new Promise((resolve, reject) =>
      server.close((error) => (error ? reject(error) : resolve())),
    );
    await rm(directory, { recursive: true, force: true });
  });
  const { port } = server.address();
  return fn(port, `http://127.0.0.1:${port}`);
}

test("verifies a real replay and publishes the derived score", async (context) => {
  await withServer(context, async (port, base) => {
    const replay = buildReplay();
    assert.equal(replay.expected.phase, "game_over");

    const socket = await open(port);
    socket.send(JSON.stringify({ type: "hello", name: "PLAYER 1" }));
    const welcome = await waitFor(socket, "welcome");
    assert.equal(welcome.players, 1);

    socket.send(JSON.stringify({
      type: "play_complete",
      seed: replay.seed,
      startLevel: replay.startLevel,
      inputs: replay.inputs,
    }));

    const accepted = await waitFor(socket, "play_accepted");
    // The published score is the SERVER's re-simulation, not the client's claim.
    assert.equal(accepted.score, replay.expected.score);
    assert.equal(accepted.lines, replay.expected.lines);

    const board = await (await fetch(`${base}/api/leaderboard`)).json();
    assert.equal(board.entries[0].name, "PLAYER 1");
    assert.equal(board.entries[0].score, replay.expected.score);
    socket.close();
  });
});

test("rejects a forged score whose inputs do not produce it", async (context) => {
  await withServer(context, async (port, base) => {
    const socket = await open(port);
    socket.send(JSON.stringify({ type: "hello", name: "CHEATER" }));
    await waitFor(socket, "welcome");

    // No inputs but a huge claimed score — the old protocol would have trusted
    // this. The new server ignores claims entirely and re-simulates.
    socket.send(JSON.stringify({
      type: "play_complete",
      seed: 0x1111,
      startLevel: 0,
      inputs: [],
      score: 999999,
    }));
    const rejected = await waitFor(socket, "play_rejected");
    assert.equal(rejected.reason, "no inputs");

    const board = await (await fetch(`${base}/api/leaderboard`)).json();
    assert.equal(board.entries.length, 0);
    socket.close();
  });
});

test("clamps a forged live snapshot score before it reaches the leaderboard", async (context) => {
  await withServer(context, async (port) => {
    const a = await open(port);
    a.send(JSON.stringify({ type: "hello", name: "FORGER" }));
    await waitFor(a, "welcome");
    a.send(JSON.stringify({ type: "start" }));

    const board = new Array(200).fill(0);
    board[199] = 1; // one locked cell so the snapshot shape is valid
    // A tampered client claims an absurd live score. Number("1e308")||0 would
    // have let this through and pinned FORGER at #1 on everyone's board; the
    // server must clamp score/lines/level to sane caps before broadcasting.
    a.send(JSON.stringify({
      type: "snapshot",
      board,
      score: 1e308,    // -> Infinity under naive coercion
      lines: -5,       // negative
      level: 99999,    // wildly out of range
      gameOver: false,
    }));

    let entry;
    for (let tries = 0; tries < 5; tries++) {
      const lb = await waitFor(a, "leaderboard");
      entry = lb.entries.find((e) => e.name === "FORGER");
      if (entry) break;
    }
    assert.ok(entry, "FORGER should still appear (live, just clamped)");
    assert.equal(entry.score, 999999, "score clamped to MAX_SCORE");
    assert.ok(Number.isFinite(entry.score), "score must be finite (no Infinity)");
    assert.equal(entry.lines, 0, "negative lines clamped to 0");
    assert.equal(entry.level, 99, "level clamped to MAX_LEVEL");
    a.close();
  });
});

test("tracks presence as players join and leave", async (context) => {
  await withServer(context, async (port) => {
    const a = await open(port);
    a.send(JSON.stringify({ type: "hello", name: "A" }));
    await waitFor(a, "welcome");

    const joined = waitFor(a, "presence");
    const b = await open(port);
    b.send(JSON.stringify({ type: "hello", name: "B" }));
    const p = await joined;
    assert.equal(p.players, 2);

    const left = waitFor(a, "presence");
    b.close();
    const p2 = await left;
    assert.equal(p2.players, 1);
    a.close();
  });
});

test("blends in-progress players into the live TOP 10", async (context) => {
  await withServer(context, async (port) => {
    const a = await open(port);
    a.send(JSON.stringify({ type: "hello", name: "ALICE" }));
    await waitFor(a, "welcome");

    // ALICE starts and pushes a live snapshot with a running score.
    a.send(JSON.stringify({ type: "start" }));
    const board = new Array(200).fill(0);
    board[199] = 1; // one locked cell so the snapshot is valid
    a.send(JSON.stringify({
      type: "snapshot",
      board,
      score: 5432,
      lines: 7,
      level: 1,
      gameOver: false,
    }));

    // The leaderboard broadcast should now contain ALICE marked live.
    let entry;
    for (let tries = 0; tries < 5; tries++) {
      const lb = await waitFor(a, "leaderboard");
      entry = lb.entries.find((e) => e.name === "ALICE");
      if (entry) break;
    }
    assert.ok(entry, "ALICE should appear in the live leaderboard");
    assert.equal(entry.score, 5432);
    assert.equal(entry.live, true);
    a.close();
  });
});
