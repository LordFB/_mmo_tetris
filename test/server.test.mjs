import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { createServer, verifyPlay } from "../server.mjs";
import { runReplay, INPUT_DOWN } from "../public/nes-engine.js";

test("serves the health endpoint and frontend", async (context) => {
  const directory = await mkdtemp(path.join(os.tmpdir(), "mmo-tetris-"));
  const server = createServer({ databasePath: path.join(directory, "plays.sqlite") });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  context.after(async () => {
    await new Promise((resolve, reject) => server.close((error) => (error ? reject(error) : resolve())));
    await rm(directory, { recursive: true, force: true });
  });

  const address = server.address();
  assert.ok(address && typeof address !== "string");
  const base = `http://127.0.0.1:${address.port}`;

  const health = await fetch(`${base}/api/health`);
  assert.equal(health.status, 200);
  assert.deepEqual(await health.json(), { ok: true, players: 0 });

  const page = await fetch(base);
  assert.equal(page.status, 200);
  assert.match(await page.text(), /MMO\s*TETRIS/i);
});

test("verifyPlay re-simulates and rejects implausible submissions", () => {
  // Build a genuine finished game.
  const inputs = [];
  for (let i = 0; i < 2000; i++) inputs.push(INPUT_DOWN);
  const expected = runReplay({ seed: 0x2222, startLevel: 0, inputs });

  const good = verifyPlay({ seed: 0x2222, startLevel: 0, inputs });
  assert.equal(good.ok, true);
  assert.equal(good.score, expected.score);

  // Out-of-range bits are illegal controller input.
  const illegal = verifyPlay({ seed: 0x2222, startLevel: 0, inputs: [255] });
  assert.equal(illegal.ok, false);

  // Empty / bad params.
  assert.equal(verifyPlay({ seed: -5, startLevel: 0, inputs: [0] }).ok, false);
  assert.equal(verifyPlay({ seed: 0, startLevel: 50, inputs: [0] }).ok, false);
  assert.equal(verifyPlay({ seed: 0, startLevel: 0, inputs: [] }).ok, false);
});

test("verifyPlay classifies rejections by severity (block + scale the troll)", () => {
  // Forged controller bytes can only be hand-built -> highest severity.
  assert.equal(verifyPlay({ seed: 0x1, startLevel: 0, inputs: [255] }).severity, 3);
  // Claiming a record for a game that never topped out -> forged -> severity 3.
  const unfinished = verifyPlay({ seed: 0x1, startLevel: 0, inputs: [0] });
  assert.equal(unfinished.ok, false);
  assert.equal(unfinished.reason, "not finished");
  assert.equal(unfinished.severity, 3);
  // Bounds-violating but possibly a stale/buggy client -> malformed -> severity 2.
  assert.equal(verifyPlay({ seed: -5, startLevel: 0, inputs: [0] }).severity, 2);
  assert.equal(verifyPlay({ seed: 0, startLevel: 50, inputs: [0] }).severity, 2);
  // A valid play carries no severity — nothing to troll.
  const inputs = [];
  for (let i = 0; i < 2000; i++) inputs.push(INPUT_DOWN);
  assert.equal(verifyPlay({ seed: 0x2222, startLevel: 0, inputs }).severity, undefined);
});
