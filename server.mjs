import { createServer as createHttpServer } from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import Database from "better-sqlite3";
import express from "express";
import { WebSocketServer, WebSocket } from "ws";
import { runReplay, MAX_REPLAY_FRAMES, INPUT_MASK } from "./public/nes-engine.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const publicDirectory = path.join(here, "public");

// A play_complete carries one byte per frame, so the message can be large.
// Cap generously but bounded: MAX_REPLAY_FRAMES bytes + JSON overhead.
const MAX_MESSAGE_BYTES = MAX_REPLAY_FRAMES + 4096;
const MAX_CLIENTS = 500;
const HEARTBEAT_MS = 30_000;
// One verification can run up to MAX_REPLAY_FRAMES steps; rate-limit submissions.
const MIN_SUBMIT_INTERVAL_MS = 3_000;

function cleanName(value) {
  return typeof value === "string"
    ? value.replace(/[^a-zA-Z0-9 _.\-]/g, "").trim().slice(0, 16) || "ANON"
    : "ANON";
}

/**
 * Anti-cheat: re-simulate the submitted replay with the shared NES engine and
 * derive the authoritative score. The client's claimed numbers are never
 * trusted — only the result of re-running { seed, startLevel, inputs }.
 *
 * Rejections carry a `severity` so the front-end can scale its (purely
 * cosmetic) response. The submission is ALWAYS dropped regardless of severity —
 * severity only governs how silly the client gets about it:
 *   1 = honest mistake (empty / no-name / rate-limited — not produced here)
 *   2 = malformed: bounds-violating but possibly a stale/buggy client
 *   3 = forged: illegal controller bytes, or claiming a record for a game that
 *       never topped out — these only happen if someone hand-built the message
 */
export function verifyPlay(message) {
  const { seed, startLevel, inputs } = message;
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffff) {
    return { ok: false, reason: "bad seed", severity: 2 };
  }
  if (!Number.isInteger(startLevel) || startLevel < 0 || startLevel > 29) {
    return { ok: false, reason: "bad level", severity: 2 };
  }
  if (!Array.isArray(inputs) || inputs.length === 0) {
    return { ok: false, reason: "no inputs", severity: 2 };
  }
  if (inputs.length > MAX_REPLAY_FRAMES) {
    return { ok: false, reason: "too long", severity: 2 };
  }
  // Inputs must be small integers within the legal controller bitfield. A byte
  // outside the legal mask cannot come from the real client — it was forged.
  const buf = new Uint8Array(inputs.length);
  for (let i = 0; i < inputs.length; i++) {
    const v = inputs[i];
    if (!Number.isInteger(v) || v < 0 || v > 0xff || (v & ~INPUT_MASK) !== 0) {
      return { ok: false, reason: "bad input", severity: 3 };
    }
    buf[i] = v;
  }

  let final;
  try {
    final = runReplay({ seed, startLevel, inputs: buf });
  } catch (err) {
    return { ok: false, reason: "replay error", severity: 2 };
  }

  // A ranked play must have actually ended (topped out). This stops someone
  // from submitting a still-running game's partial high score.
  if (final.phase !== "game_over") {
    return { ok: false, reason: "not finished", severity: 3 };
  }

  return {
    ok: true,
    score: final.score,
    lines: final.lines,
    level: final.level,
    frames: final.frame,
  };
}

export function createPlayStore(databasePath) {
  const database = new Database(databasePath);
  database.pragma("journal_mode = WAL");
  database.exec(`
    CREATE TABLE IF NOT EXISTS plays (
      id INTEGER PRIMARY KEY,
      player_name TEXT NOT NULL,
      score INTEGER NOT NULL,
      lines INTEGER NOT NULL,
      level INTEGER NOT NULL,
      seed INTEGER NOT NULL,
      start_level INTEGER NOT NULL,
      frames INTEGER NOT NULL,
      verified INTEGER NOT NULL DEFAULT 1,
      played_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS plays_leaderboard ON plays (score DESC, lines DESC, id ASC);
  `);

  // Migration: an older build created `plays` without the replay-audit columns.
  // CREATE TABLE IF NOT EXISTS leaves such a table untouched, so add any column
  // that is missing. Each ADD COLUMN must supply a constant default because the
  // table may already hold rows.
  const existing = new Set(
    database.prepare("PRAGMA table_info(plays)").all().map((c) => c.name),
  );
  const ensureColumn = (name, definition) => {
    if (!existing.has(name)) database.exec(`ALTER TABLE plays ADD COLUMN ${name} ${definition}`);
  };
  ensureColumn("seed", "INTEGER NOT NULL DEFAULT 0");
  ensureColumn("start_level", "INTEGER NOT NULL DEFAULT 0");
  ensureColumn("frames", "INTEGER NOT NULL DEFAULT 0");
  ensureColumn("verified", "INTEGER NOT NULL DEFAULT 1");

  const insert = database.prepare(
    `INSERT INTO plays (player_name, score, lines, level, seed, start_level, frames)
     VALUES (@name, @score, @lines, @level, @seed, @startLevel, @frames)`,
  );
  const leaderboard = database.prepare(`
    SELECT player_name AS name, score, lines, level
    FROM plays
    ORDER BY score DESC, lines DESC, id ASC
    LIMIT 25
  `);
  return {
    recordPlay(row) { insert.run(row); },
    leaderboard() { return leaderboard.all(); },
    close() { database.close(); },
  };
}

export function createApp(store, playerCount = () => 0) {
  const app = express();
  app.get("/api/health", (_request, response) =>
    response.json({ ok: true, players: playerCount() }),
  );
  app.get("/api/leaderboard", (_request, response) =>
    response.json({ entries: store.leaderboard() }),
  );
  app.use(express.static(publicDirectory));
  return app;
}

export function createServer({
  databasePath = process.env.DATABASE_PATH ?? path.join(here, "plays.sqlite"),
} = {}) {
  // One giant room: every connected socket is a participant.
  const clients = new Set();
  let closed = false; // set on shutdown so late socket-close handlers don't touch the DB
  const store = createPlayStore(databasePath);
  const server = createHttpServer(createApp(store, () => clients.size));
  const wss = new WebSocketServer({ noServer: true, maxPayload: MAX_MESSAGE_BYTES });

  const send = (socket, message) => {
    if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
  };

  // Snapshots arrive ~10x/sec, so a tampered stream could spam the client with
  // troll events. Notify at most once per TAMPER_NOTICE_MS per socket. The bad
  // data is always dropped regardless; this only governs the cosmetic notice.
  const TAMPER_NOTICE_MS = 4_000;
  const maybeFlagTamper = (socket, severity) => {
    const now = Date.now();
    if (now - (socket.lastTamper || 0) < TAMPER_NOTICE_MS) return;
    socket.lastTamper = now;
    send(socket, { type: "tamper", reason: "bad snapshot", severity });
  };
  const broadcast = (message) => clients.forEach((socket) => send(socket, message));
  const presence = () => broadcast({ type: "presence", players: clients.size });

  /**
   * The TOP 10, blended live. Persisted verified high scores form the base; on
   * top we fold in the current room's in-progress players (their live snapshot
   * score), marked `live: true` so the client can flag them. Live scores are
   * display-only and never persisted — only a completed, replay-verified game
   * is written to the database, so this cannot be used to forge a record.
   *
   * If a live player's running score would place them in the TOP 10, they show
   * there in real time; once they top out and the play verifies, the same score
   * reappears as a persisted (non-live) entry via the normal record path.
   */
  function combinedLeaderboard() {
    if (closed) return [];
    const persisted = store.leaderboard().map((e) => ({ ...e, live: false }));
    const live = [...clients]
      .filter((c) => c.playerName && c.snapshot && !c.snapshot.gameOver && c.liveScore > 0)
      .map((c) => ({
        name: c.playerName,
        score: c.liveScore,
        lines: c.snapshot.lines,
        level: c.snapshot.level,
        live: true,
      }));
    return [...persisted, ...live]
      .sort((a, b) => b.score - a.score || b.lines - a.lines)
      .slice(0, 10);
  }

  // Leaderboard broadcasts are coalesced. Snapshots arrive ~10x/sec per player,
  // so a naive broadcast-per-snapshot is O(players^2) messages/sec. Instead we
  // mark the board dirty and flush at most LEADERBOARD_HZ times/sec, collapsing
  // a burst of updates into a single broadcast. Verified-play and join/leave
  // events flush immediately so records and presence are never delayed.
  const LEADERBOARD_MIN_INTERVAL_MS = 200; // <= 5 broadcasts/sec
  let lbDirty = false;
  let lbLastFlush = 0;
  let lbTimer = null;

  function flushLeaderboard() {
    lbDirty = false;
    lbLastFlush = Date.now();
    if (lbTimer) { clearTimeout(lbTimer); lbTimer = null; }
    if (closed) return;
    broadcast({ type: "leaderboard", entries: combinedLeaderboard() });
  }

  /** Request a leaderboard broadcast; coalesces bursts. */
  function pushLeaderboard(immediate = false) {
    if (closed) return;
    if (immediate) return flushLeaderboard();
    lbDirty = true;
    const since = Date.now() - lbLastFlush;
    if (since >= LEADERBOARD_MIN_INTERVAL_MS) {
      flushLeaderboard();
    } else if (!lbTimer) {
      lbTimer = setTimeout(() => { lbTimer = null; if (lbDirty) flushLeaderboard(); },
        LEADERBOARD_MIN_INTERVAL_MS - since);
    }
  }

  /**
   * Rank live players by current score and assign each their two neighbours.
   * The view the user asked for: your field center, the two players ranked
   * adjacent to you on the left and right. If you are #1, you see #2 and #3.
   */
  function liveRanking() {
    return [...clients]
      .filter((c) => c.playerName)
      .sort((a, b) => (b.liveScore || 0) - (a.liveScore || 0) || a.joinOrder - b.joinOrder);
  }

  function neighboursFor(ranked, idx) {
    let leftIdx;
    let rightIdx;
    if (idx === 0) {
      // #1 sees #2 and #3.
      leftIdx = 1;
      rightIdx = 2;
    } else {
      leftIdx = idx - 1;
      rightIdx = idx + 1;
    }
    const pick = (i) => {
      const s = ranked[i];
      return s && s !== undefined
        ? { name: s.playerName, snap: s.snapshot ?? null }
        : null;
    };
    return { left: pick(leftIdx), right: pick(rightIdx) };
  }

  function pushNeighbours() {
    const ranked = liveRanking();
    ranked.forEach((socket, idx) => {
      send(socket, {
        type: "neighbours",
        position: idx + 1, // 1-based live rank
        ...neighboursFor(ranked, idx),
      });
    });
  }

  server.on("upgrade", (request, socket, head) => {
    const url = new URL(request.url ?? "/", "http://localhost");
    if (url.pathname !== "/ws" || clients.size >= MAX_CLIENTS) return socket.destroy();
    wss.handleUpgrade(request, socket, head, (ws) => wss.emit("connection", ws));
  });

  let joinCounter = 0;

  wss.on("connection", (socket) => {
    socket.alive = true;
    socket.playerName = null;
    socket.liveScore = 0;
    socket.snapshot = null;
    socket.joinOrder = joinCounter++;
    socket.lastSubmit = 0;
    socket.lastTamper = 0;
    clients.add(socket);

    socket.on("pong", () => { socket.alive = true; });

    socket.on("message", (data, isBinary) => {
      if (isBinary || data.length > MAX_MESSAGE_BYTES) return socket.close(1009, "too big");
      let message;
      try { message = JSON.parse(data.toString()); } catch { return socket.close(1008, "bad json"); }
      if (!message || typeof message.type !== "string") return;

      switch (message.type) {
        case "hello": {
          socket.playerName = cleanName(message.name);
          send(socket, {
            type: "welcome",
            players: clients.size,
            entries: combinedLeaderboard(),
          });
          presence();
          pushNeighbours();
          break;
        }

        case "start": {
          socket.liveScore = 0;
          socket.snapshot = null;
          pushNeighbours();
          pushLeaderboard();
          break;
        }

        case "snapshot": {
          if (!socket.playerName) return;
          // Trust the snapshot ONLY for the live spectator view — never for
          // ranking. Validate shape and bound it.
          if (!Array.isArray(message.board) || message.board.length !== 200) {
            return maybeFlagTamper(socket, 2);
          }
          // Drop out-of-order snapshots so neighbour views never go backwards.
          const frame = Number.isInteger(message.frame) ? message.frame : 0;
          if (socket.snapshot && frame < socket.snapshot.frame) break;
          const board = new Uint8Array(200);
          let dirtyCells = false;
          for (let i = 0; i < 200; i++) {
            const v = message.board[i];
            if (Number.isInteger(v) && v >= 0 && v <= 7) {
              board[i] = v;
            } else {
              // The real client only ever emits 0..7. An out-of-range cell is a
              // hand-edited snapshot; clamp it (block) and flag the attempt.
              board[i] = 0;
              dirtyCells = true;
            }
          }
          if (dirtyCells) maybeFlagTamper(socket, 2);
          socket.snapshot = {
            frame,
            board: Array.from(board),
            score: Number(message.score) || 0,
            lines: Number(message.lines) || 0,
            level: Number(message.level) || 0,
            gameOver: Boolean(message.gameOver),
          };
          socket.liveScore = socket.snapshot.score;
          pushNeighbours();
          pushLeaderboard();
          break;
        }

        case "play_complete": {
          if (!socket.playerName) {
            return send(socket, { type: "play_rejected", reason: "no name", severity: 1 });
          }
          const now = Date.now();
          if (now - socket.lastSubmit < MIN_SUBMIT_INTERVAL_MS) {
            return send(socket, { type: "play_rejected", reason: "rate limited", severity: 1 });
          }
          socket.lastSubmit = now;

          const result = verifyPlay(message);
          if (!result.ok) {
            // Blocked: the score is never recorded. `severity` only tells the
            // client how silly to be about the rejection — see app.js troll().
            return send(socket, {
              type: "play_rejected",
              reason: result.reason,
              severity: result.severity ?? 2,
            });
          }
          store.recordPlay({
            name: socket.playerName,
            score: result.score,
            lines: result.lines,
            level: result.level,
            seed: message.seed,
            startLevel: message.startLevel,
            frames: result.frames,
          });
          send(socket, { type: "play_accepted", score: result.score, lines: result.lines, level: result.level });
          pushLeaderboard(true); // a new record: flush at once
          break;
        }
      }
    });

    const disconnect = () => {
      if (!clients.delete(socket)) return;
      presence();
      pushNeighbours();
      pushLeaderboard(true); // a player left: flush at once
    };
    socket.on("close", disconnect);
    socket.on("error", disconnect);
  });

  const heartbeat = setInterval(() => {
    for (const socket of clients) {
      if (!socket.alive) socket.terminate();
      else { socket.alive = false; socket.ping(); }
    }
  }, HEARTBEAT_MS);

  server.on("close", () => {
    closed = true; // stop any late close-handlers from querying the DB
    clearInterval(heartbeat);
    if (lbTimer) { clearTimeout(lbTimer); lbTimer = null; }
    for (const socket of clients) socket.terminate();
    store.close();
  });

  return server;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const port = Number.parseInt(process.env.PORT ?? "3000", 10);
  createServer().listen(port, () => {
    console.log(`MMO Tetris (vanilla) listening on http://localhost:${port}`);
  });
}
