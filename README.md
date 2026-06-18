# FreeRouter-Go

Self-hosted, OpenAI-compatible LLM router. Sends each request to the cheapest
model capable of handling it, using your own API keys — no middleman markup.

Unlike keyword-only routers, routing here is **data-driven**: models live in a
table (the *vademécum*) with `tier_max / cost / weight / mcp_native`, and the
cheapest sufficient model wins. A lightweight heuristic classifier is only a
**fallback** for requests that don't declare their own tier.

## Inspiration & lineage

This project is a Go reimplementation inspired by
**[openfreerouter/freerouter](https://github.com/openfreerouter/freerouter)**
(TypeScript, MIT) — the 14-dimension classifier and the OpenAI-compatible
drop-in idea come from there. FreeRouter itself is forked from **ClawRouter**.

It is **not a 1:1 port**. Two concrete bugs in the original were fixed here, and
the routing model was changed to match how my own
[Pillbox](https://github.com/andyeswong/pillbox) agent system already routes:

| Original (freerouter, TS)                                   | FreeRouter-Go                                                          |
|-------------------------------------------------------------|-----------------------------------------------------------------------|
| Keyword match via naive `text.includes(kw)` (substring)     | Word-boundary `\b…\b` match — no more `art` matching `start`           |
| Savings baseline hardcoded to `claude-opus` pricing         | Baseline = most-expensive **enabled** model in your own table         |
| Tiers + model choice from static JSON                       | Vademécum in SQLite, CRUD + health-scan at runtime                     |
| Routing driven by classifier only                           | Caller may declare `tier` / `requires_mcp`; classifier is the fallback |

Two patterns are borrowed directly from my existing Go/agent work:

- **`scopeCandidatesFor` ordering** (`tier_max ASC, cost ASC, weight DESC`) from
  Pillbox — the cheapest sufficient model wins.
- **`requires_tooling` ≠ `requires_mcp`** (Pillbox commit `5e2448c`): plain tool
  use must **not** pin a request to an MCP-native model; only genuine agentic
  orchestration filters on `mcp_native`.
- **`Message.Content` accepts string *or* array of blocks** — a real bug from
  [cc_bridge](https://github.com/andyeswong/cc_bridge) (2026-06-17): OpenClaw's
  WhatsApp channel sends content as an array, Telegram as a string.

## Stack

Go 1.25 · Gin · Gorm · SQLite (pure-Go `glebarez/sqlite`, builds with
`CGO_ENABLED=0` into a single static binary, same as cc_bridge).

## Build & run

```sh
CGO_ENABLED=0 go build -o freerouter .
cp freerouter.config.example.json freerouter.config.json   # optional; runs on defaults otherwise
./freerouter                                                # FRGO_CONFIG_PATH overrides config location
```

## API

OpenAI-compatible surface + a small admin API for the vademécum.

| Method | Path                       | Purpose                                            |
|--------|----------------------------|----------------------------------------------------|
| GET    | `/health`                  | liveness                                           |
| GET    | `/v1/models`               | list routable models (OpenAI shape)                |
| POST   | `/v1/chat/completions`     | classify → pick model → proxy (streaming passes through) |
| GET    | `/admin/models`            | list vademécum                                     |
| POST   | `/admin/models`            | add a model                                        |
| PUT    | `/admin/models/:id`        | update a model                                     |
| DELETE | `/admin/models/:id`        | remove a model                                     |
| POST   | `/admin/models/:id/scan`   | probe endpoint, record `health`                    |

Every `/v1/chat/completions` response carries the routing decision in headers:
`X-FreeRouter-Model`, `X-FreeRouter-Tier`, `X-FreeRouter-Savings`.

### Caller hints (data-driven path)

The request body may declare routing intent, which overrides the classifier:

```json
{ "model": "auto", "tier": 2, "requires_mcp": false,
  "messages": [{ "role": "user", "content": "..." }] }
```

Or use a prompt-prefix mode override: `/simple`, `/medium`, `/complex`,
`/reason`, `/max` (also bracket form, e.g. `[simple]`).

## License

MIT, matching the upstream FreeRouter / ClawRouter lineage.
