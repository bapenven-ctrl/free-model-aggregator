# Free Model Aggregator

One server that serves the OpenAI Chat Completions API and the Anthropic Messages API — each in JSON
and in server-sent events — over the free tiers of the providers a make.fun trial brief names.

For every request the server ranks every free model that meets the request's requirements by the
brief's quality table and tries the best first, translating between the client's wire and the
provider's wire (OpenAI, Anthropic or Gemini) in both directions. A provider that refuses hands
the request to the next candidate; a refusal whose own message names a spent budget or an outage
retires that provider for the rest of the trial, and every other refusal changes nothing, because a
provider's scheduled refusals are one call each. Every choice is a function of the brief, the
requests and the providers' answers alone — no clock, no randomness — so a replay of a trial walks
the same candidates in the same order and leaves the same state.

## Run

```
node main.ts            # serves, once MAKEFUN_ENV_URL and MAKEFUN_PROMPT_FILE are set
node main.ts --list     # prints the catalog document it registers, and exits
```

Node 22's standard library alone; TypeScript runs directly, nothing to install.

## Layout

- `main.ts` — the brief, the catalog, routing and failover, the routes, register and the work poll.
- `lib/providers.ts` — the brief's parser and the catalog probe in each provider's own wire.
- `lib/wire.ts` — the request and answer translations between the three wires, JSON and SSE.

MIT licensed.
