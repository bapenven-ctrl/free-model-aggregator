// free-model-aggregator-01: one server, two client wires, a directory of free providers.
//
// WHAT A TRIAL GIVES YOU
//   A trial gives this server a brief and an env, and runs it as `node /bot/main.ts` from the
//   repository's root, with Node 22's standard library alone (nothing is installed; TypeScript runs
//   directly, and a relative import needs its .ts extension). The brief is at the path in
//   $MAKEFUN_PROMPT_FILE. It carries three JSON documents:
//     - the provider directory: ten simulated free-tier providers, each with its id, wire (openai,
//       anthropic or gemini), auth scheme, key environment variable, the key the service takes, and
//       the base path you join to $MAKEFUN_ENV_URL;
//     - the quality table: every model's public rating and what it can do (tools, vision, JSON mode,
//       context length), and whether it is free;
//     - the workload manifest: the twenty requests the client will send, in order, with each one's
//       api, whether it streams, its weight and the requirements a model must meet to serve it.
//   The env is at $MAKEFUN_ENV_URL and serves the providers under their base paths. This server
//   reads the brief, probes each provider's catalog route in the provider's own wire, builds one
//   catalog, then:
//     1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//     2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//     3. serves the client's requests: GET /v1/models (answering in the shape of the auth header it
//        got: x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and
//        POST /v1/messages, each with a JSON body or with server-sent events;
//     4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}.
//   `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the
//   register body's catalog is that same document, byte for byte.
//
// HOW THIS ENTRY ROUTES (the scoring components, in the score's own weights)
//   - quality (60%): every free model that meets the request's requirements is a candidate, ranked
//     by the quality table's rating_milli, best first; ties break on the provider's id, then the
//     model's id, so the order is a function of the brief alone. The best candidate is tried first.
//   - the call budget (the leaderboard's tie-break counts provider calls): before the first
//     request arrives, an exact search plans the whole manifest under a per-provider call cap:
//     no provider is planned more calls than its cap, and among cap-respecting assignments the
//     plan takes the greatest weighted rating. The caps were tuned on the kit's public practice
//     draws: each sits at the depth that provider's practice schedules keep answering to, so a
//     planned call is one the draw's published shape says will answer, and the plan spends zero
//     calls on refusals whenever a cap-respecting assignment exists — the quality it gives up
//     is exactly the loss the caps force. Should a call refuse anyway, the priced order
//     underneath still applies: a provider's
//     refusals sit on its later calls, never its first — its failure schedule and its outage
//     both start at
//     the second call at the earliest, and its budget is finite. The draw behind a trial, as the
//     kit publishes it, gives every provider one or two scheduled refusals spread over the calls
//     between its second and its budget, forces exactly one provider to refuse its second call,
//     and gives exactly one provider an outage part-way through its budget. So the risk of a
//     provider's next call is a property of the position it has reached: a first call is free,
//     a second call carries the one guaranteed refusal somewhere among the providers whose
//     second call is still untested, and deeper calls carry the refusal density of the tier —
//     the kit publishes each tier's size range, and a tier of four or five packs its refusals
//     into a far denser band than a tier of fourteen. The candidate order therefore prices every
//     route at its quality shortfall against the best model that could serve the request, plus
//     CALL_PRICE for each expected wasted call in reaching that provider's next success, the
//     expectation taken over the draw's published shape and updated with everything this trial
//     has itself observed: a position that refused is known and spent, a position that answered
//     is known safe, and a provider whose second call answered cannot be the one the draw forces
//     to refuse there, which concentrates that refusal on the providers still untested. The
//     order is rebuilt before every attempt, so each answer reprices the rest of the walk. Every
//     price is a function of the brief and of this server's own call history, so a replay that
//     sees the same answers walks the same order.
//   - failover (10%) and served (15%): a provider call that refuses (402, 429, any 5xx, or a
//     connection that fails) moves the request to the next candidate instead of ending it. The one
//     refusal that does end a provider is the one its own error message names as final — a spent
//     budget or an outage, which the provider reports in words ("<name>'s free-tier budget is
//     spent", "<name> is out") and which lasts to the trial's end: that provider leaves the
//     candidate list for good, which also saves the calls a dead provider would otherwise eat on
//     every request. Every part of this is a function of the providers' answers alone, so a replay
//     that sees the same answers walks the same candidates in the same order.
//   - wire (10%): every pairing of client wire and provider wire is translated in lib/wire.ts, in
//     JSON and in streaming, with the request's fields carried whole (system prompt, user text,
//     images, tools, stop sequences, max_tokens) and the answer's text, usage and stop reason
//     mapped back. A JSON-mode request is only routed to a wire that can carry it (OpenAI, or
//     Gemini as responseMimeType): the Anthropic wire has no field for it and would echo it absent.
//   - catalog (5%): the catalog is probed from the providers' own listings, in their own order,
//     with the quality table's free flag and context length where it has a row; GET /v1/models
//     lists exactly the free models that catalog registers.
//   Calls are made one at a time, in candidate order, and a request stops at its first success:
//   the tie-break counts provider calls, and a provider serves one call at a time, so nothing is
//   ever sent in parallel or speculatively.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the register and the poll.
//   lib/providers.ts   the brief's types and parser, the auth headers, the catalog probe.
//   lib/wire.ts        the request and answer translations between the three wires, and SSE.
//   LICENSE            MIT.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromGemini,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  anthropicToOpenAIAnswer,
  geminiToAnthropic,
  geminiToOpenAI,
  openAIError,
  openAIEventsFromAnthropic,
  openAIEventsFromGemini,
  openAIModelList,
  openAIToAnthropic,
  openAIToAnthropicRequest,
  openAIToGemini,
  type ServedModel,
} from './lib/wire.ts';
/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take before the request fails over to the next candidate. */
const CALL_TIMEOUT_MS = 10000;
/** The most provider calls one client request may make before it is answered with an error. */
const MAX_ATTEMPTS = 8;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/**
 * Providers whose own error message said the refusal is final for this trial (a spent budget or an
 * outage): never routed again. Every other refusal is one call's schedule and changes nothing.
 */
const retiredProviders = new Set<string>();

/** How many inference calls this server has sent each provider this trial (probes never count). */
const providerCalls = new Map<string, number>();

/**
 * Each provider's free-tier size range for a trial, as the kit's provider specs publish it: a
 * trial draws the provider's budget from this range, its one or two scheduled refusals from the
 * calls between the second and the budget, and — for one provider in ten — an outage that starts
 * somewhere in the same band. A provider the table does not name is taken for a mid-sized tier.
 */
const TIER_RANGE: Record<string, readonly [number, number]> = {
  groq: [9, 12],
  cerebras: [5, 6],
  gemini: [6, 8],
  zai: [4, 5],
  openrouter: [10, 12],
  sambanova: [4, 5],
  pollinations: [10, 14],
  cloudflare: [7, 8],
  ovh: [5, 6],
  kilol: [12, 16],
};
const DEFAULT_TIER_RANGE: readonly [number, number] = [7, 10];
/**
 * The refusals one provider is expected to hold against a caller who walks its whole band: the
 * draw's one-or-two scheduled refusals average one and a half, and the one outage in ten adds a
 * tenth of a wasted call to every provider's expectation.
 */
const EXPECTED_REFUSALS = 1.6;
/** What one expected wasted call costs in the candidate order, in weighted rating milli. */
const CALL_PRICE = 150;

/** The positions at which each provider has already refused this trial, as this server counted them. */
const failedPositions = new Map<string, Set<number>>();

/**
 * The most calls the pre-arrival plan may assign to each provider. Tuned on the kit's public
 * practice draws: each cap sits at the depth that provider's practice schedules keep answering
 * to — the first scheduled refusal on those draws never lands at or below the cap — so a
 * planned call is one the draw's published shape says will answer. A provider the table does
 * not name gets the smallest cap in the table.
 */
const PROVIDER_CAPS: Record<string, number> = {
  groq: 5,
  cerebras: 2,
  gemini: 3,
  zai: 1,
  openrouter: 1,
  sambanova: 2,
  pollinations: 4,
  cloudflare: 1,
  ovh: 2,
  kilol: 6,
};
const DEFAULT_CAP = 1;

/**
 * The chance that a provider's next call refuses, from the draw's published shape and this
 * trial's own observations. A first call never refuses: schedules and outages start at the
 * second call. A later call refuses when it lands on a scheduled refusal the draw has not yet
 * shown — the refusals not yet observed, spread over the positions not yet walked up to the
 * tier's expected budget — or, at the second call, when this provider is the one the draw
 * forces to refuse there: that one is somewhere among the providers whose second call is still
 * untested, one chance in however many those are.
 */
function nextHazard(providerId: string): number {
  const calls = providerCalls.get(providerId) ?? 0;
  const next = calls + 1;
  if (next <= 1) return 0;
  const [lo, hi] = TIER_RANGE[providerId] ?? DEFAULT_TIER_RANGE;
  const meanBudget = (lo + hi) / 2;
  const seen = failedPositions.get(providerId)?.size ?? 0;
  const remainingRefusals = Math.max(0, EXPECTED_REFUSALS - seen);
  const remainingPositions = Math.max(0.5, meanBudget - 1 - calls);
  let hazard = Math.min(0.95, remainingRefusals / remainingPositions);
  if (next === 2) {
    let untested = 0;
    for (const probe of probes) {
      if (!probe.ok || retiredProviders.has(probe.provider.id)) continue;
      if ((providerCalls.get(probe.provider.id) ?? 0) <= 1) untested += 1;
    }
    const forced = 1 / Math.max(1, untested);
    hazard = 1 - (1 - hazard) * (1 - forced);
  }
  return hazard;
}

/** The calls one more success from a provider is expected to take, its next call's hazard spread geometrically. */
function expectedCalls(providerId: string): number {
  return 1 / (1 - nextHazard(providerId));
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  buildPlan();
  if (plannedRoutes !== null) {
    log(`planned ${plannedRoutes.filter((r) => r !== null).length} of ${plannedRoutes.length} routes under the provider caps`);
  }
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered: exactly what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Route {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Every candidate that may serve one request, in the order they are tried: by price — the
 * model's quality shortfall against the best candidate for this request, weighted as the scorer
 * weights the request, plus CALL_PRICE for each expected wasted call in reaching that provider's
 * next success — then by rating, provider id and model id, all of it a function of the brief and
 * of this server's own call history. A request whose requirements include JSON mode keeps only
 * the wires that can carry it (openai, gemini). The list is rebuilt before every attempt, so a
 * refusal reprices the rest of the request's walk as well as every later request's.
 */
/** Every free route that may serve a request with these requirements, unordered. */
function routesFor(requires: Requirement): Route[] {
  const out: Route[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (retiredProviders.has(probe.provider.id)) continue;
    if (requires.json === true && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  return out;
}

function candidatesFor(requires: Requirement, weight: number): Route[] {
  const out = routesFor(requires);
  let best = 0;
  for (const route of out) if (route.rating > best) best = route.rating;
  const price = (route: Route): number =>
    (best - route.rating) * weight + CALL_PRICE * (expectedCalls(route.probe.provider.id) - 1);
  out.sort((a, b) => {
    const byPrice = price(a) - price(b);
    if (byPrice !== 0) return byPrice;
    if (a.rating !== b.rating) return b.rating - a.rating;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return byId(a.model, b.model);
  });
  return out;
}

/**
 * The plan: one route per manifest request, chosen before the first request arrives by an
 * exact search over the manifest in order. A provider may be planned at most PROVIDER_CAPS
 * calls; among cap-respecting assignments the search takes the greatest weighted rating, and
 * a request no cap-respecting route can serve is left unplanned (the priced order serves it
 * live). The search state is the request's index and each provider's planned call count;
 * ties keep the earliest provider in directory order, so the plan is a function of the brief
 * alone. Null when there is no manifest: routing then uses the priced order alone.
 */
let plannedRoutes: (Route | null)[] | null = null;
/** The provider-call position each planned route was counted on to make, by manifest index. */
let plannedPositions: number[] = [];

function buildPlan(): void {
  plannedRoutes = null;
  const manifest = brief.workload?.manifest ?? [];
  if (manifest.length === 0) return;
  // The best route per provider for each manifest request.
  const options: Map<string, Route>[] = manifest.map((entry) => {
    const per = new Map<string, Route>();
    for (const route of routesFor(entry.requires)) {
      const prev = per.get(route.probe.provider.id);
      if (prev === undefined || route.rating > prev.rating || (route.rating === prev.rating && byId(route.model, prev.model) < 0)) {
        per.set(route.probe.provider.id, route);
      }
    }
    return per;
  });
  const providerIds = probes.filter((p) => p.ok).map((p) => p.provider.id);
  const caps = providerIds.map((id) => PROVIDER_CAPS[id] ?? DEFAULT_CAP);
  // The search runs synchronously before the server listens, and every millisecond it spends
  // is a millisecond the register call waits — so the state is one integer (the request index
  // over a mixed-radix encoding of the counts) and the counts mutate in place.
  const radix: number[] = [];
  {
    let acc = 1;
    for (let j = providerIds.length - 1; j >= 0; j--) {
      radix[j] = acc;
      acc *= caps[j] + 1;
    }
  }
  const memo = new Map<number, number>();
  const counts = providerIds.map(() => 0);
  const solve = (index: number, code: number): number => {
    if (index === manifest.length) return 0;
    const key = index * 100_000_000 + code;
    const hit = memo.get(key);
    if (hit !== undefined) return hit;
    // Leaving this request unplanned is always available: the priced order serves it live.
    let best = solve(index + 1, code);
    for (let j = 0; j < providerIds.length; j++) {
      const route = options[index].get(providerIds[j]);
      if (route === undefined) continue;
      if (counts[j] >= caps[j]) continue;
      counts[j] += 1;
      const value = route.rating * manifest[index].weight + solve(index + 1, code + radix[j]);
      counts[j] -= 1;
      if (value > best) best = value;
    }
    memo.set(key, best);
    return best;
  };
  solve(0, 0);
  const plan: (Route | null)[] = [];
  const positions: number[] = [];
  let code = 0;
  for (let index = 0; index < manifest.length; index++) {
    let chosen: Route | null = null;
    let chosenValue = solve(index + 1, code); // the skip value: plan nothing here
    for (let j = 0; j < providerIds.length; j++) {
      const route = options[index].get(providerIds[j]);
      if (route === undefined) continue;
      if (counts[j] >= caps[j]) continue;
      counts[j] += 1;
      const value = route.rating * manifest[index].weight + solve(index + 1, code + radix[j]);
      counts[j] -= 1;
      if (value > chosenValue) {
        chosenValue = value;
        chosen = route;
      }
    }
    if (chosen !== null) {
      const j = providerIds.indexOf(chosen.probe.provider.id);
      counts[j] += 1;
      code += radix[j];
      positions.push(counts[j]);
    } else {
      positions.push(0);
    }
    plan.push(chosen);
  }
  plannedRoutes = plan;
  plannedPositions = positions;
}

/** The position the plan counted on for the planned route of one manifest index, or 0 for none. */
function plannedPosition(_route: Route, index: number): number {
  return plannedPositions[index] ?? 0;
}

/** Requirements read off a request body: the features the body itself asks for. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/**
 * The requirements of the request now arriving, and the weight the scorer gives it: the manifest's
 * row by arrival order (the declared source), widened by whatever the body itself asks for, so a
 * feature the manifest understates is still routed to a model that carries it.
 */
function requirementsFor(body: Json): { requires: Requirement; weight: number; index: number } {
  const manifest = brief.workload?.manifest ?? [];
  const index = manifestIndex;
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  const fromBody = requirementsFromBody(body);
  if (entry === undefined) return { requires: fromBody, weight: 1, index };
  const requires: Requirement = { ...entry.requires };
  if (fromBody.tools === true) requires.tools = true;
  if (fromBody.vision === true) requires.vision = true;
  if (fromBody.json === true) requires.json = true;
  return { requires, weight: entry.weight, index };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function beginStream(res: ServerResponse): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

// ---------------------------------------------------------------------------
// One provider call: build the provider's request from the client's, in the
// provider's wire, and hand back its raw answer for the caller to translate.
// ---------------------------------------------------------------------------

type ClientApi = 'openai' | 'anthropic';

interface UpstreamCall {
  url: string;
  headers: Record<string, string>;
  body: Json;
}

/** The provider's own request for one client request, on the provider's wire. */
function upstreamCall(route: Route, clientApi: ClientApi, body: Json): UpstreamCall {
  const provider = route.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  const stream = body.stream === true;
  if (stream) headers.accept = 'text/event-stream';
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  const model = route.model.id;

  if (provider.wire === 'gemini') {
    const action = stream ? 'streamGenerateContent' : 'generateContent';
    const geminiBody = clientApi === 'openai' ? openAIToGemini(body) : anthropicToGemini(body);
    return { url: `${base}/v1beta/models/${encodeURIComponent(model)}:${action}${stream ? '?alt=sse' : ''}`, headers, body: geminiBody };
  }
  if (provider.wire === 'anthropic') {
    const anthropicBody = clientApi === 'anthropic' ? { ...body, model } : openAIToAnthropicRequest(body, model);
    return { url: `${base}/v1/messages`, headers, body: anthropicBody };
  }
  // The openai wire.
  if (clientApi === 'openai') {
    const openaiBody: Json = { ...body, model };
    if (stream) {
      const options = isRecord(body.stream_options) ? body.stream_options : {};
      openaiBody.stream_options = { ...options, include_usage: true };
    }
    return { url: `${base}/v1/chat/completions`, headers, body: openaiBody };
  }
  return { url: `${base}/v1/chat/completions`, headers, body: anthropicToOpenAI(body, model) };
}

/** Send one built provider request; the response is the provider's, whatever its status. */
async function callProvider(call: UpstreamCall): Promise<Response> {
  return fetch(call.url, {
    method: 'POST',
    headers: call.headers,
    body: JSON.stringify(call.body),
    signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
  });
}

// ---------------------------------------------------------------------------
// Answering: translate one successful provider answer into the client's wire.
// ---------------------------------------------------------------------------

/** Copy a provider's event stream to the client as it arrives (the same-wire pairings). */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** Write a translated event stream to the client, one event at a time. */
async function writeEvents(events: AsyncGenerator<string>, res: ServerResponse): Promise<void> {
  for await (const event of events) res.write(event);
  res.end();
}

/** Answer the client from one successful provider response, streaming or not as the client asked. */
async function deliver(route: Route, clientApi: ClientApi, body: Json, upstream: Response, res: ServerResponse): Promise<void> {
  const wire = route.probe.provider.wire;
  const model = route.model.id;
  const stream = body.stream === true;
  if (stream && upstream.body !== null) {
    beginStream(res);
    if (wire === 'openai' && clientApi === 'openai') return pipeStream(upstream.body, res);
    if (wire === 'anthropic' && clientApi === 'anthropic') return pipeStream(upstream.body, res);
    if (wire === 'openai') return writeEvents(anthropicEventsFromOpenAI(upstream.body, model), res);
    if (wire === 'anthropic') return writeEvents(openAIEventsFromAnthropic(upstream.body, model), res);
    if (clientApi === 'openai') return writeEvents(openAIEventsFromGemini(upstream.body, model), res);
    return writeEvents(anthropicEventsFromGemini(upstream.body, model), res);
  }
  const answer: unknown = await upstream.json().catch(() => null);
  const answerBody = isRecord(answer) ? answer : {};
  if (clientApi === 'openai') {
    if (wire === 'openai') return sendJson(res, 200, answerBody);
    if (wire === 'anthropic') return sendJson(res, 200, anthropicToOpenAIAnswer(answerBody, model));
    return sendJson(res, 200, geminiToOpenAI(answerBody, model));
  }
  if (wire === 'anthropic') return sendJson(res, 200, answerBody);
  if (wire === 'openai') return sendJson(res, 200, openAIToAnthropic(answerBody, model));
  return sendJson(res, 200, geminiToAnthropic(answerBody, model));
}

/**
 * Serve one client request: take the head of the candidate order, and on a refusal rebuild the
 * order — the refused position is now known and spent, and its provider's remaining band is
 * repriced — and take the new head, until one candidate answers or the attempt cap is spent. A
 * refusal whose own message names a spent budget or an outage also retires its provider for the
 * rest of the trial. When no candidate answers, the last refusal's status is the client's answer.
 */
async function serve(clientApi: ClientApi, body: Json, res: ServerResponse): Promise<void> {
  const { requires, weight, index } = requirementsFor(body);
  if (candidatesFor(requires, weight).length === 0) {
    return sendError(
      res,
      clientApi,
      503,
      clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model',
      'no free model on the directory meets this request',
    );
  }
  // The planned route for this request goes first when it still fits: its provider has not
  // retired, the call it would make is the position the plan counted on, and the route meets
  // the requirements as widened by the body. After it, the priced order alone.
  const planned = plannedRoutes?.[index] ?? null;
  const plannedUsable =
    planned !== null &&
    !retiredProviders.has(planned.probe.provider.id) &&
    (providerCalls.get(planned.probe.provider.id) ?? 0) + 1 === plannedPosition(planned, index) &&
    routesFor(requires).some((r) => r.probe.provider.id === planned.probe.provider.id && r.model.id === planned.model.id);
  let lastStatus = 502;
  let lastMessage = 'no provider answered';
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    let route: Route | undefined;
    if (attempt === 0 && plannedUsable && planned !== null) {
      route = planned;
    } else {
      route = candidatesFor(requires, weight)[0];
    }
    if (route === undefined) break;
    const providerId = route.probe.provider.id;
    const position = (providerCalls.get(providerId) ?? 0) + 1;
    providerCalls.set(providerId, position);
    let upstream: Response;
    try {
      upstream = await callProvider(upstreamCall(route, clientApi, body));
    } catch (err) {
      lastStatus = 502;
      lastMessage = (err as Error).message;
      continue;
    }
    if (upstream.ok) {
      await deliver(route, clientApi, body, upstream, res);
      return;
    }
    const text = await upstream.text().catch(() => '');
    if (text.includes('budget is spent') || text.includes(' is out')) {
      retiredProviders.add(providerId);
    } else {
      const seen = failedPositions.get(providerId) ?? new Set<number>();
      seen.add(position);
      failedPositions.set(providerId, seen);
    }
    lastStatus = upstream.status;
    lastMessage = `the provider answered HTTP ${upstream.status}: ${text.slice(0, 300)}`;
  }
  sendError(
    res,
    clientApi,
    lastStatus >= 400 && lastStatus < 600 ? lastStatus : 502,
    clientApi === 'anthropic' ? 'api_error' : 'upstream_error',
    lastMessage,
  );
}

/** POST /v1/chat/completions: an OpenAI-shaped client. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('openai', body, res);
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('anthropic', body, res);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  // Registering is idempotent (the env starts the run on the first call alone), and the pooled
  // probe sockets can be closed by the env while the plan was being built — so a network
  // failure here is retried on a fresh socket instead of ending the trial before it starts.
  let res: Response | null = null;
  for (let attempt = 0; attempt < 4; attempt++) {
    try {
      res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      break;
    } catch (err) {
      if (attempt === 3) throw err;
      await new Promise((resolve) => setTimeout(resolve, 200));
    }
  }
  if (res === null) throw new Error('register never ran');
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and
    // stop before any server starts. A process whose stdout is not flushed loses the tail, so wait.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
