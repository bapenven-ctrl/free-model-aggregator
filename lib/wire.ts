// The client wires and the three upstream wires.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic
// one (POST /v1/messages), with a JSON body or with server-sent events. The entry answers from one of
// the brief's providers, which may speak the OpenAI, Anthropic or Gemini wire, so every pairing of
// client wire and provider wire is translated here:
//
//   requests:  openai -> openai (pass-through), openai -> anthropic, openai -> gemini,
//              anthropic -> openai, anthropic -> anthropic (pass-through in main.ts), anthropic -> gemini
//   answers:   each provider's JSON or SSE is rewritten into the client's wire, with the provider's
//              text whole, its usage counts as that wire reports them, and its stop reason mapped
//              (end_turn <-> stop, max_tokens <-> length, tool_use <-> tool_calls,
//              refusal <-> content_filter, STOP / MAX_TOKENS / SAFETY on the Gemini wire).
//
// Translations preserve what a provider echoes back: the system prompt's text, the last user message's
// text, the tool names in order, max_tokens, the stop sequences and any image (a base64 image keeps
// its bytes across wires, so its digest survives; a URL image keeps its URL). A JSON-mode request is
// only routed to a wire that can carry it (OpenAI natively, Gemini as responseMimeType), because the
// Anthropic wire has no field for it and would echo it absent.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body
// and the same provider answer give the same bytes on a replay. The two model listings use fixed
// timestamps for the same reason: a `created` that moved between runs would be a different document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** A tool call's arguments as a JSON string: the object an Anthropic or Gemini call carries, stringified. */
function argumentsString(value: unknown): string {
  if (typeof value === 'string') return value;
  return JSON.stringify(isRecord(value) || Array.isArray(value) ? value : {});
}

// ---------------------------------------------------------------------------
// Stop reasons, one family per wire.
// ---------------------------------------------------------------------------

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Anthropic's stop_reason as OpenAI's finish_reason. */
export function openAIFinishOf(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as OpenAI's finish_reason. */
export function openAIFinishOfGemini(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'length';
  if (finish === 'SAFETY' || finish === 'RECITATION' || finish === 'PROHIBITED_CONTENT') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as Anthropic's stop_reason. */
export function anthropicStopOfGemini(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'max_tokens';
  if (finish === 'SAFETY' || finish === 'RECITATION' || finish === 'PROHIBITED_CONTENT') return 'refusal';
  return 'end_turn';
}

// ---------------------------------------------------------------------------
// Client Anthropic -> provider OpenAI (request), provider OpenAI -> client
// Anthropic (answer): the pair the house baseline already spoke.
// ---------------------------------------------------------------------------

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

// ---------------------------------------------------------------------------
// Client OpenAI -> provider Anthropic, and the answer back.
// ---------------------------------------------------------------------------

/** An OpenAI image_url part as an Anthropic image block: a data URL keeps its bytes, a URL its address. */
function anthropicImageOf(part: Json): Json | null {
  const image = isRecord(part.image_url) ? part.image_url : null;
  const url = image !== null && typeof image.url === 'string' ? image.url : '';
  if (url === '') return null;
  const data = /^data:([^;,]+)?(;base64)?,(.*)$/s.exec(url);
  if (data !== null && data[2] === ';base64') {
    return { type: 'image', source: { type: 'base64', media_type: data[1] ?? 'image/png', data: data[3] } };
  }
  if (data !== null) return null;
  return { type: 'image', source: { type: 'url', url } };
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that names `model`: the system
 * messages join into the system prompt, image_url parts become image blocks (base64 data URLs keep
 * their bytes), tool_calls become tool_use blocks, tool messages become tool_result blocks, and the
 * output cap (max_tokens or max_completion_tokens), temperature, top_p, stop sequences, tools,
 * tool_choice and streaming pass through in the Messages spellings. response_format has no Messages
 * spelling and is never carried: routing keeps JSON-mode requests off this wire.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool' || raw.role === 'function') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content),
          },
        ],
      });
      continue;
    }
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string' || content === null || content === undefined) {
      if (assistant) {
        const blocks: Json[] = [];
        if (typeof content === 'string' && content !== '') blocks.push({ type: 'text', text: content });
        const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
        for (const call of calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          blocks.push({
            type: 'tool_use',
            id: typeof call.id === 'string' ? call.id : '',
            name: typeof call.function.name === 'string' ? call.function.name : '',
            input: argumentsOf(call.function.arguments),
          });
        }
        messages.push({ role: 'assistant', content: blocks.length > 0 ? blocks : '' });
      } else if (typeof content === 'string') {
        messages.push({ role: 'user', content });
      }
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
      else if (part.type === 'image_url') {
        const image = anthropicImageOf(part);
        if (image !== null) blocks.push(image);
      }
    }
    if (blocks.length > 0) messages.push({ role: assistant ? 'assistant' : 'user', content: blocks });
  }

  const out: Json = { model, messages };
  const system = systemParts.join('\n');
  if (system !== '') out.system = system;
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null;
  if (maxTokens !== null) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = typeof body.stop === 'string' ? [body.stop] : Array.isArray(body.stop) ? body.stop.filter((s): s is string => typeof s === 'string') : [];
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && isRecord(choice.function) && typeof choice.function.name === 'string') out.tool_choice = { type: 'tool', name: choice.function.name };
  if (body.stream === true) out.stream = true;
  return out;
}

/** An Anthropic Messages answer as a Chat Completions object. */
export function anthropicToOpenAIAnswer(body: Json, model: string): Json {
  const blocks = Array.isArray(body.content) ? body.content : [];
  const content: Json[] = [];
  let text = '';
  const toolCalls: Json[] = [];
  for (const block of blocks) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text += block.text;
    else if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : `toolu_${toolCalls.length}`,
        type: 'function',
        function: { name: typeof block.name === 'string' ? block.name : '', arguments: argumentsString(block.input) },
      });
    }
  }
  if (text !== '') content.push({ type: 'text', text });
  const message: Json = {
    role: 'assistant',
    content: text !== '' ? text : null,
    ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
  };
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: typeof body.id === 'string' ? body.id : 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: openAIFinishOf(body.stop_reason), logprobs: null }],
    usage: {
      prompt_tokens: numberOrZero(usage.input_tokens),
      completion_tokens: numberOrZero(usage.output_tokens),
      total_tokens: numberOrZero(usage.input_tokens) + numberOrZero(usage.output_tokens),
    },
  };
}

// ---------------------------------------------------------------------------
// The Gemini wire: requests from either client wire, answers to either.
// ---------------------------------------------------------------------------

/** An OpenAI image_url part as a Gemini part: a data URL becomes inlineData, a URL becomes fileData. */
function geminiImageOfOpenAI(part: Json): Json | null {
  const image = isRecord(part.image_url) ? part.image_url : null;
  const url = image !== null && typeof image.url === 'string' ? image.url : '';
  if (url === '') return null;
  const data = /^data:([^;,]+)?(;base64)?,(.*)$/s.exec(url);
  if (data !== null && data[2] === ';base64') return { inlineData: { mimeType: data[1] ?? 'image/png', data: data[3] } };
  if (data !== null) return null;
  return { fileData: { fileUri: url } };
}

/** An Anthropic image block as a Gemini part, with the same byte/URL preservation. */
function geminiImageOfAnthropic(block: Json): Json | null {
  if (!isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.data === 'string') {
    return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
  }
  if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  return null;
}

/**
 * The tail guard for a Gemini request: the provider reads the user text from the last content that is
 * not the model's, so when a tool-result content would land last and shadow the user's text, the
 * user's own last text is repeated as a final user content, where the provider reads it from.
 */
function geminiContents(contents: Json[], userText: string): Json[] {
  if (userText === '') return contents;
  let lastText = '';
  for (const content of contents) {
    if (content.role === 'model') continue;
    const parts = Array.isArray(content.parts) ? content.parts : [];
    const texts = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : []));
    lastText = texts.join('\n');
  }
  if (lastText === userText) return contents;
  return [...contents, { role: 'user', parts: [{ text: userText }] }];
}

/** The last user text of an OpenAI request, as the OpenAI wire itself reads it (string or text parts). */
function openAIUserText(body: Json): string {
  let text = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw) || raw.role !== 'user') continue;
    text = textOf(raw.content);
  }
  return text;
}

/** The last user text of an Anthropic request, as the Messages wire itself reads it. */
function anthropicUserText(body: Json): string {
  let text = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw) || raw.role !== 'user') continue;
    text = textOf(raw.content);
  }
  return text;
}

/** A Chat Completions request as a Gemini generateContent request (the model travels in the URL). */
export function openAIToGemini(body: Json): Json {
  const systemParts: string[] = [];
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool' || raw.role === 'function') {
      contents.push({
        role: 'user',
        parts: [{ functionResponse: { name: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', response: { result: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content) } } }],
      });
      continue;
    }
    const role = raw.role === 'assistant' ? 'model' : 'user';
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
        else if (part.type === 'image_url') {
          const image = geminiImageOfOpenAI(part);
          if (image !== null) parts.push(image);
        }
      }
    }
    if (raw.role === 'assistant' && Array.isArray(raw.tool_calls)) {
      for (const call of raw.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
      }
    }
    if (parts.length > 0) contents.push({ role, parts });
  }

  const out: Json = { contents: geminiContents(contents, openAIUserText(body)) };
  const system = systemParts.join('\n');
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const config: Json = {};
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null;
  if (maxTokens !== null) config.maxOutputTokens = maxTokens;
  const stops = typeof body.stop === 'string' ? [body.stop] : Array.isArray(body.stop) ? body.stop.filter((s): s is string => typeof s === 'string') : [];
  if (stops.length > 0) config.stopSequences = stops;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
    config.responseMimeType = 'application/json';
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  return out;
}

/** A Messages request as a Gemini generateContent request (the model travels in the URL). */
export function anthropicToGemini(body: Json): Json {
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role === 'assistant' ? 'model' : 'user';
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image') {
          const image = geminiImageOfAnthropic(block);
          if (image !== null) parts.push(image);
        } else if (block.type === 'tool_use') {
          parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
        } else if (block.type === 'tool_result') {
          parts.push({
            functionResponse: {
              name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
              response: { result: toolResultText(block.content) },
            },
          });
        }
      }
    }
    if (parts.length > 0) contents.push({ role, parts });
  }

  const out: Json = { contents: geminiContents(contents, anthropicUserText(body)) };
  const system = textOf(body.system);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) config.stopSequences = body.stop_sequences;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  return out;
}

/** The text and tool calls of one Gemini answer body, in either client's spelling needs. */
function geminiAnswerParts(body: Json): { text: string; calls: { name: string; args: unknown }[]; finish: unknown; input: number; output: number } {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  const calls: { name: string; args: unknown }[] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') text += part.text;
    else if (isRecord(part.functionCall)) {
      calls.push({ name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '', args: part.functionCall.args });
    }
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    calls,
    finish: candidate.finishReason,
    input: numberOrZero(usage.promptTokenCount),
    output: numberOrZero(usage.candidatesTokenCount),
  };
}

/** A Gemini generateContent answer as a Chat Completions object. */
export function geminiToOpenAI(body: Json, model: string): Json {
  const answer = geminiAnswerParts(body);
  const toolCalls = answer.calls.map((call, i) => ({
    id: `call_${i}`,
    type: 'function',
    function: { name: call.name, arguments: argumentsString(call.args) },
  }));
  return {
    id: 'chatcmpl-gemini',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: answer.text !== '' ? answer.text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) },
        finish_reason: openAIFinishOfGemini(answer.finish),
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object. */
export function geminiToAnthropic(body: Json, model: string): Json {
  const answer = geminiAnswerParts(body);
  const content: Json[] = [];
  if (answer.text !== '') content.push({ type: 'text', text: answer.text });
  answer.calls.forEach((call, i) => {
    content.push({ type: 'tool_use', id: `toolu_${i}`, name: call.name, input: argumentsOf(call.args) });
  });
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: 'msg_gemini',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStopOfGemini(answer.finish),
    stop_sequence: null,
    usage: { input_tokens: answer.input, output_tokens: answer.output },
  };
}

// ---------------------------------------------------------------------------
// Server-sent events: one frame reader, then the cross-wire translations.
// Same-wire streams are piped raw by main.ts and never reach these functions.
// ---------------------------------------------------------------------------

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** The frames of an SSE stream, one {event, data} at a time, in arrival order. */
export async function* sseFrames(upstream: ReadableStream<Uint8Array>): AsyncGenerator<{ event: string | null; data: string }> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const framesOf = (text: string): { event: string | null; data: string }[] => {
    const out: { event: string | null; data: string }[] = [];
    for (const frame of text.split('\n\n')) {
      if (frame.trim() === '') continue;
      let event: string | null = null;
      const data: string[] = [];
      for (const line of frame.split('\n')) {
        if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
        else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
      }
      if (data.length > 0) out.push({ event, data: data.join('\n') });
    }
    return out;
  };
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const parsed of framesOf(`${frame}\n\n`)) yield parsed;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const parsed of framesOf(buffer)) yield parsed;
  } finally {
    reader.releaseLock();
  }
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason
 * and usage, then message_stop. `model` is the model that served, named back to the client.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  for await (const frame of sseFrames(upstream)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') {
      yield* startEvents();
      yield* finishEvents(pendingStop ?? 'end_turn');
      continue;
    }
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    yield* startEvents();
    if (typeof delta.content === 'string' && delta.content !== '') {
      if (textIndex === -1) {
        textIndex = nextIndex;
        nextIndex += 1;
        yield sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } });
      }
      yield sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } });
    }
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
      let index = toolIndexes.get(upstreamIndex);
      if (index === undefined) {
        if (textIndex !== -1) {
          yield sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex });
          textIndex = -1;
        }
        index = nextIndex;
        nextIndex += 1;
        toolIndexes.set(upstreamIndex, index);
        const fn = isRecord(call.function) ? call.function : {};
        yield sseEvent('content_block_start', {
          type: 'content_block_start',
          index,
          content_block: {
            type: 'tool_use',
            id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
            name: typeof fn.name === 'string' ? fn.name : '',
            input: {},
          },
        });
      }
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.arguments === 'string' && fn.arguments !== '') {
        yield sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } });
      }
    }
    // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived:
    // OpenAI sends the usage after the finish_reason, and Anthropic's message_delta carries the
    // output tokens.
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
  }
  if (!finished) {
    yield* startEvents();
    yield* finishEvents(pendingStop ?? 'end_turn');
  }
}

/** An Anthropic Messages SSE stream as Chat Completions chunks, ending with the usage chunk and [DONE]. */
export async function* openAIEventsFromAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let roleSent = false;
  let finish: string | null = null;
  let inputTokens = 0;
  let outputTokens = 0;
  let toolSeq = 0;
  const blockTools = new Map<number, number>();
  const chunk = (delta: Json, finishReason: string | null): string =>
    sseEvent(null, {
      id: 'chatcmpl-stream',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [{ index: 0, delta, finish_reason: finishReason }],
    });
  const roleChunk = (): string[] => {
    if (roleSent) return [];
    roleSent = true;
    return [chunk({ role: 'assistant', content: '' }, null)];
  };

  for await (const frame of sseFrames(upstream)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message)) {
      if (isRecord(data.message.usage)) inputTokens = numberOrZero(data.message.usage.input_tokens);
      yield* roleChunk();
    } else if (frame.event === 'content_block_start' && isRecord(data.content_block) && data.content_block.type === 'tool_use') {
      yield* roleChunk();
      const blockIndex = typeof data.index === 'number' ? data.index : blockTools.size;
      const toolIndex = toolSeq;
      toolSeq += 1;
      blockTools.set(blockIndex, toolIndex);
      yield chunk(
        {
          tool_calls: [
            {
              index: toolIndex,
              id: typeof data.content_block.id === 'string' ? data.content_block.id : `toolu_${toolIndex}`,
              type: 'function',
              function: { name: typeof data.content_block.name === 'string' ? data.content_block.name : '', arguments: '' },
            },
          ],
        },
        null,
      );
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      yield* roleChunk();
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string' && data.delta.text !== '') {
        yield chunk({ content: data.delta.text }, null);
      } else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string' && data.delta.partial_json !== '') {
        const blockIndex = typeof data.index === 'number' ? data.index : 0;
        const toolIndex = blockTools.get(blockIndex) ?? 0;
        yield chunk({ tool_calls: [{ index: toolIndex, function: { arguments: data.delta.partial_json } }] }, null);
      }
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') finish = openAIFinishOf(data.delta.stop_reason);
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
    }
  }
  yield* roleChunk();
  yield chunk({}, finish ?? 'stop');
  yield sseEvent(null, {
    id: 'chatcmpl-stream',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [],
    usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens },
  });
  yield 'data: [DONE]\n\n';
}

/** One Gemini stream frame's text pieces, tool calls, finish reason and usage, as one parsed chunk. */
interface GeminiChunk {
  texts: string[];
  calls: { name: string; args: unknown }[];
  finish: unknown;
  input: number | null;
  output: number | null;
}

function geminiChunkOf(data: string): GeminiChunk | null {
  let body: unknown;
  try {
    body = JSON.parse(data);
  } catch {
    return null;
  }
  if (!isRecord(body)) return null;
  const answer = geminiAnswerParts(body);
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : null;
  return {
    texts: answer.text !== '' ? [answer.text] : [],
    calls: answer.calls,
    finish: answer.finish,
    input: usage !== null && typeof usage.promptTokenCount === 'number' ? usage.promptTokenCount : null,
    output: usage !== null && typeof usage.candidatesTokenCount === 'number' ? usage.candidatesTokenCount : null,
  };
}

/** A Gemini SSE stream as Chat Completions chunks, ending with the usage chunk and [DONE]. */
export async function* openAIEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let roleSent = false;
  let finish: string | null = null;
  let inputTokens = 0;
  let outputTokens = 0;
  let toolSeq = 0;
  const chunk = (delta: Json, finishReason: string | null): string =>
    sseEvent(null, {
      id: 'chatcmpl-stream',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [{ index: 0, delta, finish_reason: finishReason }],
    });
  const roleChunk = (): string[] => {
    if (roleSent) return [];
    roleSent = true;
    return [chunk({ role: 'assistant', content: '' }, null)];
  };

  for await (const frame of sseFrames(upstream)) {
    const parsed = geminiChunkOf(frame.data);
    if (parsed === null) continue;
    yield* roleChunk();
    for (const text of parsed.texts) yield chunk({ content: text }, null);
    for (const call of parsed.calls) {
      const toolIndex = toolSeq;
      toolSeq += 1;
      yield chunk(
        { tool_calls: [{ index: toolIndex, id: `call_${toolIndex}`, type: 'function', function: { name: call.name, arguments: argumentsString(call.args) } }] },
        null,
      );
    }
    if (parsed.finish !== null && parsed.finish !== undefined) finish = openAIFinishOfGemini(parsed.finish);
    if (parsed.input !== null) inputTokens = parsed.input;
    if (parsed.output !== null) outputTokens = parsed.output;
  }
  yield* roleChunk();
  yield chunk({}, finish ?? 'stop');
  yield sseEvent(null, {
    id: 'chatcmpl-stream',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [],
    usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens },
  });
  yield 'data: [DONE]\n\n';
}

/** A Gemini SSE stream as the Anthropic event sequence, ending with message_delta and message_stop. */
export async function* anthropicEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let textOpen = false;
  let nextIndex = 0;
  let textIndex = -1;
  let finish: string | null = null;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        },
      }),
    ];
  };

  for await (const frame of sseFrames(upstream)) {
    const parsed = geminiChunkOf(frame.data);
    if (parsed === null) continue;
    yield* startEvents();
    for (const text of parsed.texts) {
      if (!textOpen) {
        textIndex = nextIndex;
        nextIndex += 1;
        textOpen = true;
        yield sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } });
      }
      yield sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text } });
    }
    for (const call of parsed.calls) {
      if (textOpen) {
        yield sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex });
        textOpen = false;
      }
      const index = nextIndex;
      nextIndex += 1;
      yield sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'tool_use', id: `toolu_${index}`, name: call.name, input: {} },
      });
      const args = argumentsString(call.args);
      if (args !== '' && args !== '{}') {
        yield sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: args } });
      }
      yield sseEvent('content_block_stop', { type: 'content_block_stop', index });
    }
    if (parsed.finish !== null && parsed.finish !== undefined) finish = anthropicStopOfGemini(parsed.finish);
    if (parsed.output !== null) outputTokens = parsed.output;
  }
  yield* startEvents();
  if (textOpen) yield sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex });
  yield sseEvent('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: finish ?? 'end_turn', stop_sequence: null },
    usage: { output_tokens: outputTokens },
  });
  yield sseEvent('message_stop', { type: 'message_stop' });
}
