/**
 * Gemini → OpenAI API 兼容代理（Deno Deploy 版）
 *
 * 作用：把 OpenAI 格式的 /v1/chat/completions 请求翻译成 Google Gemini
 *       的 generateContent / streamGenerateContent，再把结果翻译回 OpenAI 格式。
 *       这样任何"支持自定义 OpenAI Base URL"的客户端（AstrBot、NextChat、
 *       LobeChat、OpenWebUI 等）都能直接用。
 *
 * 部署（Deno Deploy 空白项目）：
 *   1. 把本文件内容全选粘贴进项目的入口文件（main.ts / index.ts）。
 *   2. 项目 Settings → Environment Variables 里加：
 *        GEMINI_API_KEY = 你的 Google AI Studio API Key
 *      可选：
 *        GEMINI_MODEL   = gemini-2.0-flash   （默认就是这个）
 *        PROXY_TOKEN    = 随便设一个字符串     （设了之后客户端必须带同样的 Bearer）
 *   3. 点 Deploy，得到 https://<项目名>.deno.dev
 *   4. 客户端里填：
 *        Base URL = https://<项目名>.deno.dev/v1
 *        API Key  = GEMINI_API_KEY 的值（或你设的 PROXY_TOKEN）
 *        Model    = gemini-2.0-flash
 *
 * 支持：文本、图片（data URL 或 http 图片）、流式 / 非流式、system 提示词、
 *       工具调用（function calling）、/v1/models、CORS 预检。
 */

// ---------------------------------------------------------------- 配置

const GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta";
const DEFAULT_MODEL = Deno.env.get("GEMINI_MODEL") || "gemini-2.0-flash";
const PROXY_TOKEN = Deno.env.get("PROXY_TOKEN") || "";

// ---------------------------------------------------------------- CORS

const CORS_HEADERS: Record<string, string> = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, HEAD",
  "Access-Control-Allow-Headers":
    "Authorization, Content-Type, Accept, Origin, X-Requested-With, " +
    "api-key, x-api-key, anthropic-version, anthropic-beta, openai-organization",
  "Access-Control-Expose-Headers": "Content-Type, Content-Length",
  "Access-Control-Max-Age": "86400",
};

function withCors(init: ResponseInit = {}): ResponseInit {
  const headers = new Headers(init.headers || {});
  for (const [k, v] of Object.entries(CORS_HEADERS)) headers.set(k, v);
  return { ...init, headers };
}

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), withCors({
    status,
    headers: { "Content-Type": "application/json; charset=utf-8" },
  }));
}

/** OpenAI 风格的错误体 */
function oaiError(message: string, status = 400, type = "invalid_request_error", code?: string): Response {
  return json({
    error: {
      message,
      type,
      param: null,
      code: code ?? String(status),
    },
  }, status);
}

function nowSec(): number {
  return Math.floor(Date.now() / 1000);
}

function randId(prefix = "chatcmpl"): string {
  return `${prefix}-${crypto.randomUUID().replace(/-/g, "").slice(0, 24)}`;
}

// ---------------------------------------------------------------- 鉴权

function resolveApiKey(req: Request): string {
  // 优先用环境变量里的 Gemini Key
  const envKey = Deno.env.get("GEMINI_API_KEY") || Deno.env.get("GOOGLE_API_KEY") || "";
  const auth = req.headers.get("authorization") || "";
  const bearer = auth.toLowerCase().startsWith("bearer ")
    ? auth.slice(7).trim()
    : (req.headers.get("x-api-key") || req.headers.get("api-key") || "").trim();

  // 设了 PROXY_TOKEN：客户端必须带它，且真实 Key 从环境变量取
  if (PROXY_TOKEN) {
    if (bearer !== PROXY_TOKEN && bearer !== envKey) return "";
    return envKey || bearer;
  }

  // 没设 PROXY_TOKEN：环境变量优先；否则把 Bearer 当作 Gemini Key 用
  return envKey || bearer;
}

// ---------------------------------------------------------------- 类型

type OAIContentPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string; detail?: string } }
  | { type: string; [k: string]: unknown };

interface OAIMessage {
  role: "system" | "developer" | "user" | "assistant" | "tool" | string;
  content?: string | OAIContentPart[] | null;
  name?: string;
  tool_calls?: unknown[];
  tool_call_id?: string;
}

interface OAITool {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
}

interface OAIRequest {
  model?: string;
  messages?: OAIMessage[];
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  max_completion_tokens?: number;
  stop?: string | string[] | null;
  stream?: boolean;
  n?: number;
  tools?: OAITool[];
  tool_choice?: unknown;
  presence_penalty?: number;
  frequency_penalty?: number;
  response_format?: { type?: string };
}

// ---------------------------------------------------------------- 请求转换

function textOf(content: OAIMessage["content"]): string {
  if (content == null) return "";
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((p) => {
        if (typeof p === "string") return p;
        if (p && typeof p === "object" && (p as OAIContentPart).type === "text") {
          return String((p as { text?: string }).text ?? "");
        }
        return "";
      })
      .join("");
  }
  return String(content);
}

async function partFromContentPart(p: OAIContentPart): Promise<Record<string, unknown> | null> {
  if (typeof p === "string") return { text: p };

  const type = (p as { type?: string }).type;

  if (type === "text") {
    return { text: String((p as { text?: string }).text ?? "") };
  }

  if (type === "image_url") {
    const url = String(
      (p as { image_url?: { url?: string } }).image_url?.url ?? "",
    );
    if (!url) return null;

    // data URL: data:image/png;base64,xxxx
    const m = /^data:([^;]+);base64,(.*)$/s.exec(url);
    if (m) {
      return { inlineData: { mimeType: m[1], data: m[2] } };
    }

    // 普通 http(s) 图片：下载后内联（Gemini 不走 fileData 需要先上传）
    try {
      const r = await fetch(url);
      if (!r.ok) return null;
      const ct = r.headers.get("content-type") || "image/jpeg";
      const buf = new Uint8Array(await r.arrayBuffer());
      let bin = "";
      const chunk = 0x8000;
      for (let i = 0; i < buf.length; i += chunk) {
        bin += String.fromCharCode(...buf.subarray(i, i + chunk));
      }
      return { inlineData: { mimeType: ct.split(";")[0], data: btoa(bin) } };
    } catch {
      return null;
    }
  }

  // 其它类型（如 input_audio）忽略
  return null;
}

async function convertMessages(
  messages: OAIMessage[],
): Promise<{ systemInstruction?: unknown; contents: unknown[] }> {
  const systemTexts: string[] = [];
  const contents: Array<{ role: "user" | "model"; parts: unknown[] }> = [];

  for (const msg of messages) {
    const role = String(msg.role || "user");

    if (role === "system" || role === "developer") {
      const t = textOf(msg.content);
      if (t.trim()) systemTexts.push(t);
      continue;
    }

    if (role === "tool") {
      // 工具结果当作 user 侧的文本回填
      const t = textOf(msg.content);
      contents.push({
        role: "user",
        parts: [{ text: `[tool result ${msg.tool_call_id ?? ""}] ${t}` }],
      });
      continue;
    }

    if (role === "assistant") {
      const parts: unknown[] = [];
      const t = textOf(msg.content);
      if (t) parts.push({ text: t });

      // assistant 的 tool_calls 转成 functionCall
      const tcs = Array.isArray(msg.tool_calls) ? msg.tool_calls : [];
      for (const tc of tcs) {
        const fn = (tc as { function?: { name?: string; arguments?: string } }).function;
        if (fn?.name) {
          let args: Record<string, unknown> = {};
          try {
            args = fn.arguments ? JSON.parse(fn.arguments) : {};
          } catch {
            args = { _raw: fn.arguments ?? "" };
          }
          parts.push({ functionCall: { name: fn.name, args } });
        }
      }

      if (parts.length) contents.push({ role: "model", parts });
      continue;
    }

    // user
    const parts: unknown[] = [];
    if (Array.isArray(msg.content)) {
      for (const p of msg.content) {
        const conv = await partFromContentPart(p as OAIContentPart);
        if (conv) parts.push(conv);
      }
    } else {
      const t = textOf(msg.content);
      if (t) parts.push({ text: t });
    }
    if (parts.length) contents.push({ role: "user", parts });
  }

  // Gemini 要求 contents 第一条最好是 user；若为空补一个占位
  if (contents.length === 0) contents.push({ role: "user", parts: [{ text: "hi" }] });

  const out: { systemInstruction?: unknown; contents: unknown[] } = { contents };
  if (systemTexts.length) {
    out.systemInstruction = { parts: [{ text: systemTexts.join("\n\n") }] };
  }
  return out;
}

function convertTools(tools: OAITool[] | undefined): unknown[] | undefined {
  if (!Array.isArray(tools) || tools.length === 0) return undefined;
  const decls = tools
    .filter((t) => t?.type === "function" && t.function?.name)
    .map((t) => ({
      name: t.function.name,
      description: t.function.description ?? "",
      parameters: t.function.parameters ?? { type: "object", properties: {} },
    }));
  return decls.length ? [{ functionDeclarations: decls }] : undefined;
}

// ---------------------------------------------------------------- 响应转换

interface GeminiCandidate {
  content?: { parts?: Array<{ text?: string; functionCall?: { name?: string; args?: unknown } }>; role?: string };
  finishReason?: string;
  index?: number;
}

interface GeminiResponse {
  candidates?: GeminiCandidate[];
  promptFeedback?: { blockReason?: string };
  usageMetadata?: {
    promptTokenCount?: number;
    candidatesTokenCount?: number;
    totalTokenCount?: number;
  };
}

function finishReason(g?: GeminiResponse): string {
  const raw = g.candidates?.[0]?.finishReason ?? "";
  switch (raw) {
    case "STOP":
      return "stop";
    case "MAX_TOKENS":
      return "length";
    case "SAFETY":
    case "RECITATION":
    case "PROHIBITED_CONTENT":
    case "BLOCKLIST":
      return "content_filter";
    default:
      return raw ? "stop" : "stop";
  }
}

function collect(g: GeminiResponse): {
  text: string;
  toolCalls: Array<{ id: string; type: "function"; function: { name: string; arguments: string } }>;
} {
  let text = "";
  const toolCalls: Array<{ id: string; type: "function"; function: { name: string; arguments: string } }> = [];

  for (const cand of g.candidates ?? []) {
    for (const part of cand.content?.parts ?? []) {
      if (typeof part.text === "string") text += part.text;
      if (part.functionCall?.name) {
        toolCalls.push({
          id: `call_${crypto.randomUUID().replace(/-/g, "").slice(0, 20)}`,
          type: "function",
          function: {
            name: part.functionCall.name,
            arguments: JSON.stringify(part.functionCall.args ?? {}),
          },
        });
      }
    }
  }
  return { text, toolCalls };
}

function usageOf(g: GeminiResponse) {
  const u = g.usageMetadata || {};
  return {
    prompt_tokens: u.promptTokenCount ?? 0,
    completion_tokens: u.candidatesTokenCount ?? 0,
    total_tokens: u.totalTokenCount ?? 0,
  };
}

// ---------------------------------------------------------------- 主处理

async function handleChatCompletions(req: Request): Promise<Response> {
  const apiKey = resolveApiKey(req);
  if (!apiKey) {
    return oaiError(
      "缺少 Gemini API Key：请在 Deno Deploy 项目里设置环境变量 GEMINI_API_KEY，" +
        "或在请求头里带 Authorization: Bearer <你的 Gemini Key>",
      401,
      "invalid_request_error",
      "missing_api_key",
    );
  }

  let body: OAIRequest;
  try {
    body = await req.json();
  } catch {
    return oaiError("请求体不是合法 JSON", 400);
  }

  const messages = Array.isArray(body.messages) ? body.messages : [];
  if (messages.length === 0) return oaiError("messages 不能为空", 400);

  // 模型：优先用请求里显式指定的 gemini-* 模型，否则用默认
  const reqModel = String(body.model || "").trim();
  const model = reqModel.toLowerCase().includes("gemini") ? reqModel : DEFAULT_MODEL;

  const { systemInstruction, contents } = await convertMessages(messages);

  const generationConfig: Record<string, unknown> = {};
  if (typeof body.temperature === "number") generationConfig.temperature = body.temperature;
  if (typeof body.top_p === "number") generationConfig.topP = body.top_p;
  const maxTokens = body.max_tokens ?? body.max_completion_tokens;
  if (typeof maxTokens === "number" && maxTokens > 0) generationConfig.maxOutputTokens = maxTokens;
  if (body.stop) generationConfig.stopSequences = Array.isArray(body.stop) ? body.stop : [body.stop];

  const payload: Record<string, unknown> = { contents };
  if (systemInstruction) payload.systemInstruction = systemInstruction;
  if (Object.keys(generationConfig).length) payload.generationConfig = generationConfig;

  const tools = convertTools(body.tools);
  if (tools) payload.tools = tools;

  const stream = body.stream === true;
  const url =
    `${GEMINI_BASE}/models/${encodeURIComponent(model)}:` +
    (stream ? "streamGenerateContent?alt=sse&key=" : "generateContent?key=") +
    encodeURIComponent(apiKey);

  let upstream: Response;
  try {
    upstream = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (e) {
    return oaiError(`连接 Gemini 失败：${(e as Error).message}`, 502, "api_error");
  }

  if (!upstream.ok) {
    const errText = await upstream.text().catch(() => "");
    let msg = errText;
    try {
      const j = JSON.parse(errText);
      msg = j?.error?.message || errText;
    } catch { /* 保持原文 */ }
    return oaiError(
      `Gemini 返回 ${upstream.status}：${msg.slice(0, 800)}`,
      upstream.status === 400 ? 400 : upstream.status === 403 ? 403 : upstream.status === 429 ? 429 : 502,
      "api_error",
    );
  }

  const created = nowSec();
  const id = randId();

  // ---------- 非流式 ----------
  if (!stream) {
    let g: GeminiResponse;
    try {
      g = await upstream.json();
    } catch (e) {
      return oaiError(`解析 Gemini 响应失败：${(e as Error).message}`, 502, "api_error");
    }

    if (g.promptFeedback?.blockReason && !g.candidates?.length) {
      return oaiError(
        `内容被 Gemini 安全策略拦截：${g.promptFeedback.blockReason}`,
        400,
        "content_filter",
        "content_filter",
      );
    }

    const { text, toolCalls } = collect(g);
    const message: Record<string, unknown> = { role: "assistant", content: text };
    if (toolCalls.length) {
      message.tool_calls = toolCalls;
      if (!text) message.content = null;
    }

    return json({
      id,
      object: "chat.completion",
      created,
      model: model,
      choices: [
        {
          index: 0,
          message,
          finish_reason: toolCalls.length ? "tool_calls" : finishReason(g),
          logprobs: null,
        },
      ],
      usage: usageOf(g),
    });
  }

  // ---------- 流式 ----------
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();

  const readable = new ReadableStream<Uint8Array>({
    async start(controller) {
      const send = (obj: unknown) => {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(obj)}\n\n`));
      };

      // 首个 chunk 带 role
      send({
        id,
        object: "chat.completion.chunk",
        created,
        model,
        choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }],
      });

      let buffer = "";
      let finish = "stop";
      let usage = { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };

      try {
        const reader = upstream.body!.getReader();
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          let idx: number;
          // SSE 事件以空行分隔
          while ((idx = buffer.indexOf("\n\n")) >= 0) {
            const raw = buffer.slice(0, idx);
            buffer = buffer.slice(idx + 2);

            for (const line of raw.split("\n")) {
              const trimmed = line.trim();
              if (!trimmed.startsWith("data:")) continue;
              const dataStr = trimmed.slice(5).trim();
              if (!dataStr || dataStr === "[DONE]") continue;

              let g: GeminiResponse;
              try {
                g = JSON.parse(dataStr);
              } catch {
                continue;
              }

              if (g.usageMetadata) usage = usageOf(g);
              finish = finishReason(g);

              const { text, toolCalls } = collect(g);

              if (text) {
                send({
                  id,
                  object: "chat.completion.chunk",
                  created,
                  model,
                  choices: [{ index: 0, delta: { content: text }, finish_reason: null }],
                });
              }

              if (toolCalls.length) {
                send({
                  id,
                  object: "chat.completion.chunk",
                  created,
                  model,
                  choices: [{
                    index: 0,
                    delta: { tool_calls: toolCalls.map((tc, i) => ({ index: i, ...tc })) },
                    finish_reason: null,
                  }],
                });
              }
            }
          }
        }
      } catch (e) {
        // 上游中断：给客户端一个错误 chunk，然后正常收尾
        send({
          id,
          object: "chat.completion.chunk",
          created,
          model,
          choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
          error: { message: `上游流中断：${(e as Error).message}`, type: "api_error" },
        });
      }

      // 收尾 chunk
      send({
        id,
        object: "chat.completion.chunk",
        created,
        model,
        choices: [{ index: 0, delta: {}, finish_reason: finish }],
        usage,
      });
      controller.enqueue(encoder.encode("data: [DONE]\n\n"));
      controller.close();
    },
  });

  return new Response(readable, withCors({
    status: 200,
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      "Connection": "keep-alive",
      "X-Accel-Buffering": "no",
    },
  }));
}

function handleModels(): Response {
  const model = DEFAULT_MODEL;
  return json({
    object: "list",
    data: [
      { id: model, object: "model", created: 0, owned_by: "google" },
      { id: "gemini-2.0-flash", object: "model", created: 0, owned_by: "google" },
      { id: "gemini-2.0-flash-lite", object: "model", created: 0, owned_by: "google" },
      { id: "gemini-2.5-flash", object: "model", created: 0, owned_by: "google" },
      { id: "gemini-2.5-pro", object: "model", created: 0, owned_by: "google" },
    ],
  });
}

// ---------------------------------------------------------------- 入口

Deno.serve(async (req: Request): Promise<Response> => {
  const url = new URL(req.url);
  const path = url.pathname.replace(/\/+$/, "") || "/";

  // CORS 预检
  if (req.method === "OPTIONS") {
    return new Response(null, withCors({ status: 204 }));
  }

  // 健康检查 / 首页
  if (path === "/" || path === "/healthz") {
    return json({
      ok: true,
      service: "gemini-openai-proxy",
      model: DEFAULT_MODEL,
      endpoints: [
        "POST /v1/chat/completions",
        "GET  /v1/models",
      ],
      note: "Base URL 填 https://<你的项目>.deno.dev/v1",
      hasEnvKey: Boolean(Deno.env.get("GEMINI_API_KEY") || Deno.env.get("GOOGLE_API_KEY")),
      hasProxyToken: Boolean(PROXY_TOKEN),
    });
  }

  if (path === "/v1/models" || path === "/models") {
    return handleModels();
  }

  if (path === "/v1/chat/completions" || path === "/chat/completions") {
    if (req.method !== "POST") {
      return oaiError("该接口只接受 POST", 405, "invalid_request_error", "method_not_allowed");
    }
    return await handleChatCompletions(req);
  }

  // 部分客户端会探测 /v1/responses（新版 OpenAI 接口）：简单回 501，别报错
  if (path === "/v1/responses") {
    return oaiError("本代理只实现 /v1/chat/completions", 501, "invalid_request_error", "not_implemented");
  }

  return oaiError(`未知路径：${path}`, 404, "invalid_request_error", "not_found");
});
