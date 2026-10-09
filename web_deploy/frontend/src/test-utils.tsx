// What the view tests share: a fake gateway that answers by path and records what was sent.

export interface Call {
  method: string;
  url: string;
  body: any;
}

type Reply = { status?: number; body: unknown } | ((call: Call) => { status?: number; body: unknown });

export function fakeGateway(routes: Record<string, Reply>): Call[] {
  const calls: Call[] = [];
  globalThis.fetch = vi.fn(async (url: any, init?: any) => {
    const call = { method: init?.method ?? "GET", url: String(url), body: init?.body ? JSON.parse(init.body) : null };
    calls.push(call);
    const key = Object.keys(routes).find((k) => k === `${call.method} ${call.url.split("?")[0]}` || k === call.url.split("?")[0]);
    const route = key ? routes[key] : { status: 404, body: { error: "Not found" } };
    const reply = typeof route === "function" ? route(call) : route;
    return new Response(JSON.stringify(reply.body), { status: reply.status ?? 200 });
  }) as any;
  return calls;
}

export const FIRM_STAGE = { stage: "firm", console: null, firm: null, user: null, languages: ["en", "sr"] };
export const ACCOUNT_STAGE = {
  stage: "account", console: "firm", firm: { slug: "legal-demo", name: "Legal demo", default_language: "en" },
  user: null, languages: ["en", "sr"],
};
