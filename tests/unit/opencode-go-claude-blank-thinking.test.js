// OpenCode Go's Claude models sit on the /messages transport with Anthropic's
// thinking pass-back constraints — but stricter: the upstream rejects a
// thinking block with no signature outright (400 "thinking.signature: Field
// required"), and a blank one regardless of signature. prepareClaudeRequest's
// thinking pass was gated on handlesThinkingBlocks(provider), which does not
// list opencode-go, so ocg/claude-* replayed a blank or unsigned block verbatim
// and 400'd. They now get the same treatment as DeepSeek-served models: keep
// non-blank blocks as-is (original signature), drop blank ones. Non-Claude
// models on the same provider (qwen/glm/kimi) are untouched — their upstreams
// accept the client's signature verbatim and had been working.
import { describe, it, expect } from "vitest";
import { prepareClaudeRequest } from "../../open-sse/translator/formats/claude.js";
import { DEFAULT_THINKING_CLAUDE_SIGNATURE } from "../../open-sse/config/defaultThinkingSignature.js";

const body = (thinking, sig) => ({
  model: "claude-haiku-5-5",
  max_tokens: 2048,
  thinking: { type: "enabled", budget_tokens: 1024 },
  messages: [
    { role: "user", content: "hi" },
    { role: "assistant", content: sig
      ? [{ type: "thinking", thinking, signature: sig }, { type: "text", text: "ok" }]
      : [{ type: "thinking", thinking }, { type: "text", text: "ok" }] },
    { role: "user", content: "again" },
  ],
});

const assistantBlocks = (out) => out.messages.find((m) => m.role === "assistant").content.map((b) => b.type);

describe("prepareClaudeRequest — ocg Claude models clean thinking blocks", () => {
  it("drops a blank unsigned thinking block (400 signature: Field required)", () => {
    expect(assistantBlocks(prepareClaudeRequest(body("", null), "opencode-go"))).toEqual(["text"]);
  });

  it("drops a blank block even with a fake signature", () => {
    expect(assistantBlocks(prepareClaudeRequest(body("   ", "fakesig==="), "opencode-go"))).toEqual(["text"]);
  });

  it("keeps a non-blank block with its original signature", () => {
    const out = prepareClaudeRequest(body("real thought", "clientsig==="), "opencode-go");
    const a = out.messages.find((m) => m.role === "assistant");
    const th = a.content.find((b) => b.type === "thinking");
    expect(a.content.map((b) => b.type)).toEqual(["thinking", "text"]);
    expect(th.thinking).toBe("real thought");
    expect(th.signature).toBe("clientsig===");
  });

  it("leaves non-Claude models on the same provider untouched", () => {
    const out = prepareClaudeRequest({
      model: "qwen3.8-flash",
      max_tokens: 100,
      messages: [
        { role: "user", content: "hi" },
        { role: "assistant", content: [{ type: "thinking", thinking: "", signature: DEFAULT_THINKING_CLAUDE_SIGNATURE }, { type: "text", text: "ok" }] },
        { role: "user", content: "go" },
      ],
    }, "opencode-go");
    const a = out.messages.find((m) => m.role === "assistant");
    // passthrough behavior preserved: the block rides through as-is
    expect(a.content.map((b) => b.type)).toEqual(["thinking", "text"]);
  });

  it("deepseek-served behavior is unchanged (regression guard)", () => {
    const out = prepareClaudeRequest(body("real thought", DEFAULT_THINKING_CLAUDE_SIGNATURE), "deepseek");
    expect(assistantBlocks(out)).toEqual(["thinking", "text"]);
  });
});
