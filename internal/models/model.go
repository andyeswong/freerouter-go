// Package models holds the "vademécum" — the table of LLM models the router
// can choose from. Mirrors Pillbox's LlmModel: routing is data-driven, not
// keyword-driven. The cheapest sufficient model wins.
package models

import "time"

// Tier is the coarse complexity bucket a request maps to. 1 = trivial, 5 = hardest.
type Tier int

const (
	TierSimple    Tier = 1
	TierMedium    Tier = 2
	TierComplex   Tier = 3
	TierReasoning Tier = 4
	TierMax       Tier = 5
)

// Health is the result of the last scan against the model endpoint.
type Health string

const (
	HealthUnknown  Health = "unknown"
	HealthUp       Health = "up"
	HealthDegraded Health = "degraded"
	HealthDown     Health = "down"
)

// LlmModel is one routable model. Pricing is per 1M tokens.
//
// Routing key fields (see repo.CandidatesFor):
//   - TierMax: the highest tier this model is allowed to serve. A request at
//     tier T is eligible for any model whose TierMax >= T.
//   - Cost: relative cost rank used for ordering (lower wins).
//   - Weight: tie-breaker preference (higher wins).
//   - McpNative: true only for models that can drive a real agentic tool loop
//     (e.g. Claude Code via cc_bridge). Pills/requests that genuinely need MCP
//     orchestration filter on this — NOT plain tool use. See Pillbox commit
//     5e2448c: requires_tooling != requires_mcp.
type LlmModel struct {
	ID uint `gorm:"primaryKey" json:"id"`

	Name       string `gorm:"uniqueIndex;not null" json:"name"`        // human label, also the id callers may pin
	Provider   string `json:"provider"`                                // anthropic | openai | ollama | ...
	ModelID    string `gorm:"not null" json:"model_id"`                // upstream model id sent over the wire
	APIBaseURL string `json:"api_base_url"`                            // OpenAI-compatible base, e.g. http://localhost:11434/v1
	APIKeyRef  string `json:"api_key_ref"`                             // name of env var / secret holding the key (never the key itself)

	InputPrice    float64 `json:"input_price"`  // USD per 1M input tokens
	OutputPrice   float64 `json:"output_price"` // USD per 1M output tokens
	ContextWindow int     `json:"context_window"`

	// CharsPerToken is the empirical chars-per-token ratio for this model,
	// auto-calibrated (EMA) from real usage.prompt_tokens on every request.
	// Used to size context for routing and to estimate token counts when a
	// provider doesn't report them. Default 4.0.
	CharsPerToken float64 `gorm:"default:4" json:"chars_per_token"`

	TierMax   Tier `gorm:"index" json:"tier_max"`
	Cost      int  `gorm:"index" json:"cost"`   // ordering rank, lower = cheaper
	Weight    int  `json:"weight"`              // tie-breaker, higher = preferred
	McpNative bool `gorm:"index" json:"mcp_native"`

	// CustomSystemPrompt, if set, is injected as a system message at the front
	// of every request routed to this model — per-model behavior steering
	// (e.g. "do not execute tools, return the command instead").
	CustomSystemPrompt string `json:"custom_system_prompt"`

	// ForceNoExec injects "no_exec":true into every request to this model. For
	// cc_bridge-backed Claude models it means: don't execute tools, return the
	// tool call to the parent harness instead. See cc_bridge no_exec mode.
	ForceNoExec bool `json:"force_no_exec"`

	Enabled bool   `gorm:"index" json:"enabled"`
	Health  Health `gorm:"default:unknown" json:"health"`
	LatencyP50Ms int `json:"latency_p50_ms"`

	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}
