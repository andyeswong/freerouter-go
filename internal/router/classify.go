// Package router decides which tier a request needs and which model serves it.
//
// Tier resolution order:
//  1. Explicit: caller declared a tier (or mode override like /max, [simple]).
//  2. Heuristic fallback: lightweight keyword scoring — used ONLY when no tier
//     was declared. Unlike upstream freerouter, matching is word-boundary, not
//     naive substring (no more "art" matching "start").
package router

import (
	"regexp"
	"strings"

	"github.com/andyeswong/freerouter-go/internal/models"
)

// ModeOverride scans the prompt prefix for a forced tier, e.g. "/max do X" or
// "[simple] hi". Returns the tier and the prompt with the marker stripped.
func ModeOverride(prompt string) (models.Tier, string, bool) {
	p := strings.TrimSpace(prompt)
	overrides := map[string]models.Tier{
		"/simple": models.TierSimple, "[simple]": models.TierSimple,
		"/medium": models.TierMedium, "[medium]": models.TierMedium,
		"/complex": models.TierComplex, "[complex]": models.TierComplex,
		"/max": models.TierMax, "[max]": models.TierMax,
		"/reason": models.TierReasoning, "[reason]": models.TierReasoning,
	}
	for marker, tier := range overrides {
		if strings.HasPrefix(strings.ToLower(p), marker) {
			return tier, strings.TrimSpace(p[len(marker):]), true
		}
	}
	return 0, prompt, false
}

// dimension is one weighted signal.
type dimension struct {
	keywords []string
	weight   float64
}

// HeuristicConfig is the fallback classifier's knobs (loaded from config JSON).
type HeuristicConfig struct {
	SimpleKeywords    []string `json:"simple_keywords"`
	ReasoningKeywords []string `json:"reasoning_keywords"`
	CodeKeywords      []string `json:"code_keywords"`
	AgenticKeywords   []string `json:"agentic_keywords"`
	// Score boundaries mapping aggregate score -> tier.
	SimpleMedium    float64 `json:"simple_medium"`
	MediumComplex   float64 `json:"medium_complex"`
	ComplexReason   float64 `json:"complex_reason"`
}

// wordBoundaryCount counts keywords present as whole words (case-insensitive).
// This is the fix for freerouter bug #1 (substring false positives).
func wordBoundaryCount(text string, keywords []string) int {
	n := 0
	lower := strings.ToLower(text)
	for _, kw := range keywords {
		// \b around the (escaped) keyword; cheap enough for short prompts.
		re := regexp.MustCompile(`\b` + regexp.QuoteMeta(strings.ToLower(kw)) + `\b`)
		if re.MatchString(lower) {
			n++
		}
	}
	return n
}

// Classify maps a prompt to a tier using the heuristic fallback. estTokens is a
// rough char/4 estimate of the USER prompt only (system prompt excluded so a
// "hi" with a 10K system prompt doesn't route to the top tier — same reasoning
// as upstream freerouter index.ts).
func Classify(prompt string, estTokens int, cfg HeuristicConfig) (models.Tier, float64) {
	score := 0.0

	// Length signal.
	switch {
	case estTokens < 40:
		score -= 1.0
	case estTokens > 800:
		score += 1.0
	}

	// Keyword signals (word-boundary).
	if wordBoundaryCount(prompt, cfg.SimpleKeywords) > 0 {
		score -= 1.0
	}
	score += 0.7 * float64(min(wordBoundaryCount(prompt, cfg.CodeKeywords), 2))
	score += 0.9 * float64(min(wordBoundaryCount(prompt, cfg.ReasoningKeywords), 2))

	// Multi-step patterns.
	if regexp.MustCompile(`(?i)first.*then|step \d|\d\.\s`).MatchString(prompt) {
		score += 0.5
	}

	// Direct reasoning override: 2+ reasoning markers => REASONING.
	if wordBoundaryCount(prompt, cfg.ReasoningKeywords) >= 2 {
		return models.TierReasoning, 0.85
	}

	switch {
	case score < cfg.SimpleMedium:
		return models.TierSimple, confidence(cfg.SimpleMedium - score)
	case score < cfg.MediumComplex:
		return models.TierMedium, confidence(min(score-cfg.SimpleMedium, cfg.MediumComplex-score))
	case score < cfg.ComplexReason:
		return models.TierComplex, confidence(min(score-cfg.MediumComplex, cfg.ComplexReason-score))
	default:
		return models.TierReasoning, confidence(score - cfg.ComplexReason)
	}
}

// AgenticScore returns true when the prompt looks like it needs an MCP-native
// model (real tool loop), kept SEPARATE from plain tier (Pillbox 5e2448c).
func AgenticScore(prompt string, cfg HeuristicConfig) bool {
	return wordBoundaryCount(prompt, cfg.AgenticKeywords) >= 4
}

func confidence(distance float64) float64 {
	// sigmoid-ish, clamped to [0.5, 1.0]
	c := 0.5 + 0.5*distance
	if c > 1.0 {
		return 1.0
	}
	if c < 0.5 {
		return 0.5
	}
	return c
}
