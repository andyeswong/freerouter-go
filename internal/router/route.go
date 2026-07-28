package router

import (
	"errors"

	"github.com/andyeswong/freerouter-go/internal/models"
)

// Request is what the router needs to make a decision. Fields beyond Prompt are
// optional caller hints — when present they override the heuristic classifier.
type Request struct {
	Prompt             string
	SystemPrompt       string
	MaxTokens          int
	ContextChars       int  // total chars across ALL messages (for context sizing); 0 = derive from prompt+system
	HasTools           bool // request carries function-calling tools → only ToolsOK models
	RequiresJSONSchema bool // response_format=json_schema → only JSONSchemaOK models

	// Optional declared metadata (data-driven path, preferred over heuristics).
	Tier         models.Tier // 0 = let the classifier decide
	RequiresMCP  *bool       // nil = infer from AgenticScore
}

// Decision is the routing result plus metadata for logging / savings.
type Decision struct {
	Model       models.LlmModel
	Tier        models.Tier
	Confidence  float64
	Method      string // "declared" | "override" | "heuristic"
	RequiresMCP bool
	CostEstimate float64
	BaselineCost float64
	Savings      float64 // fraction [0,1] vs most-expensive enabled model
	Reason       string
}

var ErrNoCandidate = errors.New("no eligible model for request")

// ErrNoJSONSchemaModel distinguishes the structured-output dead end from a
// generic no-candidate result: the fix is to enable/mark a capable model, not
// to shrink the request.
var ErrNoJSONSchemaModel = errors.New("no eligible model supports response_format=json_schema (mark a capable model json_schema_ok)")

// MaxOutputClamp caps the caller-supplied max_output used for the context-fit
// check in CandidatesFor. No real model streams hundreds of thousands of output
// tokens; some clients (e.g. Coder's aibridge) reserve a huge output budget
// derived from a 1M context_limit, which — added to a large input — wrongly
// pushes even 1M-window models past the context filter, yielding a spurious
// "no eligible model". Clamping only affects ROUTING eligibility; the untouched
// request body (with the caller's original max_tokens) is still relayed upstream.
var MaxOutputClamp = 131072

// Router ties the classifier to the vademécum.
type Router struct {
	repo *models.Repo
	cfg  HeuristicConfig
}

func New(repo *models.Repo, cfg HeuristicConfig) *Router {
	return &Router{repo: repo, cfg: cfg}
}

func estTokens(s string) int { return (len(s) + 3) / 4 }

// Route picks the cheapest sufficient healthy model.
func (rt *Router) Route(req Request) (*Decision, error) {
	prompt := req.Prompt
	method := "heuristic"
	tier := req.Tier
	conf := 0.6

	// 1. Mode override (prompt prefix) beats everything.
	if ot, stripped, ok := ModeOverride(prompt); ok {
		tier, prompt, method, conf = ot, stripped, "override", 0.99
	} else if tier != 0 {
		method, conf = "declared", 0.95
	}

	// 2. Heuristic fallback only if still no tier.
	if tier == 0 {
		tier, conf = Classify(prompt, estTokens(prompt), rt.cfg)
	}

	// 3. requires_mcp: declared wins, else infer — kept separate from tier.
	requiresMCP := false
	if req.RequiresMCP != nil {
		requiresMCP = *req.RequiresMCP
	} else {
		requiresMCP = AgenticScore(prompt, rt.cfg)
	}

	// Size the real request by ALL messages, not just the last prompt.
	ctxChars := req.ContextChars
	if ctxChars == 0 {
		ctxChars = len(req.SystemPrompt) + len(req.Prompt)
	}

	// Clamp an unrealistic caller max_output for the context-fit check only.
	maxOut := req.MaxTokens
	if maxOut > MaxOutputClamp {
		maxOut = MaxOutputClamp
	}

	cands, err := rt.repo.CandidatesFor(models.CandidateQuery{
		Tier:               tier,
		RequiresMCP:        requiresMCP,
		HasTools:           req.HasTools,
		RequiresJSONSchema: req.RequiresJSONSchema,
		ContextChars:       ctxChars,
		MaxOutput:          maxOut,
		Margin:             1.2,
	})
	if err != nil {
		return nil, err
	}

	// First healthy candidate; fall back to first candidate if none are "up".
	var pick *models.LlmModel
	for i := range cands {
		if cands[i].Health == models.HealthUp || cands[i].Health == models.HealthUnknown {
			pick = &cands[i]
			break
		}
	}
	if pick == nil && len(cands) > 0 {
		pick = &cands[0]
	}
	if pick == nil {
		if req.RequiresJSONSchema {
			return nil, ErrNoJSONSchemaModel
		}
		return nil, ErrNoCandidate
	}

	d := &Decision{
		Model:       *pick,
		Tier:        tier,
		Confidence:  conf,
		Method:      method,
		RequiresMCP: requiresMCP,
	}
	// Estimate input tokens for the savings calc using the chosen model's ratio.
	cpt := pick.CharsPerToken
	if cpt <= 0 {
		cpt = 4
	}
	estInput := int(float64(ctxChars) / cpt)
	rt.computeSavings(d, estInput, req.MaxTokens)
	d.Reason = method + " -> tier " + tierName(tier)
	return d, nil
}

func tierName(t models.Tier) string {
	switch t {
	case models.TierSimple:
		return "SIMPLE"
	case models.TierMedium:
		return "MEDIUM"
	case models.TierComplex:
		return "COMPLEX"
	case models.TierReasoning:
		return "REASONING"
	default:
		return "MAX"
	}
}
