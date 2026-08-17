// Package server wires the HTTP surface: token-gated OpenAI-compatible proxy,
// admin model/token CRUD, and a usage-reporting API.
package server

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/andyeswong/freerouter-go/internal/auth"
	"github.com/andyeswong/freerouter-go/internal/models"
	"github.com/andyeswong/freerouter-go/internal/promptlog"
	"github.com/andyeswong/freerouter-go/internal/providers"
	"github.com/andyeswong/freerouter-go/internal/quota"
	"github.com/andyeswong/freerouter-go/internal/router"
	"github.com/andyeswong/freerouter-go/internal/secrets"
	"github.com/andyeswong/freerouter-go/internal/usage"
)

type Server struct {
	repo       *models.Repo
	rt         *router.Router
	tokens     *auth.Repo
	usage      *usage.Repo
	secrets    *secrets.Repo
	quota      *quota.Tracker
	prompts    *promptlog.Logger
	adminToken string

	seriesMu    sync.Mutex
	seriesCache map[string]seriesEntry
}

func New(repo *models.Repo, rt *router.Router, tokens *auth.Repo, usageRepo *usage.Repo, secretsRepo *secrets.Repo, quotaTracker *quota.Tracker, prompts *promptlog.Logger, adminToken string) *Server {
	return &Server{repo: repo, rt: rt, tokens: tokens, usage: usageRepo, secrets: secretsRepo, quota: quotaTracker, prompts: prompts, adminToken: adminToken}
}

func (s *Server) Engine() *gin.Engine {
	r := gin.New()
	r.Use(gin.Recovery())
	// Minimal access log: gin.New() ships with none, so a fast-failing request
	// (e.g. a proxy error returned before any slow/erroring DB query) left zero
	// trace anywhere. This makes every request's outcome visible.
	r.Use(func(c *gin.Context) {
		start := time.Now()
		c.Next()
		who := "-"
		if tok, ok := auth.TokenFromCtx(c); ok {
			who = tok.Name
		}
		if len(c.Errors) > 0 {
			log.Printf("%s %s status=%d dur=%s token=%s err=%s", c.Request.Method, c.Request.URL.Path, c.Writer.Status(), time.Since(start), who, c.Errors.String())
		} else {
			log.Printf("%s %s status=%d dur=%s token=%s", c.Request.Method, c.Request.URL.Path, c.Writer.Status(), time.Since(start), who)
		}
	})

	r.GET("/health", func(c *gin.Context) { c.JSON(200, gin.H{"status": "ok"}) })

	// Consumer surface — requires a per-dev frgo_ token.
	v1 := r.Group("/v1", s.tokens.RequireToken())
	{
		v1.GET("/models", s.listModelsOpenAI)
		// Quota only gates the endpoint that actually spends tokens; listing
		// models must keep working for a client that is over budget.
		v1.POST("/chat/completions", quota.Enforce(s.quota), s.chat)
	}

	// Admin surface — gated by the static admin token.
	admin := r.Group("/admin", auth.RequireAdmin(s.adminToken))
	{
		admin.GET("/models", s.adminList)
		admin.POST("/models", s.adminCreate)
		admin.PUT("/models/:id", s.adminUpdate)
		admin.DELETE("/models/:id", s.adminDelete)
		admin.POST("/models/:id/scan", s.adminScan)

		admin.GET("/tokens", s.tokenList)
		admin.POST("/tokens", s.tokenIssue)
		admin.POST("/tokens/:id/revoke", s.tokenRevoke)
		admin.POST("/tokens/:id/enable", s.tokenEnable)
		admin.PUT("/tokens/:id/limits", s.tokenSetLimits)

		admin.GET("/usage", s.usageReport)
		admin.GET("/usage/recent", s.usageRecent)
		admin.GET("/usage/series", s.usageSeries)
		admin.GET("/usage/by-tier", s.usageByTier)
		admin.GET("/usage/models-stats", s.usageModelStats)

		admin.GET("/prompt-log", s.promptLogGet)
		admin.PUT("/prompt-log", s.promptLogSet)

		admin.GET("/secrets", s.secretList)
		admin.POST("/secrets", s.secretSet)
		admin.DELETE("/secrets/:name", s.secretDelete)
		admin.GET("/keys", s.keyList)
	}
	return r
}

// chat: auth (middleware) -> classify -> pick model -> proxy -> relay + record usage.
func (s *Server) chat(c *gin.Context) {
	start := time.Now()
	raw, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.JSON(400, gin.H{"error": "cannot read body"})
		return
	}

	var req providers.ChatRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		c.JSON(400, gin.H{"error": "invalid JSON: " + err.Error()})
		return
	}

	user, system := providers.ExtractPrompt(req.Messages)
	ctxChars := providers.ContextChars(req.Messages)
	decision, err := s.rt.Route(router.Request{
		Prompt:             user,
		SystemPrompt:       system,
		MaxTokens:          req.MaxTokens,
		ContextChars:       ctxChars,
		HasTools:           req.HasTools(),
		RequiresJSONSchema: req.RequiresJSONSchema(),
		Tier:               models.Tier(req.Tier),
	})
	if err != nil {
		if tok, ok := auth.TokenFromCtx(c); ok {
			go s.recordFailure(tok, "(no candidate)", 0, "", 503, int(time.Since(start).Milliseconds()), 0)
		}
		c.JSON(503, gin.H{"error": err.Error()})
		return
	}

	c.Header("X-FreeRouter-Model", decision.Model.Name)
	c.Header("X-FreeRouter-Tier", strconv.Itoa(int(decision.Tier)))
	c.Header("X-FreeRouter-Savings", strconv.FormatFloat(decision.Savings, 'f', 3, 64))

	// Capture the prompt (no-op unless prompt logging is on for this token).
	// Placed after routing so the entry records which model the traffic hit, and
	// before proxying so a request that dies upstream still leaves its prompt.
	if tok, ok := auth.TokenFromCtx(c); ok {
		s.capturePrompt(tok, req, decision, ctxChars)
	}

	resp, err := providers.Proxy(decision.Model, raw)
	if err != nil {
		if tok, ok := auth.TokenFromCtx(c); ok {
			go s.recordFailure(tok, decision.Model.Name, int(decision.Tier), decision.Method, 502, int(time.Since(start).Milliseconds()), decision.Savings)
		}
		c.JSON(502, gin.H{"error": "upstream error: " + err.Error()})
		return
	}
	defer resp.Body.Close()

	c.Status(resp.StatusCode)
	if ct := resp.Header.Get("Content-Type"); ct != "" {
		c.Header("Content-Type", ct)
	}

	// A schema served by instruction (see providers.EmulatesJSONSchema) can come
	// back wrapped in a fence or a sentence of preamble. The caller is about to
	// parse it, so tighten it to the JSON object here — only on the non-streaming
	// path, which is the one that was failing; a stream is relayed untouched.
	if providers.EmulatesJSONSchema(decision.Model, req) && !req.Stream && resp.StatusCode < 400 {
		body, readErr := io.ReadAll(resp.Body)
		if readErr == nil {
			out := tightenToJSON(body)
			c.Header("X-FreeRouter-JSONSchema", "emulated")
			_, _ = c.Writer.Write(out)
			if tok, ok := auth.TokenFromCtx(c); ok {
				go s.recordUsage(tok, decision, out, ctxChars, int(time.Since(start).Milliseconds()), resp.StatusCode)
			}
			return
		}
		// Couldn't read it: fall through and relay whatever is left.
	}

	// Tee: relay to the client while capturing the full body for usage billing.
	var buf bytes.Buffer
	_, _ = io.Copy(c.Writer, io.TeeReader(resp.Body, &buf))

	if resp.StatusCode < 400 {
		if tok, ok := auth.TokenFromCtx(c); ok {
			// Billing/calibration writes are not on the client's critical path
			// (the response body is already fully relayed above) — run them
			// off-request so a busy SQLite writer never stalls the next request's
			// auth check behind this one's bookkeeping.
			go s.recordUsage(tok, decision, buf.Bytes(), ctxChars, int(time.Since(start).Milliseconds()), resp.StatusCode)
		}
	}
}

// tightenToJSON rewrites a completion's message content down to the JSON object
// it contains. Returns the body untouched whenever anything is not as expected
// (not a completion, no object found, …) — a best-effort cleanup must never turn
// a usable answer into a broken one.
func tightenToJSON(body []byte) []byte {
	var doc map[string]any
	if json.Unmarshal(body, &doc) != nil {
		return body
	}
	choices, ok := doc["choices"].([]any)
	if !ok || len(choices) == 0 {
		return body
	}
	changed := false
	for _, ch := range choices {
		choice, ok := ch.(map[string]any)
		if !ok {
			continue
		}
		msg, ok := choice["message"].(map[string]any)
		if !ok {
			continue
		}
		content, ok := msg["content"].(string)
		if !ok || content == "" {
			continue
		}
		if extracted, found := providers.ExtractJSON(content); found && extracted != content {
			msg["content"] = extracted
			changed = true
		}
	}
	if !changed {
		return body
	}
	out, err := json.Marshal(doc)
	if err != nil {
		return body
	}
	return out
}

// recordUsage parses upstream token counts, calibrates the model's
// chars-per-token ratio (EMA), and writes a usage row. promptChars is the size
// of the input (all messages) — used to estimate prompt tokens when the
// provider doesn't report them (e.g. Ollama Cloud only returns completion).
func (s *Server) recordUsage(tok *auth.ApiToken, d *router.Decision, body []byte, promptChars, durationMs, status int) {
	u := providers.ParseUsage(body)

	estimated := false
	if u.PromptTokens > 0 {
		// Real count: fold the observed ratio into the model's EMA (alpha 0.2).
		observed := float64(promptChars) / float64(u.PromptTokens)
		_ = s.repo.UpdateCharsPerToken(d.Model.ID, observed, 0.2)
	} else if promptChars > 0 {
		// Provider didn't report prompt tokens — estimate from the model's ratio.
		cpt := d.Model.CharsPerToken
		if cpt <= 0 {
			cpt = 4
		}
		u.PromptTokens = int(float64(promptChars) / cpt)
		estimated = true
	}
	if u.TotalTokens == 0 || estimated {
		u.TotalTokens = u.PromptTokens + u.CompletionTokens
	}

	cost := float64(u.PromptTokens)/1e6*d.Model.InputPrice +
		float64(u.CompletionTokens)/1e6*d.Model.OutputPrice

	// Book the spend against the live quota counters BEFORE the row lands in
	// SQLite: a hydration racing this Add can then only miss an in-flight row,
	// never count it twice (see quota.Tracker.Add).
	if s.quota != nil {
		s.quota.Add(tok.ID, int64(u.TotalTokens))
	}

	_ = s.usage.Add(&usage.Record{
		TokenID:          tok.ID,
		User:             tok.Name,
		Model:            d.Model.Name,
		Tier:             int(d.Tier),
		PromptTokens:     u.PromptTokens,
		CompletionTokens: u.CompletionTokens,
		TotalTokens:      u.TotalTokens,
		CostEstimate:     cost,
		Estimated:        estimated,
		DurationMs:       durationMs,
		Status:           status,
		Method:           d.Method,
		Savings:          d.Savings,
	})
}

// recordFailure books a row for a request that never reached (or came back from)
// an upstream: a 503 (no eligible model) or a 502 (proxy error). Keeps the
// routing/reliability view honest — a failure is data, not a gap in the log.
func (s *Server) recordFailure(tok *auth.ApiToken, model string, tier int, method string, status, durationMs int, savings float64) {
	_ = s.usage.Add(&usage.Record{
		TokenID:    tok.ID,
		User:       tok.Name,
		Model:      model,
		Tier:       tier,
		Status:     status,
		Method:     method,
		Savings:    savings,
		DurationMs: durationMs,
	})
}

func (s *Server) listModelsOpenAI(c *gin.Context) {
	ms, err := s.repo.List()
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	data := make([]gin.H, 0, len(ms))
	for _, m := range ms {
		data = append(data, gin.H{"id": m.Name, "object": "model", "owned_by": m.Provider})
	}
	c.JSON(200, gin.H{"object": "list", "data": data})
}

// ---- admin: models ----

func (s *Server) adminList(c *gin.Context) {
	ms, err := s.repo.List()
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	c.JSON(200, ms)
}

func (s *Server) adminCreate(c *gin.Context) {
	var m models.LlmModel
	if err := c.ShouldBindJSON(&m); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	if err := s.repo.Create(&m); err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	c.JSON(201, m)
}

func (s *Server) adminUpdate(c *gin.Context) {
	id, _ := strconv.Atoi(c.Param("id"))
	m, err := s.repo.Get(uint(id))
	if err != nil {
		c.JSON(404, gin.H{"error": "not found"})
		return
	}
	if err := c.ShouldBindJSON(m); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	m.ID = uint(id)
	if err := s.repo.Save(m); err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	c.JSON(200, m)
}

func (s *Server) adminDelete(c *gin.Context) {
	id, _ := strconv.Atoi(c.Param("id"))
	if err := s.repo.Delete(uint(id)); err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	c.Status(204)
}

func (s *Server) adminScan(c *gin.Context) {
	id, _ := strconv.Atoi(c.Param("id"))
	m, err := s.repo.Get(uint(id))
	if err != nil {
		c.JSON(404, gin.H{"error": "not found"})
		return
	}
	probe := []byte(`{"model":"` + m.ModelID + `","messages":[{"role":"user","content":"ping"}],"max_tokens":1}`)
	resp, err := providers.Proxy(*m, probe)
	if err != nil {
		m.Health = models.HealthDown
	} else {
		if resp.StatusCode < 400 {
			m.Health = models.HealthUp
		} else {
			m.Health = models.HealthDegraded
		}
		providers.Drain(resp.Body)
	}
	_ = s.repo.Save(m)
	c.JSON(200, gin.H{"id": m.ID, "health": m.Health})
}

// ---- admin: tokens ----

// tokenWithQuota is a token row plus its live per-window consumption, so the
// dashboard can render "used / limit" without a second round trip.
type tokenWithQuota struct {
	auth.ApiToken
	Quota []quota.WindowState `json:"quota"`
}

func (s *Server) tokenList(c *gin.Context) {
	ts, err := s.tokens.List()
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	out := make([]tokenWithQuota, 0, len(ts))
	for i := range ts {
		row := tokenWithQuota{ApiToken: ts[i]}
		if s.quota != nil {
			row.Quota = s.quota.Snapshot(ts[i].ID, quota.LimitsOf(&ts[i]))
		}
		out = append(out, row)
	}
	c.JSON(200, out)
}

// tokenSetLimits updates a token's ceilings. Omitted fields keep their current
// value; an explicit 0 clears that window's limit (unlimited).
func (s *Server) tokenSetLimits(c *gin.Context) {
	id, _ := strconv.Atoi(c.Param("id"))
	var body struct {
		Daily   *int64 `json:"limit_daily_tokens"`
		Weekly  *int64 `json:"limit_weekly_tokens"`
		Monthly *int64 `json:"limit_monthly_tokens"`
	}
	if err := c.ShouldBindJSON(&body); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	for _, v := range []*int64{body.Daily, body.Weekly, body.Monthly} {
		if v != nil && *v < 0 {
			c.JSON(400, gin.H{"error": "limits must be >= 0 (0 = unlimited)"})
			return
		}
	}
	if err := s.tokens.SetLimits(uint(id), body.Daily, body.Weekly, body.Monthly); err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	tok, err := s.tokens.Get(uint(id))
	if err != nil {
		c.JSON(404, gin.H{"error": "not found"})
		return
	}
	// Drop cached counters so the response reflects freshly-summed usage.
	if s.quota != nil {
		s.quota.Forget(tok.ID)
	}
	row := tokenWithQuota{ApiToken: *tok}
	if s.quota != nil {
		row.Quota = s.quota.Snapshot(tok.ID, quota.LimitsOf(tok))
	}
	c.JSON(200, row)
}

// capturePrompt hands the parsed request to the prompt log. Everything about
// whether it actually gets written (enabled? this token? truncation?) lives in
// promptlog, so the proxy path stays one call.
func (s *Server) capturePrompt(tok *auth.ApiToken, req providers.ChatRequest, decision *router.Decision, ctxChars int) {
	if s.prompts == nil || decision == nil {
		return
	}
	msgs := make([]promptlog.Message, 0, len(req.Messages))
	for _, m := range req.Messages {
		msgs = append(msgs, promptlog.Message{Role: m.Role, Content: m.Content})
	}
	s.prompts.Log(promptlog.Entry{
		Token:        tok.Name,
		TokenID:      tok.ID,
		Model:        decision.Model.Name,
		Tier:         int(decision.Tier),
		Method:       decision.Method,
		Stream:       req.Stream,
		Tools:        req.HasTools(),
		ContextChars: ctxChars,
		Messages:     msgs,
	})
}

// promptLogGet reports whether prompts are being captured, for which tokens,
// and how big the file has grown.
func (s *Server) promptLogGet(c *gin.Context) {
	if s.prompts == nil {
		c.JSON(200, gin.H{"enabled": false, "note": "prompt log not wired"})
		return
	}
	cfg := s.prompts.Config()
	c.JSON(200, gin.H{
		"enabled":   cfg.Enabled,
		"path":      cfg.Path,
		"tokens":    cfg.Tokens,
		"max_chars": cfg.MaxChars,
		"max_bytes": cfg.MaxBytes,
		"size":      s.prompts.Size(),
	})
}

// promptLogSet flips capture on/off and optionally retargets it, live — no
// restart, which matters because restarting drops in-flight SSE streams.
// Omitted fields keep their current value; "tokens":[] widens capture to every
// token, so it must be sent deliberately.
func (s *Server) promptLogSet(c *gin.Context) {
	if s.prompts == nil {
		c.JSON(503, gin.H{"error": "prompt log not wired"})
		return
	}
	var body struct {
		Enabled  *bool     `json:"enabled"`
		Path     *string   `json:"path"`
		Tokens   *[]string `json:"tokens"`
		MaxChars *int      `json:"max_chars"`
		MaxBytes *int64    `json:"max_bytes"`
	}
	if err := c.ShouldBindJSON(&body); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	cfg := s.prompts.Config()
	if body.Enabled != nil {
		cfg.Enabled = *body.Enabled
	}
	if body.Path != nil {
		if *body.Path == "" {
			c.JSON(400, gin.H{"error": "path cannot be empty"})
			return
		}
		cfg.Path = *body.Path
	}
	if body.Tokens != nil {
		cfg.Tokens = *body.Tokens
	}
	if body.MaxChars != nil {
		if *body.MaxChars < 0 {
			c.JSON(400, gin.H{"error": "max_chars must be >= 0 (0 = whole message)"})
			return
		}
		cfg.MaxChars = *body.MaxChars
	}
	if body.MaxBytes != nil {
		if *body.MaxBytes < 0 {
			c.JSON(400, gin.H{"error": "max_bytes must be >= 0 (0 = never rotate)"})
			return
		}
		cfg.MaxBytes = *body.MaxBytes
	}
	if err := s.prompts.Configure(cfg); err != nil {
		c.JSON(500, gin.H{"error": "cannot open " + cfg.Path + ": " + err.Error()})
		return
	}
	log.Printf("prompt log: enabled=%t path=%s tokens=%v", cfg.Enabled, cfg.Path, cfg.Tokens)
	s.promptLogGet(c)
}

// tokenIssue creates a dev token. The plaintext is returned ONCE here.
func (s *Server) tokenIssue(c *gin.Context) {
	var body struct {
		Name string `json:"name" binding:"required"`
	}
	if err := c.ShouldBindJSON(&body); err != nil {
		c.JSON(400, gin.H{"error": "name required"})
		return
	}
	tok, plain, err := s.tokens.Issue(body.Name)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	c.JSON(201, gin.H{
		"id":    tok.ID,
		"name":  tok.Name,
		"token": plain, // show once — not stored in plaintext
		"note":  "store this now; it cannot be retrieved again",
	})
}

func (s *Server) tokenRevoke(c *gin.Context) {
	id, _ := strconv.Atoi(c.Param("id"))
	if err := s.tokens.SetEnabled(uint(id), false); err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	c.JSON(200, gin.H{"id": id, "enabled": false})
}

func (s *Server) tokenEnable(c *gin.Context) {
	id, _ := strconv.Atoi(c.Param("id"))
	if err := s.tokens.SetEnabled(uint(id), true); err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	c.JSON(200, gin.H{"id": id, "enabled": true})
}

// ---- admin: usage ----

func parseTime(s string) *time.Time {
	if s == "" {
		return nil
	}
	if t, err := time.Parse(time.RFC3339, s); err == nil {
		return &t
	}
	if t, err := time.Parse("2006-01-02", s); err == nil {
		return &t
	}
	return nil
}

func (s *Server) usageFilter(c *gin.Context) usage.Filter {
	return usage.Filter{
		User:  c.Query("user"),
		Model: c.Query("model"),
		From:  parseTime(c.Query("from")),
		To:    parseTime(c.Query("to")),
	}
}

// usageReport answers "which dev used how many tokens of which model".
func (s *Server) usageReport(c *gin.Context) {
	buckets, err := s.usage.Aggregate(s.usageFilter(c))
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	c.JSON(200, gin.H{"buckets": buckets})
}
func (s *Server) usageByTier(c *gin.Context) {
	rows, err := s.usage.ByTier(s.usageFilter(c))
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	c.JSON(200, gin.H{"tiers": rows})
}
func (s *Server) usageModelStats(c *gin.Context) {
	rows, err := s.usage.ModelStats(s.usageFilter(c))
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	c.JSON(200, gin.H{"models": rows})
}

// usageSeries answers "how much per day (or hour), by whom" — the aggregation
// the dashboard's trends and per-user daily averages are computed from.
// Buckets are labeled in the quota timezone so a "day" here is the same day a
// quota window uses.
// seriesTTL caches the aggregation briefly. The query is O(all history) — it
// groups by a per-row strftime, which no index can satisfy — and it measured
// 333ms in production. With a single pooled connection those milliseconds
// serialize every other request, including an incoming chat's auth lookup, so a
// dashboard polling every 15s must not translate into a slow query every 15s.
// Usage figures tolerate being a minute stale; the router stalling does not.
const seriesTTL = 60 * time.Second

type seriesEntry struct {
	at   time.Time
	body gin.H
}

func (s *Server) usageSeries(c *gin.Context) {
	loc := time.UTC
	if s.quota != nil {
		loc = s.quota.Location()
	}

	key := c.Request.URL.RawQuery
	s.seriesMu.Lock()
	if e, ok := s.seriesCache[key]; ok && time.Since(e.at) < seriesTTL {
		s.seriesMu.Unlock()
		c.Header("X-FreeRouter-Cache", "hit")
		c.JSON(200, e.body)
		return
	}
	s.seriesMu.Unlock()

	points, err := s.usage.Series(s.usageFilter(c), c.Query("bucket"), c.Query("group"), loc)
	if err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	body := gin.H{"points": points, "timezone": loc.String()}

	s.seriesMu.Lock()
	if s.seriesCache == nil {
		s.seriesCache = map[string]seriesEntry{}
	}
	s.seriesCache[key] = seriesEntry{at: time.Now(), body: body}
	s.seriesMu.Unlock()

	c.Header("X-FreeRouter-Cache", "miss")
	c.JSON(200, body)
}

func (s *Server) usageRecent(c *gin.Context) {
	limit, _ := strconv.Atoi(c.DefaultQuery("limit", "100"))
	recs, err := s.usage.Recent(s.usageFilter(c), limit)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	c.JSON(200, gin.H{"records": recs})
}

// ---- admin: secrets (provider keys, DB-backed, hot — no restart) ----

func (s *Server) secretList(c *gin.Context) {
	items, err := s.secrets.List()
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	c.JSON(200, items)
}

func (s *Server) secretSet(c *gin.Context) {
	var body struct {
		Name  string `json:"name" binding:"required"`
		Value string `json:"value" binding:"required"`
	}
	if err := c.ShouldBindJSON(&body); err != nil {
		c.JSON(400, gin.H{"error": "name and value required"})
		return
	}
	if !validSecretName(body.Name) {
		c.JSON(400, gin.H{"error": "name must be UPPER_SNAKE (A-Z 0-9 _)"})
		return
	}
	if err := s.secrets.Set(body.Name, body.Value); err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	c.JSON(200, gin.H{"name": body.Name, "ok": true})
}

func (s *Server) secretDelete(c *gin.Context) {
	if err := s.secrets.Delete(c.Param("name")); err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	c.Status(204)
}

// keyList returns every provider-key reference, combining the DB secret store
// and the api_key_ref values used by models — each tagged with where it
// actually resolves from (db | env | missing) and a masked preview.
func (s *Server) keyList(c *gin.Context) {
	dbItems, _ := s.secrets.List()
	inDB := map[string]string{}
	for _, it := range dbItems {
		inDB[it.Name] = it.Preview
	}

	usedBy := map[string]int{}
	order := []string{}
	seen := map[string]bool{}
	add := func(name string) {
		if name == "" || seen[name] {
			return
		}
		seen[name] = true
		order = append(order, name)
	}

	ms, _ := s.repo.List()
	for _, m := range ms {
		if m.APIKeyRef != "" {
			usedBy[m.APIKeyRef]++
		}
		add(m.APIKeyRef)
	}
	for _, it := range dbItems {
		add(it.Name)
	}

	out := make([]gin.H, 0, len(order))
	for _, name := range order {
		source, preview := "missing", ""
		if p, ok := inDB[name]; ok {
			source, preview = "db", p
		} else if v := os.Getenv(name); v != "" {
			source, preview = "env", maskKey(v)
		}
		out = append(out, gin.H{"name": name, "source": source, "preview": preview, "used_by": usedBy[name]})
	}
	c.JSON(200, out)
}

func maskKey(v string) string {
	if len(v) <= 6 {
		return "••••"
	}
	return v[:4] + "…" + v[len(v)-2:]
}

func validSecretName(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if !(r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '_') {
			return false
		}
	}
	return true
}
