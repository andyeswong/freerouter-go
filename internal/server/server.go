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
	"time"

	"github.com/gin-gonic/gin"

	"github.com/andyeswong/freerouter-go/internal/auth"
	"github.com/andyeswong/freerouter-go/internal/models"
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
	adminToken string
}

func New(repo *models.Repo, rt *router.Router, tokens *auth.Repo, usageRepo *usage.Repo, secretsRepo *secrets.Repo, quotaTracker *quota.Tracker, adminToken string) *Server {
	return &Server{repo: repo, rt: rt, tokens: tokens, usage: usageRepo, secrets: secretsRepo, quota: quotaTracker, adminToken: adminToken}
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

		admin.GET("/secrets", s.secretList)
		admin.POST("/secrets", s.secretSet)
		admin.DELETE("/secrets/:name", s.secretDelete)
		admin.GET("/keys", s.keyList)
	}
	return r
}

// chat: auth (middleware) -> classify -> pick model -> proxy -> relay + record usage.
func (s *Server) chat(c *gin.Context) {
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
		RequiresMCP:        req.RequiresMCP,
	})
	if err != nil {
		c.JSON(503, gin.H{"error": err.Error()})
		return
	}

	c.Header("X-FreeRouter-Model", decision.Model.Name)
	c.Header("X-FreeRouter-Tier", strconv.Itoa(int(decision.Tier)))
	c.Header("X-FreeRouter-Savings", strconv.FormatFloat(decision.Savings, 'f', 3, 64))

	resp, err := providers.Proxy(decision.Model, raw)
	if err != nil {
		c.JSON(502, gin.H{"error": "upstream error: " + err.Error()})
		return
	}
	defer resp.Body.Close()

	c.Status(resp.StatusCode)
	if ct := resp.Header.Get("Content-Type"); ct != "" {
		c.Header("Content-Type", ct)
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
			go s.recordUsage(tok, decision, buf.Bytes(), ctxChars)
		}
	}
}

// recordUsage parses upstream token counts, calibrates the model's
// chars-per-token ratio (EMA), and writes a usage row. promptChars is the size
// of the input (all messages) — used to estimate prompt tokens when the
// provider doesn't report them (e.g. Ollama Cloud only returns completion).
func (s *Server) recordUsage(tok *auth.ApiToken, d *router.Decision, body []byte, promptChars int) {
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
