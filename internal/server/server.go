// Package server wires the HTTP surface: token-gated OpenAI-compatible proxy,
// admin model/token CRUD, and a usage-reporting API.
package server

import (
	"bytes"
	"encoding/json"
	"io"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/andyeswong/freerouter-go/internal/auth"
	"github.com/andyeswong/freerouter-go/internal/models"
	"github.com/andyeswong/freerouter-go/internal/providers"
	"github.com/andyeswong/freerouter-go/internal/router"
	"github.com/andyeswong/freerouter-go/internal/usage"
)

type Server struct {
	repo       *models.Repo
	rt         *router.Router
	tokens     *auth.Repo
	usage      *usage.Repo
	adminToken string
}

func New(repo *models.Repo, rt *router.Router, tokens *auth.Repo, usageRepo *usage.Repo, adminToken string) *Server {
	return &Server{repo: repo, rt: rt, tokens: tokens, usage: usageRepo, adminToken: adminToken}
}

func (s *Server) Engine() *gin.Engine {
	r := gin.New()
	r.Use(gin.Recovery())

	r.GET("/health", func(c *gin.Context) { c.JSON(200, gin.H{"status": "ok"}) })

	// Consumer surface — requires a per-dev frgo_ token.
	v1 := r.Group("/v1", s.tokens.RequireToken())
	{
		v1.GET("/models", s.listModelsOpenAI)
		v1.POST("/chat/completions", s.chat)
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

		admin.GET("/usage", s.usageReport)
		admin.GET("/usage/recent", s.usageRecent)
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
	decision, err := s.rt.Route(router.Request{
		Prompt:       user,
		SystemPrompt: system,
		MaxTokens:    req.MaxTokens,
		Tier:         models.Tier(req.Tier),
		RequiresMCP:  req.RequiresMCP,
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
		s.recordUsage(c, decision, buf.Bytes())
	}
}

// recordUsage parses upstream token counts and writes a usage row for the dev.
func (s *Server) recordUsage(c *gin.Context, d *router.Decision, body []byte) {
	tok, ok := auth.TokenFromCtx(c)
	if !ok {
		return
	}
	u := providers.ParseUsage(body)
	cost := float64(u.PromptTokens)/1e6*d.Model.InputPrice +
		float64(u.CompletionTokens)/1e6*d.Model.OutputPrice

	_ = s.usage.Add(&usage.Record{
		TokenID:          tok.ID,
		User:             tok.Name,
		Model:            d.Model.Name,
		Tier:             int(d.Tier),
		PromptTokens:     u.PromptTokens,
		CompletionTokens: u.CompletionTokens,
		TotalTokens:      u.TotalTokens,
		CostEstimate:     cost,
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

func (s *Server) tokenList(c *gin.Context) {
	ts, err := s.tokens.List()
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	c.JSON(200, ts)
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
