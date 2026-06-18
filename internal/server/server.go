// Package server wires the HTTP surface: OpenAI-compatible proxy + admin CRUD.
package server

import (
	"encoding/json"
	"io"
	"strconv"

	"github.com/gin-gonic/gin"

	"github.com/andyeswong/freerouter-go/internal/models"
	"github.com/andyeswong/freerouter-go/internal/providers"
	"github.com/andyeswong/freerouter-go/internal/router"
)

type Server struct {
	repo *models.Repo
	rt   *router.Router
}

func New(repo *models.Repo, rt *router.Router) *Server {
	return &Server{repo: repo, rt: rt}
}

func (s *Server) Engine() *gin.Engine {
	r := gin.New()
	r.Use(gin.Recovery())

	r.GET("/health", func(c *gin.Context) { c.JSON(200, gin.H{"status": "ok"}) })
	r.GET("/v1/models", s.listModelsOpenAI)
	r.POST("/v1/chat/completions", s.chat)

	admin := r.Group("/admin")
	{
		admin.GET("/models", s.adminList)
		admin.POST("/models", s.adminCreate)
		admin.PUT("/models/:id", s.adminUpdate)
		admin.DELETE("/models/:id", s.adminDelete)
		admin.POST("/models/:id/scan", s.adminScan)
	}
	return r
}

// chat: classify -> pick model -> proxy -> relay (incl. streaming passthrough).
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

	// Surface the routing decision for observability.
	c.Header("X-FreeRouter-Model", decision.Model.Name)
	c.Header("X-FreeRouter-Tier", strconv.Itoa(int(decision.Tier)))
	c.Header("X-FreeRouter-Savings", strconv.FormatFloat(decision.Savings, 'f', 3, 64))

	resp, err := providers.Proxy(decision.Model, raw)
	if err != nil {
		c.JSON(502, gin.H{"error": "upstream error: " + err.Error()})
		return
	}
	defer resp.Body.Close()

	// Relay status + content-type, stream the body straight through.
	c.Status(resp.StatusCode)
	if ct := resp.Header.Get("Content-Type"); ct != "" {
		c.Header("Content-Type", ct)
	}
	_, _ = io.Copy(c.Writer, resp.Body)
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

// ---- admin CRUD ----

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

// adminScan pings the model endpoint and records health (estilo Pillbox model scan).
func (s *Server) adminScan(c *gin.Context) {
	id, _ := strconv.Atoi(c.Param("id"))
	m, err := s.repo.Get(uint(id))
	if err != nil {
		c.JSON(404, gin.H{"error": "not found"})
		return
	}
	// Minimal probe: 1-token completion. Cheap health signal.
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
