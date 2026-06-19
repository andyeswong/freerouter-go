package models

import "gorm.io/gorm"

// Repo wraps DB access to the vademécum.
type Repo struct{ db *gorm.DB }

func NewRepo(db *gorm.DB) *Repo { return &Repo{db: db} }

// AutoMigrate creates/updates the schema.
func (r *Repo) AutoMigrate() error { return r.db.AutoMigrate(&LlmModel{}) }

// CandidateQuery describes what a routing decision needs from a model.
type CandidateQuery struct {
	Tier         Tier // minimum capability the request needs
	RequiresMCP  bool // true only when the task needs an agentic/MCP-native model
	ContextChars int  // total chars across ALL messages; 0 = skip context filter
	MaxOutput    int  // tokens to reserve for the completion
	Margin       float64 // safety multiplier on estimated input tokens (default 1.2)
}

// CandidatesFor returns eligible models ordered cheapest-sufficient first.
//
// This is the heart of data-driven routing, ported from Pillbox's
// LlmModel::scopeCandidatesFor. Order: tier_max ASC, cost ASC, weight DESC.
// The first healthy result is the model to use.
//
// Filters:
//   - enabled = true
//   - tier_max >= requested tier (model is capable enough)
//   - mcp_native = true ONLY when RequiresMCP (plain tool use must NOT pin here)
//   - context fits: each model is judged by ITS OWN chars_per_token ratio —
//     estimated_input = context_chars/chars_per_token, and the model qualifies
//     when context_window >= estimated_input*margin + max_output (0 window = kept)
func (r *Repo) CandidatesFor(q CandidateQuery) ([]LlmModel, error) {
	tx := r.db.Where("enabled = ?", true).
		Where("tier_max >= ?", q.Tier)

	if q.RequiresMCP {
		tx = tx.Where("mcp_native = ?", true)
	}
	if q.ContextChars > 0 {
		margin := q.Margin
		if margin <= 0 {
			margin = 1.2
		}
		// per-row arithmetic: divide by the model's own chars_per_token (guard 0).
		tx = tx.Where(
			"context_window = 0 OR context_window >= "+
				"(CAST(? AS REAL) / (CASE WHEN chars_per_token > 0 THEN chars_per_token ELSE 4 END)) * ? + ?",
			q.ContextChars, margin, q.MaxOutput)
	}

	var out []LlmModel
	err := tx.Order("tier_max ASC, cost ASC, weight DESC").Find(&out).Error
	return out, err
}

// UpdateCharsPerToken folds an observed ratio into the model's EMA. Observations
// outside a sane range are ignored (provider quirks / bad data). alpha in (0,1].
func (r *Repo) UpdateCharsPerToken(id uint, observed, alpha float64) error {
	if observed < 1.5 || observed > 8 {
		return nil
	}
	var m LlmModel
	if err := r.db.First(&m, id).Error; err != nil {
		return err
	}
	cur := m.CharsPerToken
	if cur <= 0 {
		cur = 4
	}
	next := alpha*observed + (1-alpha)*cur
	return r.db.Model(&LlmModel{}).Where("id = ?", id).Update("chars_per_token", next).Error
}

// MostExpensiveEnabled returns the priciest enabled model, used as the honest
// savings baseline (fixes freerouter's hardcoded claude-opus baseline).
func (r *Repo) MostExpensiveEnabled() (*LlmModel, error) {
	var m LlmModel
	err := r.db.Where("enabled = ?", true).
		Order("(input_price + output_price) DESC").
		First(&m).Error
	if err != nil {
		return nil, err
	}
	return &m, nil
}

func (r *Repo) List() ([]LlmModel, error) {
	var out []LlmModel
	return out, r.db.Order("tier_max ASC, cost ASC").Find(&out).Error
}

func (r *Repo) Get(id uint) (*LlmModel, error) {
	var m LlmModel
	return &m, r.db.First(&m, id).Error
}

func (r *Repo) Create(m *LlmModel) error { return r.db.Create(m).Error }
func (r *Repo) Save(m *LlmModel) error   { return r.db.Save(m).Error }
func (r *Repo) Delete(id uint) error     { return r.db.Delete(&LlmModel{}, id).Error }
