package models

import "gorm.io/gorm"

// Repo wraps DB access to the vademécum.
type Repo struct{ db *gorm.DB }

func NewRepo(db *gorm.DB) *Repo { return &Repo{db: db} }

// AutoMigrate creates/updates the schema.
func (r *Repo) AutoMigrate() error { return r.db.AutoMigrate(&LlmModel{}) }

// CandidateQuery describes what a routing decision needs from a model.
type CandidateQuery struct {
	Tier        Tier // minimum capability the request needs
	RequiresMCP bool // true only when the task needs an agentic/MCP-native model
	MinContext  int  // total tokens (prompt+completion) the model must fit; 0 = ignore
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
//   - context_window covers MinContext (0 context_window = unknown, kept)
func (r *Repo) CandidatesFor(q CandidateQuery) ([]LlmModel, error) {
	tx := r.db.Where("enabled = ?", true).
		Where("tier_max >= ?", q.Tier)

	if q.RequiresMCP {
		tx = tx.Where("mcp_native = ?", true)
	}
	if q.MinContext > 0 {
		// keep models with unknown (0) context window OR enough room (10% headroom)
		tx = tx.Where("context_window = 0 OR context_window >= ?", int(float64(q.MinContext)*1.1))
	}

	var out []LlmModel
	err := tx.Order("tier_max ASC, cost ASC, weight DESC").Find(&out).Error
	return out, err
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
