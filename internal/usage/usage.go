// Package usage records per-request token consumption and answers
// "which dev used how many tokens of which model".
package usage

import (
	"time"

	"gorm.io/gorm"
)

// Record is one billed request.
type Record struct {
	ID      uint   `gorm:"primaryKey" json:"id"`
	TokenID uint   `gorm:"index" json:"token_id"`
	User    string `gorm:"index" json:"user"`  // ApiToken.Name (dev identity)
	Model   string `gorm:"index" json:"model"` // LlmModel.Name chosen by the router
	Tier    int    `json:"tier"`

	PromptTokens     int     `json:"prompt_tokens"`
	CompletionTokens int     `json:"completion_tokens"`
	TotalTokens      int     `json:"total_tokens"`
	CostEstimate     float64 `json:"cost_estimate"` // USD, from model pricing × tokens
	Estimated        bool    `json:"estimated"`     // true when the provider didn't report prompt_tokens and we estimated them

	CreatedAt time.Time `gorm:"index" json:"created_at"`
}

type Repo struct{ db *gorm.DB }

func NewRepo(db *gorm.DB) *Repo { return &Repo{db: db} }

func (r *Repo) AutoMigrate() error { return r.db.AutoMigrate(&Record{}) }

func (r *Repo) Add(rec *Record) error { return r.db.Create(rec).Error }

// Filter narrows aggregation/list queries. Zero values = no filter.
type Filter struct {
	User  string
	Model string
	From  *time.Time
	To    *time.Time
}

func (r *Repo) scope(f Filter) *gorm.DB {
	tx := r.db.Model(&Record{})
	if f.User != "" {
		tx = tx.Where("user = ?", f.User)
	}
	if f.Model != "" {
		tx = tx.Where("model = ?", f.Model)
	}
	if f.From != nil {
		tx = tx.Where("created_at >= ?", *f.From)
	}
	if f.To != nil {
		tx = tx.Where("created_at <= ?", *f.To)
	}
	return tx
}

// Bucket is one aggregated (user, model) row.
type Bucket struct {
	User             string  `json:"user"`
	Model            string  `json:"model"`
	Requests         int64   `json:"requests"`
	PromptTokens     int64   `json:"prompt_tokens"`
	CompletionTokens int64   `json:"completion_tokens"`
	TotalTokens      int64   `json:"total_tokens"`
	CostEstimate     float64 `json:"cost_estimate"`
}

// Aggregate groups usage by user+model (the core "who used what" query).
func (r *Repo) Aggregate(f Filter) ([]Bucket, error) {
	var out []Bucket
	err := r.scope(f).
		Select(`user, model,
			COUNT(*) as requests,
			COALESCE(SUM(prompt_tokens),0) as prompt_tokens,
			COALESCE(SUM(completion_tokens),0) as completion_tokens,
			COALESCE(SUM(total_tokens),0) as total_tokens,
			COALESCE(SUM(cost_estimate),0) as cost_estimate`).
		Group("user, model").
		Order("total_tokens DESC").
		Scan(&out).Error
	return out, err
}

// Recent returns the latest raw records (for drill-down), capped by limit.
func (r *Repo) Recent(f Filter, limit int) ([]Record, error) {
	if limit <= 0 || limit > 500 {
		limit = 100
	}
	var out []Record
	err := r.scope(f).Order("created_at DESC").Limit(limit).Find(&out).Error
	return out, err
}
