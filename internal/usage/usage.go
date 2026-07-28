// Package usage records per-request token consumption and answers
// "which dev used how many tokens of which model".
package usage

import (
	"fmt"
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
	// Bounds are normalized to UTC: created_at is stored in UTC and the driver
	// drops the zone of a zoned parameter, which would shift the range.
	if f.From != nil {
		tx = tx.Where("created_at >= ?", f.From.UTC())
	}
	if f.To != nil {
		tx = tx.Where("created_at <= ?", f.To.UTC())
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

// SeriesPoint is one time bucket of usage, optionally split by user or model.
type SeriesPoint struct {
	Bucket string `json:"bucket"` // "2026-07-28" (day) or "2026-07-28T14" (hour), in the requested zone
	Key    string `json:"key"`    // user or model when grouped; "" otherwise

	Requests         int64   `json:"requests"`
	PromptTokens     int64   `json:"prompt_tokens"`
	CompletionTokens int64   `json:"completion_tokens"`
	TotalTokens      int64   `json:"total_tokens"`
	CostEstimate     float64 `json:"cost_estimate"`
}

// Series aggregates usage into calendar buckets, which the per-user/per-day
// figures (averages, trends, peaks) are built from. Aggregating in SQL is the
// point: the raw-record endpoint is capped, so anything longer than a couple of
// hours cannot be derived client-side.
//
// Buckets are labeled in loc, using loc's CURRENT offset. Across a DST change
// the boundary of older buckets can therefore be off by an hour — acceptable
// for usage reporting, and the alternative (per-row zone math) is not something
// SQLite can do without its own tz database.
func (r *Repo) Series(f Filter, bucket, group string, loc *time.Location) ([]SeriesPoint, error) {
	var stamp string
	switch bucket {
	case "", "day":
		stamp = "%Y-%m-%d"
	case "hour":
		stamp = "%Y-%m-%dT%H"
	default:
		return nil, fmt.Errorf("unsupported bucket %q (day|hour)", bucket)
	}

	// Whitelisted: these are interpolated into SQL, never taken raw.
	var keyExpr string
	switch group {
	case "", "none":
		keyExpr = "''"
	case "user":
		keyExpr = "user"
	case "model":
		keyExpr = "model"
	default:
		return nil, fmt.Errorf("unsupported group %q (none|user|model)", group)
	}

	if loc == nil {
		loc = time.UTC
	}
	_, offset := time.Now().In(loc).Zone()

	var out []SeriesPoint
	err := r.scope(f).
		Select(fmt.Sprintf(`strftime('%s', created_at, '%d seconds') as bucket,
			%s as key,
			COUNT(*) as requests,
			COALESCE(SUM(prompt_tokens),0) as prompt_tokens,
			COALESCE(SUM(completion_tokens),0) as completion_tokens,
			COALESCE(SUM(total_tokens),0) as total_tokens,
			COALESCE(SUM(cost_estimate),0) as cost_estimate`, stamp, offset, keyExpr)).
		Group("bucket, key").
		Order("bucket ASC, total_tokens DESC").
		Scan(&out).Error
	return out, err
}

// SumTokensSince totals one token's consumption from `since` onward. This is
// the authoritative number the quota tracker hydrates its in-memory counters
// from (at startup and on every calendar-window rollover).
//
// `since` is normalized to UTC before it is bound: created_at is stored in UTC,
// and the sqlite driver drops the zone when serializing a zoned time, so a
// cutoff like "00:00 -07:00" would otherwise compare as 00:00 UTC and shift the
// whole window by the zone's offset.
func (r *Repo) SumTokensSince(tokenID uint, since time.Time) (int64, error) {
	var row struct{ Total int64 }
	err := r.db.Model(&Record{}).
		Where("token_id = ? AND created_at >= ?", tokenID, since.UTC()).
		Select("COALESCE(SUM(total_tokens),0) as total").
		Scan(&row).Error
	return row.Total, err
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
