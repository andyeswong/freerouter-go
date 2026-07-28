// Package auth issues and verifies per-dev API tokens so consumers hit
// FreeRouter with a single token instead of holding every provider key.
package auth

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"time"

	"gorm.io/gorm"
)

// ApiToken is one credential handed to a dev/service. The plaintext token is
// shown ONCE at creation; only its sha256 hash is persisted.
type ApiToken struct {
	ID        uint       `gorm:"primaryKey" json:"id"`
	Name      string     `gorm:"index;not null" json:"name"` // dev/user identity, e.g. "gerardo"
	TokenHash string     `gorm:"uniqueIndex;not null" json:"-"`
	Prefix    string     `json:"prefix"` // first chars, for display ("frgo_a1b2…")
	Enabled   bool       `gorm:"index" json:"enabled"`
	LastUsedAt *time.Time `json:"last_used_at"`
	CreatedAt time.Time  `json:"created_at"`

	// Optional consumption ceilings, in TOTAL tokens (prompt + completion) per
	// calendar window. 0 = unlimited, which is the default for every token.
	// Enforced by internal/quota; windows reset on the calendar boundary in the
	// configured timezone, not on a rolling basis.
	LimitDailyTokens   int64 `gorm:"default:0" json:"limit_daily_tokens"`
	LimitWeeklyTokens  int64 `gorm:"default:0" json:"limit_weekly_tokens"`
	LimitMonthlyTokens int64 `gorm:"default:0" json:"limit_monthly_tokens"`
}

// hashToken returns the hex sha256 of a plaintext token.
func hashToken(plain string) string {
	sum := sha256.Sum256([]byte(plain))
	return hex.EncodeToString(sum[:])
}

// generatePlaintext returns a new token like "frgo_<32 hex>".
func generatePlaintext() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	return "frgo_" + hex.EncodeToString(b)
}

// Repo wraps token persistence.
type Repo struct{ db *gorm.DB }

func NewRepo(db *gorm.DB) *Repo { return &Repo{db: db} }

func (r *Repo) AutoMigrate() error { return r.db.AutoMigrate(&ApiToken{}) }

// Issue creates a token for name and returns the row plus the ONE-TIME plaintext.
func (r *Repo) Issue(name string) (*ApiToken, string, error) {
	plain := generatePlaintext()
	tok := &ApiToken{
		Name:      name,
		TokenHash: hashToken(plain),
		Prefix:    plain[:13], // "frgo_" + 8 hex
		Enabled:   true,
	}
	if err := r.db.Create(tok).Error; err != nil {
		return nil, "", err
	}
	return tok, plain, nil
}

// Verify resolves a plaintext bearer to an enabled token. last_used_at is
// touched asynchronously — a busy writer elsewhere (e.g. another dev's usage
// insert) must never make an unrelated request wait on this bookkeeping write.
func (r *Repo) Verify(plain string) (*ApiToken, bool) {
	var tok ApiToken
	err := r.db.Where("token_hash = ? AND enabled = ?", hashToken(plain), true).First(&tok).Error
	if err != nil {
		return nil, false
	}
	id := tok.ID
	go func() {
		now := time.Now()
		r.db.Model(&ApiToken{}).Where("id = ?", id).Update("last_used_at", &now)
	}()
	return &tok, true
}

func (r *Repo) List() ([]ApiToken, error) {
	var out []ApiToken
	return out, r.db.Order("created_at DESC").Find(&out).Error
}

// SetEnabled flips a token on/off (revoke = false).
func (r *Repo) SetEnabled(id uint, enabled bool) error {
	return r.db.Model(&ApiToken{}).Where("id = ?", id).Update("enabled", enabled).Error
}

// Get loads one token by id.
func (r *Repo) Get(id uint) (*ApiToken, error) {
	var tok ApiToken
	return &tok, r.db.First(&tok, id).Error
}

// SetLimits updates the per-window token ceilings. A nil pointer leaves that
// window untouched; 0 clears the limit (unlimited).
func (r *Repo) SetLimits(id uint, daily, weekly, monthly *int64) error {
	fields := map[string]any{}
	if daily != nil {
		fields["limit_daily_tokens"] = *daily
	}
	if weekly != nil {
		fields["limit_weekly_tokens"] = *weekly
	}
	if monthly != nil {
		fields["limit_monthly_tokens"] = *monthly
	}
	if len(fields) == 0 {
		return nil
	}
	return r.db.Model(&ApiToken{}).Where("id = ?", id).Updates(fields).Error
}
