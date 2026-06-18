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

// Verify resolves a plaintext bearer to an enabled token, touching last_used_at.
func (r *Repo) Verify(plain string) (*ApiToken, bool) {
	var tok ApiToken
	err := r.db.Where("token_hash = ? AND enabled = ?", hashToken(plain), true).First(&tok).Error
	if err != nil {
		return nil, false
	}
	now := time.Now()
	r.db.Model(&tok).Update("last_used_at", &now)
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
