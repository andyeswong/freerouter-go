// Package secrets is a DB-backed store for provider API keys, so an operator
// can add/rotate keys via the API (and dashboard) without editing the VM's
// .env or restarting the service. A model's api_key_ref resolves to a secret
// here first, then falls back to an environment variable of the same name.
package secrets

import (
	"time"

	"gorm.io/gorm"
)

// Secret is one named credential (e.g. "DEEPSEEK_KEY" -> "sk-...").
type Secret struct {
	ID        uint      `gorm:"primaryKey" json:"id"`
	Name      string    `gorm:"uniqueIndex;not null" json:"name"`
	Value     string    `json:"-"` // never serialized
	UpdatedAt time.Time `json:"updated_at"`
}

// Info is the masked view returned to clients (value never leaves the server).
type Info struct {
	Name      string    `json:"name"`
	Preview   string    `json:"preview"` // first chars + ellipsis
	UpdatedAt time.Time `json:"updated_at"`
}

type Repo struct{ db *gorm.DB }

func NewRepo(db *gorm.DB) *Repo { return &Repo{db: db} }

func (r *Repo) AutoMigrate() error { return r.db.AutoMigrate(&Secret{}) }

// Set upserts a secret by name.
func (r *Repo) Set(name, value string) error {
	var s Secret
	err := r.db.Where("name = ?", name).First(&s).Error
	if err == gorm.ErrRecordNotFound {
		return r.db.Create(&Secret{Name: name, Value: value}).Error
	}
	if err != nil {
		return err
	}
	s.Value = value
	return r.db.Save(&s).Error
}

// Get returns the value and whether it exists.
func (r *Repo) Get(name string) (string, bool) {
	var s Secret
	if r.db.Where("name = ?", name).First(&s).Error != nil {
		return "", false
	}
	return s.Value, true
}

func (r *Repo) List() ([]Info, error) {
	var rows []Secret
	if err := r.db.Order("name ASC").Find(&rows).Error; err != nil {
		return nil, err
	}
	out := make([]Info, 0, len(rows))
	for _, s := range rows {
		out = append(out, Info{Name: s.Name, Preview: mask(s.Value), UpdatedAt: s.UpdatedAt})
	}
	return out, nil
}

func (r *Repo) Delete(name string) error {
	return r.db.Where("name = ?", name).Delete(&Secret{}).Error
}

func mask(v string) string {
	if len(v) <= 6 {
		return "••••"
	}
	return v[:4] + "…" + v[len(v)-2:]
}
