package usage

import (
	"testing"
	"time"

	"github.com/glebarez/sqlite"
	"gorm.io/gorm"
)

func newTestRepo(t *testing.T) *Repo {
	t.Helper()
	db, err := gorm.Open(sqlite.Open(t.TempDir()+"/test.db"), &gorm.Config{})
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	r := NewRepo(db)
	if err := r.AutoMigrate(); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	return r
}

// SumTokensSince must honor the OFFSET of the cutoff it is given. Quota windows
// start at local midnight (e.g. 00:00 -07:00 = 07:00 UTC); if the offset is
// dropped when the parameter is bound, the window silently shifts by the zone's
// offset and counts consumption from the previous evening.
func TestSumTokensSinceRespectsZoneOffset(t *testing.T) {
	r := newTestRepo(t)
	loc, err := time.LoadLocation("America/Tijuana")
	if err != nil {
		t.Fatalf("load zone: %v", err)
	}

	// 2026-07-28 03:00 UTC == 2026-07-27 20:00 Pacific — the evening BEFORE the
	// Pacific day being measured.
	before := time.Date(2026, 7, 28, 3, 0, 0, 0, time.UTC)
	// 2026-07-28 09:00 UTC == 2026-07-28 02:00 Pacific — inside the Pacific day.
	after := time.Date(2026, 7, 28, 9, 0, 0, 0, time.UTC)

	for _, rec := range []struct {
		at  time.Time
		tok int
	}{{before, 1000}, {after, 7}} {
		if err := r.db.Create(&Record{TokenID: 1, User: "u", Model: "m", TotalTokens: rec.tok, CreatedAt: rec.at}).Error; err != nil {
			t.Fatalf("seed: %v", err)
		}
	}

	midnightPacific := time.Date(2026, 7, 28, 0, 0, 0, 0, loc) // == 07:00 UTC
	got, err := r.SumTokensSince(1, midnightPacific)
	if err != nil {
		t.Fatalf("sum: %v", err)
	}
	if got != 7 {
		t.Errorf("SumTokensSince(midnight Pacific) = %d, want 7 — the 1000-token row from the previous Pacific evening leaked in, so the cutoff's -07:00 offset was ignored", got)
	}
}

func TestSumTokensSinceIsScopedToOneToken(t *testing.T) {
	r := newTestRepo(t)
	at := time.Now().UTC()
	r.db.Create(&Record{TokenID: 1, TotalTokens: 10, CreatedAt: at})
	r.db.Create(&Record{TokenID: 2, TotalTokens: 500, CreatedAt: at})

	got, err := r.SumTokensSince(1, at.Add(-time.Hour))
	if err != nil {
		t.Fatalf("sum: %v", err)
	}
	if got != 10 {
		t.Errorf("sum = %d, want 10 (token 2 must not be counted)", got)
	}
}
