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

func TestSeriesBucketsByLocalDay(t *testing.T) {
	r := newTestRepo(t)
	loc, err := time.LoadLocation("America/Tijuana")
	if err != nil {
		t.Fatalf("load zone: %v", err)
	}

	// 06:00 UTC on the 28th is still the 27th in Pacific (23:00). It must land
	// in the 27th's bucket, not the 28th's.
	seed := []struct {
		at   time.Time
		user string
		tok  int
	}{
		{time.Date(2026, 7, 28, 6, 0, 0, 0, time.UTC), "ana", 100},
		{time.Date(2026, 7, 28, 9, 0, 0, 0, time.UTC), "ana", 30},
		{time.Date(2026, 7, 28, 10, 0, 0, 0, time.UTC), "beto", 7},
	}
	for _, s := range seed {
		if err := r.db.Create(&Record{TokenID: 1, User: s.user, Model: "m", TotalTokens: s.tok, CreatedAt: s.at}).Error; err != nil {
			t.Fatalf("seed: %v", err)
		}
	}

	pts, err := r.Series(Filter{}, "day", "none", loc)
	if err != nil {
		t.Fatalf("series: %v", err)
	}
	got := map[string]int64{}
	for _, p := range pts {
		got[p.Bucket] = p.TotalTokens
	}
	if got["2026-07-27"] != 100 {
		t.Errorf("bucket 2026-07-27 = %d, want 100 (06:00Z is 23:00 Pacific of the 27th)", got["2026-07-27"])
	}
	if got["2026-07-28"] != 37 {
		t.Errorf("bucket 2026-07-28 = %d, want 37", got["2026-07-28"])
	}

	byUser, err := r.Series(Filter{}, "day", "user", loc)
	if err != nil {
		t.Fatalf("series by user: %v", err)
	}
	seen := map[string]int64{}
	for _, p := range byUser {
		if p.Bucket == "2026-07-28" {
			seen[p.Key] = p.TotalTokens
		}
	}
	if seen["ana"] != 30 || seen["beto"] != 7 {
		t.Errorf("grouped by user on the 28th = %v, want ana:30 beto:7", seen)
	}
}

func TestSeriesRejectsUnknownBucketAndGroup(t *testing.T) {
	r := newTestRepo(t)
	if _, err := r.Series(Filter{}, "week", "none", time.UTC); err == nil {
		t.Error("bucket=week should be rejected, not silently accepted")
	}
	if _, err := r.Series(Filter{}, "day", "user; drop table records", time.UTC); err == nil {
		t.Error("group must be whitelisted — an arbitrary value reaches SQL otherwise")
	}
}
