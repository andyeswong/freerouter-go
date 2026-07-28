package quota

import (
	"sync"
	"testing"
	"time"
)

// fakeSummer stands in for usage.Repo, counting how often it is asked.
type fakeSummer struct {
	mu    sync.Mutex
	total int64
	calls int
}

func (f *fakeSummer) SumTokensSince(tokenID uint, since time.Time) (int64, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls++
	return f.total, nil
}

func mustLoc(t *testing.T, name string) *time.Location {
	t.Helper()
	loc, err := time.LoadLocation(name)
	if err != nil {
		t.Fatalf("load %s: %v", name, err)
	}
	return loc
}

func TestWindowBoundsCalendar(t *testing.T) {
	loc := mustLoc(t, "America/Tijuana")
	tr := NewTracker(nil, loc)

	// A Sunday: the ISO week must still start on the PREVIOUS Monday, which is
	// the case Go's Weekday()==0 gets wrong without remapping.
	sunday := time.Date(2026, 7, 26, 15, 30, 0, 0, loc)
	cases := []struct {
		w             Window
		wantStart     time.Time
		wantResetsAt  time.Time
		wantKeyPrefix string
	}{
		{Daily,
			time.Date(2026, 7, 26, 0, 0, 0, 0, loc),
			time.Date(2026, 7, 27, 0, 0, 0, 0, loc), "2026-07-26"},
		{Weekly,
			time.Date(2026, 7, 20, 0, 0, 0, 0, loc), // Monday
			time.Date(2026, 7, 27, 0, 0, 0, 0, loc), "2026-W30"},
		{Monthly,
			time.Date(2026, 7, 1, 0, 0, 0, 0, loc),
			time.Date(2026, 8, 1, 0, 0, 0, 0, loc), "2026-07"},
	}
	for _, tc := range cases {
		start, reset, key := tr.windowBounds(tc.w, sunday)
		if !start.Equal(tc.wantStart) {
			t.Errorf("%s start = %s, want %s", tc.w, start, tc.wantStart)
		}
		if !reset.Equal(tc.wantResetsAt) {
			t.Errorf("%s resetsAt = %s, want %s", tc.w, reset, tc.wantResetsAt)
		}
		if key != tc.wantKeyPrefix {
			t.Errorf("%s key = %s, want %s", tc.w, key, tc.wantKeyPrefix)
		}
	}
}

func TestHydratesOnceThenCountsInMemory(t *testing.T) {
	src := &fakeSummer{total: 300}
	tr := NewTracker(src, time.UTC)
	lim := Limits{Daily: 1000}

	if v := tr.Exceeded(1, lim); v != nil {
		t.Fatalf("300/1000 should pass, got violation %+v", v)
	}
	// Three windows hydrate on first touch; nothing after that.
	if src.calls != 3 {
		t.Fatalf("hydration calls = %d, want 3", src.calls)
	}
	tr.Add(1, 800)
	if src.calls != 3 {
		t.Errorf("Add must not hit the DB; calls = %d", src.calls)
	}
	v := tr.Exceeded(1, lim)
	if v == nil {
		t.Fatal("1100/1000 should be blocked")
	}
	if v.Window != Daily || v.Used != 1100 || v.Limit != 1000 {
		t.Errorf("violation = %+v, want daily 1100/1000", v)
	}
}

func TestZeroLimitIsUnlimitedAndFree(t *testing.T) {
	src := &fakeSummer{total: 1 << 40}
	tr := NewTracker(src, time.UTC)
	if v := tr.Exceeded(1, Limits{}); v != nil {
		t.Fatalf("no limits configured must never block, got %+v", v)
	}
	// The common case must not even reach storage.
	if src.calls != 0 {
		t.Errorf("unlimited token hit the DB %d times, want 0", src.calls)
	}
}

func TestNarrowestWindowWins(t *testing.T) {
	tr := NewTracker(&fakeSummer{}, time.UTC)
	tr.Add(7, 500)
	v := tr.Exceeded(7, Limits{Daily: 400, Weekly: 450, Monthly: 100})
	if v == nil || v.Window != Daily {
		t.Fatalf("expected the daily window to be named first, got %+v", v)
	}
}

func TestConcurrentAddIsExact(t *testing.T) {
	tr := NewTracker(&fakeSummer{}, time.UTC)
	var wg sync.WaitGroup
	for i := 0; i < 200; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			tr.Add(42, 10)
		}()
	}
	wg.Wait()
	states := tr.Snapshot(42, Limits{Daily: 1})
	if states[0].Used != 2000 {
		t.Errorf("used = %d, want 2000", states[0].Used)
	}
}

func TestForgetForcesRehydration(t *testing.T) {
	src := &fakeSummer{total: 50}
	tr := NewTracker(src, time.UTC)
	tr.Snapshot(3, Limits{Daily: 100})
	before := src.calls
	tr.Forget(3)
	tr.Snapshot(3, Limits{Daily: 100})
	if src.calls != before+3 {
		t.Errorf("calls after Forget = %d, want %d", src.calls, before+3)
	}
}

func TestRetryAfterFloorsAtOneSecond(t *testing.T) {
	now := time.Now()
	v := Violation{ResetsAt: now.Add(-time.Hour)}
	if got := v.RetryAfter(now); got != 1 {
		t.Errorf("RetryAfter on a past reset = %d, want 1", got)
	}
}
