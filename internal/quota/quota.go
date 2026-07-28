// Package quota enforces optional per-token consumption limits, measured in
// TOTAL TOKENS over calendar windows (day / ISO week / month) in a configured
// timezone. A limit of 0 means "unlimited" — quotas are opt-in per token.
//
// Why counters live in memory: the limit check runs on the hot path of every
// chat request. Issuing a SUM() against SQLite there would put a read in front
// of every proxied call and re-create the write-contention stall that caused
// the 502s of 2026-07-27. Instead the tracker keeps the running total per
// (token, window) in memory and only touches the DB when a window is first
// seen — at startup, on a token's first request, or when a calendar boundary
// rolls over. FreeRouter is a single process, so in-memory totals are
// authoritative between hydrations.
package quota

import (
	"fmt"
	"sync"
	"time"
)

// Window is one of the three calendar buckets a limit can apply to.
type Window int

const (
	Daily Window = iota
	Weekly
	Monthly
	windowCount
)

func (w Window) String() string {
	switch w {
	case Daily:
		return "daily"
	case Weekly:
		return "weekly"
	case Monthly:
		return "monthly"
	}
	return "unknown"
}

// Limits is a token's configured ceilings in total tokens. 0 = unlimited.
type Limits struct {
	Daily   int64
	Weekly  int64
	Monthly int64
}

func (l Limits) get(w Window) int64 {
	switch w {
	case Daily:
		return l.Daily
	case Weekly:
		return l.Weekly
	case Monthly:
		return l.Monthly
	}
	return 0
}

// IsZero reports whether no limit at all is configured (the common case).
func (l Limits) IsZero() bool { return l.Daily == 0 && l.Weekly == 0 && l.Monthly == 0 }

// Summer reads authoritative consumption from storage. Implemented by usage.Repo.
type Summer interface {
	SumTokensSince(tokenID uint, since time.Time) (int64, error)
}

// counter is one window's running total for one token.
type counter struct {
	key      string // identifies the calendar window ("2026-07-27", "2026-W31", "2026-07")
	resetsAt time.Time
	used     int64
}

type entry struct{ w [windowCount]counter }

// Tracker holds live per-token consumption. Safe for concurrent use.
type Tracker struct {
	mu      sync.Mutex
	src     Summer
	loc     *time.Location
	byToken map[uint]*entry
}

func NewTracker(src Summer, loc *time.Location) *Tracker {
	if loc == nil {
		loc = time.UTC
	}
	return &Tracker{src: src, loc: loc, byToken: make(map[uint]*entry)}
}

// Location is the timezone calendar windows are computed in.
func (t *Tracker) Location() *time.Location { return t.loc }

// windowBounds returns the current window's start, reset instant, and key.
func (t *Tracker) windowBounds(w Window, now time.Time) (start, resetsAt time.Time, key string) {
	n := now.In(t.loc)
	midnight := time.Date(n.Year(), n.Month(), n.Day(), 0, 0, 0, 0, t.loc)
	switch w {
	case Daily:
		start = midnight
		resetsAt = start.AddDate(0, 0, 1)
		key = start.Format("2006-01-02")
	case Weekly:
		// ISO week: Monday is day 1. Go's Sunday==0 needs remapping to 7.
		wd := int(n.Weekday())
		if wd == 0 {
			wd = 7
		}
		start = midnight.AddDate(0, 0, -(wd - 1))
		resetsAt = start.AddDate(0, 0, 7)
		y, week := start.ISOWeek()
		key = fmt.Sprintf("%04d-W%02d", y, week)
	default: // Monthly
		start = time.Date(n.Year(), n.Month(), 1, 0, 0, 0, 0, t.loc)
		resetsAt = start.AddDate(0, 1, 0)
		key = start.Format("2006-01")
	}
	return start, resetsAt, key
}

// ensure refreshes any window whose calendar bucket has rolled over (or was
// never loaded) from the authoritative store. Caller must hold t.mu.
func (t *Tracker) ensure(tokenID uint, now time.Time) *entry {
	e, ok := t.byToken[tokenID]
	if !ok {
		e = &entry{}
		t.byToken[tokenID] = e
	}
	for w := Window(0); w < windowCount; w++ {
		start, resetsAt, key := t.windowBounds(w, now)
		if e.w[w].key == key {
			continue
		}
		used := int64(0)
		if t.src != nil {
			if v, err := t.src.SumTokensSince(tokenID, start); err == nil {
				used = v
			}
			// On a read error the window starts at 0: a transient DB hiccup
			// must never lock a dev out, and the next boundary re-syncs.
		}
		e.w[w] = counter{key: key, resetsAt: resetsAt, used: used}
	}
	return e
}

// Violation describes the window that blocked a request.
type Violation struct {
	Window   Window
	Used     int64
	Limit    int64
	ResetsAt time.Time
}

// RetryAfter is the whole seconds until the offending window resets (min 1).
func (v Violation) RetryAfter(now time.Time) int {
	d := int(v.ResetsAt.Sub(now).Seconds())
	if d < 1 {
		return 1
	}
	return d
}

// Exceeded returns the first window whose limit is spent, or nil when the
// token may proceed. Windows are checked narrowest-first so the error names
// the most immediate ceiling.
func (t *Tracker) Exceeded(tokenID uint, lim Limits) *Violation {
	if lim.IsZero() {
		return nil
	}
	now := time.Now()
	t.mu.Lock()
	defer t.mu.Unlock()
	e := t.ensure(tokenID, now)
	for w := Window(0); w < windowCount; w++ {
		max := lim.get(w)
		if max <= 0 {
			continue
		}
		if e.w[w].used >= max {
			return &Violation{Window: w, Used: e.w[w].used, Limit: max, ResetsAt: e.w[w].resetsAt}
		}
	}
	return nil
}

// Add books consumption against every window.
//
// Call this BEFORE the usage row is inserted. Hydration reads the DB, so a
// hydration that races an Add can only miss a row still in flight (a tiny
// undercount that the next boundary corrects) — whereas adding after the
// INSERT would let a racing hydration count the same tokens twice.
func (t *Tracker) Add(tokenID uint, tokens int64) {
	if tokens <= 0 {
		return
	}
	now := time.Now()
	t.mu.Lock()
	defer t.mu.Unlock()
	e := t.ensure(tokenID, now)
	for w := Window(0); w < windowCount; w++ {
		e.w[w].used += tokens
	}
}

// WindowState is one window's live state, for the admin API.
type WindowState struct {
	Window   string    `json:"window"`
	Used     int64     `json:"used"`
	Limit    int64     `json:"limit"` // 0 = unlimited
	Exceeded bool      `json:"exceeded"`
	ResetsAt time.Time `json:"resets_at"`
}

// Snapshot reports current consumption per window for one token.
func (t *Tracker) Snapshot(tokenID uint, lim Limits) []WindowState {
	now := time.Now()
	t.mu.Lock()
	defer t.mu.Unlock()
	e := t.ensure(tokenID, now)
	out := make([]WindowState, 0, windowCount)
	for w := Window(0); w < windowCount; w++ {
		max := lim.get(w)
		out = append(out, WindowState{
			Window:   w.String(),
			Used:     e.w[w].used,
			Limit:    max,
			Exceeded: max > 0 && e.w[w].used >= max,
			ResetsAt: e.w[w].resetsAt,
		})
	}
	return out
}

// Forget drops a token's cached counters, forcing a re-read on next use.
// Used after a limit change so the admin sees fresh numbers immediately.
func (t *Tracker) Forget(tokenID uint) {
	t.mu.Lock()
	delete(t.byToken, tokenID)
	t.mu.Unlock()
}
