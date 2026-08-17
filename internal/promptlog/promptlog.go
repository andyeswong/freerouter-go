// Package promptlog captures, as plain text, the prompts consumers send through
// the router. It exists because usage records answer "how many tokens" but never
// "what was in them" — and when a token burns millions of tokens on traffic no
// human is behind (an agent heartbeat, a stuck daemon), the prompt itself is the
// only thing that identifies the source.
//
// Off by default. The capture holds whatever the consumer sent — client data,
// credentials pasted into a chat, internal context — so the file is created 0600
// and the operator opts in per token.
package promptlog

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// Config is the `prompt_log` block of freerouter.config.json.
type Config struct {
	Enabled bool   `json:"enabled"`
	Path    string `json:"path"` // plain-text log; default prompts.txt next to the DB

	// Tokens restricts capture to these token names (as shown by /admin/tokens).
	// Empty means every token — rarely what you want on a shared router.
	Tokens []string `json:"tokens"`

	// MaxChars truncates each message; 0 keeps it whole. A truncated message
	// still records its real length, so the size pattern stays readable.
	MaxChars int `json:"max_chars"`

	// MaxBytes rotates the file to <path>.1 once it grows past this; 0 never
	// rotates. The router's VM has a 20GB disk — an unbounded capture on a
	// chatty token fills it.
	MaxBytes int64 `json:"max_bytes"`
}

// Message is one turn as the router parsed it (content already flattened from
// whatever block shape the client used).
type Message struct {
	Role    string
	Content string
}

// Entry is a single captured request: the routing verdict plus what was sent.
type Entry struct {
	Token        string
	TokenID      uint
	Model        string
	Tier         int
	Method       string
	Stream       bool
	Tools        bool
	ContextChars int
	Messages     []Message
}

// Logger serializes writes to the capture file. All exported methods are safe
// for concurrent use; a disabled logger costs one mutex-free atomic read.
type Logger struct {
	mu       sync.Mutex
	cfg      Config
	tokens   map[string]bool // nil = capture every token
	f        *os.File
	size     int64
	failed   bool // stop re-reporting the same open error every request
	nowFn    func() time.Time
	openFile func(string) (*os.File, error)
}

// New builds a Logger from config. A configured-on logger opens its file
// immediately so a bad path surfaces at boot, not on the first request.
func New(cfg Config) *Logger {
	l := &Logger{
		nowFn:    time.Now,
		openFile: openAppend,
	}
	l.Configure(cfg)
	return l
}

func openAppend(path string) (*os.File, error) {
	if dir := filepath.Dir(path); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			return nil, err
		}
	}
	// 0600: the capture is prompt content, readable only by the router's user.
	return os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
}

// Configure replaces the whole config (used at boot and by the admin endpoint).
// Returns the error from opening the file when enabling, so the caller can
// report it instead of silently capturing nothing.
func (l *Logger) Configure(cfg Config) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	if cfg.Path == "" {
		cfg.Path = "prompts.txt"
	}
	l.cfg = cfg
	l.tokens = nil
	if len(cfg.Tokens) > 0 {
		l.tokens = make(map[string]bool, len(cfg.Tokens))
		for _, name := range cfg.Tokens {
			l.tokens[strings.TrimSpace(name)] = true
		}
	}

	l.closeLocked()
	l.failed = false
	if !cfg.Enabled {
		return nil
	}
	return l.openLocked()
}

// SetEnabled flips capture on or off, leaving the rest of the config alone.
func (l *Logger) SetEnabled(on bool) error {
	l.mu.Lock()
	cfg := l.cfg
	l.mu.Unlock()
	cfg.Enabled = on
	return l.Configure(cfg)
}

// Config returns the current settings (for the admin GET).
func (l *Logger) Config() Config {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.cfg
}

// Size reports the bytes captured so far, so an operator can see the file
// growing without shelling into the host. -1 when capture is off.
func (l *Logger) Size() int64 {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.f == nil {
		return -1
	}
	return l.size
}

func (l *Logger) openLocked() error {
	f, err := l.openFile(l.cfg.Path)
	if err != nil {
		l.failed = true
		return err
	}
	l.f = f
	l.size = 0
	if st, err := f.Stat(); err == nil {
		l.size = st.Size()
	}
	return nil
}

func (l *Logger) closeLocked() {
	if l.f != nil {
		_ = l.f.Close()
		l.f = nil
		l.size = 0
	}
}

// captures reports whether this token's prompts should be written.
func (l *Logger) captures(token string) bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	if !l.cfg.Enabled || l.f == nil {
		return false
	}
	if l.tokens == nil {
		return true
	}
	return l.tokens[token]
}

// Log appends one entry. It is a no-op when capture is off or the entry's token
// is not in the allowlist, and it never returns an error to the request path —
// a broken capture must not break proxying.
func (l *Logger) Log(e Entry) {
	if l == nil || !l.captures(e.Token) {
		return
	}
	text := l.render(e)

	l.mu.Lock()
	defer l.mu.Unlock()
	if l.f == nil {
		return // disabled between the check and here
	}
	l.rotateLocked(int64(len(text)))
	if l.f == nil {
		return // rotation failed and disabled capture
	}
	n, err := l.f.WriteString(text)
	l.size += int64(n)
	if err != nil && !l.failed {
		l.failed = true
	}
}

// rotateLocked moves the current file aside once it would exceed MaxBytes,
// keeping exactly one previous generation.
func (l *Logger) rotateLocked(incoming int64) {
	if l.cfg.MaxBytes <= 0 || l.size+incoming <= l.cfg.MaxBytes {
		return
	}
	l.closeLocked()
	if err := os.Rename(l.cfg.Path, l.cfg.Path+".1"); err != nil && !os.IsNotExist(err) {
		l.failed = true
		return
	}
	_ = l.openLocked()
}

func (l *Logger) render(e Entry) string {
	l.mu.Lock()
	maxChars := l.cfg.MaxChars
	l.mu.Unlock()

	var b strings.Builder
	fmt.Fprintf(&b, "===== %s token=%s(%d) model=%s tier=%d method=%s stream=%t tools=%t ctx_chars=%d msgs=%d\n",
		l.nowFn().UTC().Format(time.RFC3339Nano), e.Token, e.TokenID, e.Model, e.Tier,
		e.Method, e.Stream, e.Tools, e.ContextChars, len(e.Messages))

	for i, m := range e.Messages {
		content := m.Content
		full := len(content)
		truncated := false
		if maxChars > 0 && full > maxChars {
			content = content[:maxChars]
			truncated = true
		}
		fmt.Fprintf(&b, "--- [%d] %s (%d chars) ---\n", i+1, m.Role, full)
		b.WriteString(content)
		if truncated {
			fmt.Fprintf(&b, "\n... [truncated %d chars]", full-maxChars)
		}
		b.WriteString("\n")
	}
	b.WriteString("=====\n\n")
	return b.String()
}
