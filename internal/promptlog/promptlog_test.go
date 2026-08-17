package promptlog

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func entry(token, content string) Entry {
	return Entry{
		Token:    token,
		TokenID:  10,
		Model:    "minimax",
		Tier:     4,
		Method:   "heuristic",
		Messages: []Message{{Role: "user", Content: content}},
	}
}

func read(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return string(b)
}

func TestDisabledWritesNothing(t *testing.T) {
	path := filepath.Join(t.TempDir(), "prompts.txt")
	l := New(Config{Enabled: false, Path: path})
	l.Log(entry("Olazo", "secreto"))

	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("capture off must not create the file, got err=%v", err)
	}
	if l.Size() != -1 {
		t.Fatalf("Size() on a disabled logger = %d, want -1", l.Size())
	}
}

func TestEnabledCapturesPromptAndMetadata(t *testing.T) {
	path := filepath.Join(t.TempDir(), "prompts.txt")
	l := New(Config{Enabled: true, Path: path})
	l.Log(entry("Olazo", "revisa el estado del servidor"))

	got := read(t, path)
	for _, want := range []string{"token=Olazo(10)", "model=minimax", "tier=4", "revisa el estado del servidor"} {
		if !strings.Contains(got, want) {
			t.Errorf("capture missing %q\n--- got ---\n%s", want, got)
		}
	}
}

// The file holds prompt content, so it must not be group/world readable.
func TestFileIsPrivate(t *testing.T) {
	path := filepath.Join(t.TempDir(), "prompts.txt")
	l := New(Config{Enabled: true, Path: path})
	l.Log(entry("Olazo", "hola"))

	st, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if perm := st.Mode().Perm(); perm != 0o600 {
		t.Fatalf("mode = %o, want 600", perm)
	}
}

func TestTokenAllowlist(t *testing.T) {
	path := filepath.Join(t.TempDir(), "prompts.txt")
	l := New(Config{Enabled: true, Path: path, Tokens: []string{"Olazo"}})
	l.Log(entry("Olazo", "capturame"))
	l.Log(entry("maia", "a mi no"))

	got := read(t, path)
	if !strings.Contains(got, "capturame") {
		t.Error("allowlisted token was not captured")
	}
	if strings.Contains(got, "a mi no") {
		t.Error("token outside the allowlist leaked into the capture")
	}
}

func TestEmptyAllowlistCapturesEveryToken(t *testing.T) {
	path := filepath.Join(t.TempDir(), "prompts.txt")
	l := New(Config{Enabled: true, Path: path})
	l.Log(entry("Olazo", "uno"))
	l.Log(entry("maia", "dos"))

	got := read(t, path)
	if !strings.Contains(got, "uno") || !strings.Contains(got, "dos") {
		t.Errorf("empty allowlist should capture everyone, got:\n%s", got)
	}
}

func TestTruncationKeepsRealLength(t *testing.T) {
	path := filepath.Join(t.TempDir(), "prompts.txt")
	l := New(Config{Enabled: true, Path: path, MaxChars: 10})
	l.Log(entry("Olazo", strings.Repeat("x", 100)))

	got := read(t, path)
	if !strings.Contains(got, "(100 chars)") {
		t.Error("truncated message must still record its real length")
	}
	if !strings.Contains(got, "[truncated 90 chars]") {
		t.Errorf("missing truncation marker:\n%s", got)
	}
	if strings.Contains(got, strings.Repeat("x", 11)) {
		t.Error("content was not truncated to max_chars")
	}
}

func TestRotationKeepsOneGeneration(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "prompts.txt")
	l := New(Config{Enabled: true, Path: path, MaxBytes: 200})
	for i := 0; i < 8; i++ {
		l.Log(entry("Olazo", strings.Repeat("y", 100)))
	}

	if _, err := os.Stat(path + ".1"); err != nil {
		t.Fatalf("expected rotated file: %v", err)
	}
	if sz := l.Size(); sz > 400 {
		t.Errorf("live file kept growing past the rotation point: %d bytes", sz)
	}
}

func TestSetEnabledTogglesLive(t *testing.T) {
	path := filepath.Join(t.TempDir(), "prompts.txt")
	l := New(Config{Enabled: false, Path: path})

	l.Log(entry("Olazo", "antes"))
	if err := l.SetEnabled(true); err != nil {
		t.Fatal(err)
	}
	l.Log(entry("Olazo", "durante"))
	if err := l.SetEnabled(false); err != nil {
		t.Fatal(err)
	}
	l.Log(entry("Olazo", "despues"))

	got := read(t, path)
	if strings.Contains(got, "antes") || strings.Contains(got, "despues") {
		t.Errorf("captured while disabled:\n%s", got)
	}
	if !strings.Contains(got, "durante") {
		t.Errorf("missed the capture while enabled:\n%s", got)
	}
	if l.Config().Path != path {
		t.Error("SetEnabled must not disturb the rest of the config")
	}
}

func TestEnableReportsBadPath(t *testing.T) {
	// A path whose parent is a file, not a directory: MkdirAll must fail.
	dir := t.TempDir()
	blocker := filepath.Join(dir, "notadir")
	if err := os.WriteFile(blocker, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	l := New(Config{Enabled: false, Path: filepath.Join(blocker, "prompts.txt")})
	if err := l.SetEnabled(true); err == nil {
		t.Fatal("enabling with an unusable path must return an error")
	}
	l.Log(entry("Olazo", "no debe explotar"))
}

func TestNilLoggerIsSafe(t *testing.T) {
	var l *Logger
	l.Log(entry("Olazo", "sin logger"))
}
