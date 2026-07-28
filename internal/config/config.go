// Package config loads runtime config from an external JSON file so models and
// routing knobs can change without a rebuild/restart (estilo cc_bridge).
// Path resolution: FRGO_CONFIG_PATH env, else ./freerouter.config.json.
package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"time"

	// Embed the IANA database: the binary is built static (CGO_ENABLED=0) and
	// may run on a host without tzdata, where LoadLocation would silently
	// leave quota windows on UTC.
	_ "time/tzdata"

	"github.com/andyeswong/freerouter-go/internal/router"
)

type Config struct {
	Listen     string                 `json:"listen"`      // e.g. ":8080"
	DBPath     string                 `json:"db_path"`     // sqlite file
	AdminToken string                 `json:"admin_token"` // gates /admin/*; env FRGO_ADMIN_TOKEN overrides
	Heuristic  router.HeuristicConfig `json:"heuristic"`

	// QuotaTimezone decides when a token's daily/weekly/monthly budget rolls
	// over. IANA name; env FRGO_QUOTA_TZ overrides. Defaults to Pacific
	// (Tijuana/Las Vegas) so windows line up with the operators' working day
	// rather than UTC midnight.
	QuotaTimezone string `json:"quota_timezone"`
}

func defaults() Config {
	return Config{
		Listen:        ":8080",
		DBPath:        "freerouter.db",
		QuotaTimezone: "America/Tijuana",
		Heuristic: router.HeuristicConfig{
			SimpleKeywords:    []string{"hi", "hello", "thanks", "yes", "no", "ok"},
			ReasoningKeywords: []string{"prove", "derive", "analyze", "explain why", "reason", "step by step"},
			CodeKeywords:      []string{"function", "code", "bug", "refactor", "compile", "regex", "sql"},
			AgenticKeywords:   []string{"run", "execute", "ssh", "deploy", "ping", "curl", "shell", "browse"},
			SimpleMedium:      0.0,
			MediumComplex:     1.0,
			ComplexReason:     2.0,
		},
	}
}

func Path() string {
	if p := os.Getenv("FRGO_CONFIG_PATH"); p != "" {
		return p
	}
	return "freerouter.config.json"
}

// Load reads the config file, falling back to defaults for missing fields.
func Load() (Config, error) {
	cfg := defaults()
	p := Path()
	b, err := os.ReadFile(p)
	if os.IsNotExist(err) {
		return cfg, nil // run on defaults; admin can seed models via API
	}
	if err != nil {
		return cfg, err
	}
	if err := json.Unmarshal(b, &cfg); err != nil {
		return cfg, err
	}
	if cfg.DBPath != "" && !filepath.IsAbs(cfg.DBPath) {
		// keep relative to cwd; explicit for clarity
		cfg.DBPath = filepath.Clean(cfg.DBPath)
	}
	if env := os.Getenv("FRGO_ADMIN_TOKEN"); env != "" {
		cfg.AdminToken = env
	}
	if env := os.Getenv("FRGO_QUOTA_TZ"); env != "" {
		cfg.QuotaTimezone = env
	}
	if cfg.QuotaTimezone == "" {
		cfg.QuotaTimezone = defaults().QuotaTimezone
	}
	return cfg, nil
}

// QuotaLocation resolves the configured timezone, falling back to UTC (with
// the error, so the caller can warn) when the name can't be loaded.
func (c Config) QuotaLocation() (*time.Location, error) {
	loc, err := time.LoadLocation(c.QuotaTimezone)
	if err != nil {
		return time.UTC, err
	}
	return loc, nil
}
