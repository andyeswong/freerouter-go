// FreeRouter-Go — data-driven, OpenAI-compatible LLM router with per-dev tokens.
// Build static: CGO_ENABLED=0 go build -o freerouter .
package main

import (
	"log"

	"github.com/glebarez/sqlite" // pure-Go sqlite driver (no CGO)
	"gorm.io/gorm"

	"os"

	"github.com/andyeswong/freerouter-go/internal/auth"
	"github.com/andyeswong/freerouter-go/internal/config"
	"github.com/andyeswong/freerouter-go/internal/models"
	"github.com/andyeswong/freerouter-go/internal/providers"
	"github.com/andyeswong/freerouter-go/internal/quota"
	"github.com/andyeswong/freerouter-go/internal/router"
	"github.com/andyeswong/freerouter-go/internal/secrets"
	"github.com/andyeswong/freerouter-go/internal/server"
	"github.com/andyeswong/freerouter-go/internal/usage"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("config: %v", err)
	}

	db, err := gorm.Open(sqlite.Open(cfg.DBPath), &gorm.Config{})
	if err != nil {
		log.Fatalf("db: %v", err)
	}
	// WAL lets readers (router candidate lookups, secret resolution) proceed
	// without waiting on an in-flight writer. busy_timeout makes any remaining
	// writer-vs-writer overlap wait instead of failing with SQLITE_BUSY.
	// A single pooled connection means this process never contends with itself.
	if err := db.Exec("PRAGMA journal_mode=WAL;").Error; err != nil {
		log.Fatalf("db pragma journal_mode: %v", err)
	}
	if err := db.Exec("PRAGMA busy_timeout=5000;").Error; err != nil {
		log.Fatalf("db pragma busy_timeout: %v", err)
	}
	if err := db.Exec("PRAGMA synchronous=NORMAL;").Error; err != nil {
		log.Fatalf("db pragma synchronous: %v", err)
	}
	if sqlDB, err := db.DB(); err == nil {
		sqlDB.SetMaxOpenConns(1)
	} else {
		log.Fatalf("db pool: %v", err)
	}

	repo := models.NewRepo(db)
	tokens := auth.NewRepo(db)
	usageRepo := usage.NewRepo(db)
	secretsRepo := secrets.NewRepo(db)
	for _, m := range []func() error{repo.AutoMigrate, tokens.AutoMigrate, usageRepo.AutoMigrate, secretsRepo.AutoMigrate} {
		if err := m(); err != nil {
			log.Fatalf("migrate: %v", err)
		}
	}

	// Resolve provider keys from the DB secret store first, then env vars.
	providers.KeyResolver = func(ref string) string {
		if v, ok := secretsRepo.Get(ref); ok {
			return v
		}
		return os.Getenv(ref)
	}

	if cfg.AdminToken == "" {
		log.Print("WARNING: admin token unset — /admin/* is OPEN. Set FRGO_ADMIN_TOKEN.")
	}

	// Per-token consumption limits (opt-in, 0 = unlimited). Counters live in
	// memory and hydrate from the usage table, so the hot path stays read-free.
	loc, tzErr := cfg.QuotaLocation()
	if tzErr != nil {
		log.Printf("WARNING: quota timezone %q unavailable (%v) — falling back to UTC", cfg.QuotaTimezone, tzErr)
	}
	quotaTracker := quota.NewTracker(usageRepo, loc)

	rt := router.New(repo, cfg.Heuristic)
	srv := server.New(repo, rt, tokens, usageRepo, secretsRepo, quotaTracker, cfg.AdminToken)

	log.Printf("FreeRouter-Go listening on %s (db=%s, quota_tz=%s)", cfg.Listen, cfg.DBPath, loc)
	if err := srv.Engine().Run(cfg.Listen); err != nil {
		log.Fatal(err)
	}
}
