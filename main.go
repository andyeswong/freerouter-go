// FreeRouter-Go — data-driven, OpenAI-compatible LLM router with per-dev tokens.
// Build static: CGO_ENABLED=0 go build -o freerouter .
package main

import (
	"log"

	"github.com/glebarez/sqlite" // pure-Go sqlite driver (no CGO)
	"gorm.io/gorm"

	"github.com/andyeswong/freerouter-go/internal/auth"
	"github.com/andyeswong/freerouter-go/internal/config"
	"github.com/andyeswong/freerouter-go/internal/models"
	"github.com/andyeswong/freerouter-go/internal/router"
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

	repo := models.NewRepo(db)
	tokens := auth.NewRepo(db)
	usageRepo := usage.NewRepo(db)
	for _, m := range []func() error{repo.AutoMigrate, tokens.AutoMigrate, usageRepo.AutoMigrate} {
		if err := m(); err != nil {
			log.Fatalf("migrate: %v", err)
		}
	}

	if cfg.AdminToken == "" {
		log.Print("WARNING: admin token unset — /admin/* is OPEN. Set FRGO_ADMIN_TOKEN.")
	}

	rt := router.New(repo, cfg.Heuristic)
	srv := server.New(repo, rt, tokens, usageRepo, cfg.AdminToken)

	log.Printf("FreeRouter-Go listening on %s (db=%s)", cfg.Listen, cfg.DBPath)
	if err := srv.Engine().Run(cfg.Listen); err != nil {
		log.Fatal(err)
	}
}
