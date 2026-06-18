// FreeRouter-Go — data-driven, OpenAI-compatible LLM router.
// Build static: CGO_ENABLED=0 go build -o freerouter .
package main

import (
	"log"

	"github.com/glebarez/sqlite" // pure-Go sqlite driver (no CGO)
	"gorm.io/gorm"

	"github.com/andyeswong/freerouter-go/internal/config"
	"github.com/andyeswong/freerouter-go/internal/models"
	"github.com/andyeswong/freerouter-go/internal/router"
	"github.com/andyeswong/freerouter-go/internal/server"
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
	if err := repo.AutoMigrate(); err != nil {
		log.Fatalf("migrate: %v", err)
	}

	rt := router.New(repo, cfg.Heuristic)
	srv := server.New(repo, rt)

	log.Printf("FreeRouter-Go listening on %s (db=%s)", cfg.Listen, cfg.DBPath)
	if err := srv.Engine().Run(cfg.Listen); err != nil {
		log.Fatal(err)
	}
}
