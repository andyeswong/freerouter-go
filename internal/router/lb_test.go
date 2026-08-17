package router

import (
	"testing"

	"github.com/andyeswong/freerouter-go/internal/models"
)

func m(name, group string, tier models.Tier, cost, weight int, h models.Health) models.LlmModel {
	return models.LlmModel{Name: name, Group: group, TierMax: tier, Cost: cost, Weight: weight, Health: h}
}

// No group → the deterministic winner is returned untouched.
func TestLbPickNoGroup(t *testing.T) {
	cands := []models.LlmModel{m("a", "", 5, 3, 100, models.HealthUp), m("b", "", 5, 3, 100, models.HealthUp)}
	if got := lbPick(cands, &cands[0]); got.Name != "a" {
		t.Fatalf("want a, got %s", got.Name)
	}
}

// A lone healthy peer (the other is down) → all traffic to the survivor. This is
// the failover property: a group of 2 degrades to a single model, not to error.
func TestLbPickFailover(t *testing.T) {
	cands := []models.LlmModel{
		m("minimax", "t5", 5, 3, 100, models.HealthUp),
		m("glm", "t5", 5, 3, 100, models.HealthDown),
	}
	for i := 0; i < 50; i++ {
		if got := lbPick(cands, &cands[0]); got.Name != "minimax" {
			t.Fatalf("down peer got traffic: %s", got.Name)
		}
	}
}

// Two equal-weight healthy peers → both get a meaningful share (~50/50).
func TestLbPickSplits(t *testing.T) {
	cands := []models.LlmModel{
		m("minimax", "t5", 5, 3, 100, models.HealthUp),
		m("glm", "t5", 5, 3, 100, models.HealthUp),
	}
	count := map[string]int{}
	for i := 0; i < 4000; i++ {
		count[lbPick(cands, &cands[0]).Name]++
	}
	for _, n := range []string{"minimax", "glm"} {
		if count[n] < 1400 { // expect ~2000 each; generous floor
			t.Fatalf("%s got only %d/4000, split is broken", n, count[n])
		}
	}
}

// A peer in the same group but a DIFFERENT (tier_max,cost) bucket is not pooled:
// LB never crosses the ordering keys, so cheapest-sufficient still holds.
func TestLbPickBucketIsolation(t *testing.T) {
	cands := []models.LlmModel{
		m("cheap", "t5", 5, 3, 100, models.HealthUp),
		m("pricier", "t5", 5, 9, 100, models.HealthUp),
	}
	for i := 0; i < 50; i++ {
		if got := lbPick(cands, &cands[0]); got.Name != "cheap" {
			t.Fatalf("LB crossed cost bucket: %s", got.Name)
		}
	}
}
