package router

// computeSavings estimates cost vs the HONEST baseline: the most expensive
// enabled model in the vademécum (fixes freerouter bug #2, which hardcoded
// claude-opus pricing whether or not you configured it).
func (rt *Router) computeSavings(d *Decision, inputTokens, maxOutput int) {
	cost := func(in, out float64) float64 {
		return float64(inputTokens)/1e6*in + float64(maxOutput)/1e6*out
	}

	d.CostEstimate = cost(d.Model.InputPrice, d.Model.OutputPrice)

	base, err := rt.repo.MostExpensiveEnabled()
	if err != nil || base == nil {
		d.BaselineCost = d.CostEstimate
		d.Savings = 0
		return
	}
	d.BaselineCost = cost(base.InputPrice, base.OutputPrice)
	if d.BaselineCost > 0 {
		s := (d.BaselineCost - d.CostEstimate) / d.BaselineCost
		if s < 0 {
			s = 0
		}
		d.Savings = s
	}
}
