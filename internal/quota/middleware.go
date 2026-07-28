package quota

import (
	"strconv"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/andyeswong/freerouter-go/internal/auth"
)

// LimitsOf reads the ceilings configured on a token.
func LimitsOf(tok *auth.ApiToken) Limits {
	return Limits{
		Daily:   tok.LimitDailyTokens,
		Weekly:  tok.LimitWeeklyTokens,
		Monthly: tok.LimitMonthlyTokens,
	}
}

// Enforce rejects requests from tokens that have spent a configured window's
// budget. Mount it after RequireToken on endpoints that actually consume
// tokens (chat completions) — not on cheap metadata routes like /v1/models.
//
// The check is post-hoc by design: a request is blocked once the budget is
// already spent, so the request that crosses the line still goes through.
// Blocking mid-flight would mean predicting an unknown completion size.
func Enforce(t *Tracker) gin.HandlerFunc {
	return func(c *gin.Context) {
		tok, ok := auth.TokenFromCtx(c)
		if !ok {
			c.Next()
			return
		}
		lim := LimitsOf(tok)
		if lim.IsZero() {
			c.Next()
			return
		}
		v := t.Exceeded(tok.ID, lim)
		if v == nil {
			c.Next()
			return
		}

		now := time.Now()
		c.Header("X-FreeRouter-Quota-Window", v.Window.String())
		c.Header("X-FreeRouter-Quota-Used", strconv.FormatInt(v.Used, 10))
		c.Header("X-FreeRouter-Quota-Limit", strconv.FormatInt(v.Limit, 10))
		c.Header("X-FreeRouter-Quota-Reset", v.ResetsAt.Format(time.RFC3339))
		c.Header("Retry-After", strconv.Itoa(v.RetryAfter(now)))

		// OpenAI-shaped error so stock clients surface it as a rate limit
		// instead of an opaque failure.
		c.AbortWithStatusJSON(429, gin.H{"error": gin.H{
			"message": v.Window.String() + " token quota exceeded: " +
				strconv.FormatInt(v.Used, 10) + "/" + strconv.FormatInt(v.Limit, 10) +
				" tokens used; resets " + v.ResetsAt.Format(time.RFC3339),
			"type":  "rate_limit_exceeded",
			"code":  "quota_exceeded",
			"param": v.Window.String(),
		}})
	}
}
