package auth

import (
	"strings"

	"github.com/gin-gonic/gin"
)

const ctxTokenKey = "frgo_token"

// bearer extracts the token from an Authorization header.
func bearer(c *gin.Context) string {
	h := c.GetHeader("Authorization")
	if strings.HasPrefix(h, "Bearer ") {
		return strings.TrimSpace(h[7:])
	}
	return ""
}

// RequireToken gates consumer endpoints (/v1/*). It sets the resolved ApiToken
// in the gin context for downstream usage recording.
func (r *Repo) RequireToken() gin.HandlerFunc {
	return func(c *gin.Context) {
		plain := bearer(c)
		if plain == "" {
			c.AbortWithStatusJSON(401, gin.H{"error": "missing bearer token"})
			return
		}
		tok, ok := r.Verify(plain)
		if !ok {
			c.AbortWithStatusJSON(401, gin.H{"error": "invalid or disabled token"})
			return
		}
		c.Set(ctxTokenKey, tok)
		c.Next()
	}
}

// TokenFromCtx returns the ApiToken stored by RequireToken, if any.
func TokenFromCtx(c *gin.Context) (*ApiToken, bool) {
	v, ok := c.Get(ctxTokenKey)
	if !ok {
		return nil, false
	}
	t, ok := v.(*ApiToken)
	return t, ok
}

// RequireAdmin gates /admin/* with a single static admin token. If adminToken
// is empty, the gate is open (dev mode) — log a warning at startup if so.
func RequireAdmin(adminToken string) gin.HandlerFunc {
	return func(c *gin.Context) {
		if adminToken == "" {
			c.Next()
			return
		}
		if bearer(c) != adminToken {
			c.AbortWithStatusJSON(401, gin.H{"error": "admin token required"})
			return
		}
		c.Next()
	}
}
