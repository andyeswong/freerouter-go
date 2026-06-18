// Package providers proxies a chat request to the chosen model's
// OpenAI-compatible endpoint.
package providers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/andyeswong/freerouter-go/internal/models"
)

// Message accepts content as a string OR an array of {type,text} blocks.
// Hard-won lesson from cc_bridge (2026-06-17): OpenClaw's WhatsApp channel
// sends content as an array while Telegram sends a string; a plain `string`
// field 400s on the array. Flatten to a single string for downstream use.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

func (m *Message) UnmarshalJSON(b []byte) error {
	var raw struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}
	m.Role = raw.Role

	// Try string first.
	var s string
	if err := json.Unmarshal(raw.Content, &s); err == nil {
		m.Content = s
		return nil
	}
	// Else array of blocks.
	var blocks []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw.Content, &blocks); err != nil {
		return fmt.Errorf("content is neither string nor block array: %w", err)
	}
	var parts []string
	for _, blk := range blocks {
		if blk.Text != "" {
			parts = append(parts, blk.Text)
		}
	}
	m.Content = strings.Join(parts, "\n")
	return nil
}

// ChatRequest is the subset of the OpenAI chat schema the router inspects.
// The full original body is proxied verbatim except `model`, which is rewritten
// to the chosen upstream model id.
type ChatRequest struct {
	Model     string    `json:"model"`
	Messages  []Message `json:"messages"`
	MaxTokens int       `json:"max_tokens"`
	Stream    bool      `json:"stream"`

	// FreeRouter extensions (optional, ignored by upstream after stripping).
	Tier        int   `json:"tier,omitempty"`
	RequiresMCP *bool `json:"requires_mcp,omitempty"`
}

var httpClient = &http.Client{Timeout: 300 * time.Second}

// Proxy forwards rawBody to the model's endpoint, rewriting the model id and
// injecting the API key resolved from the model's APIKeyRef env var. Returns
// the upstream response for the handler to stream/relay.
func Proxy(m models.LlmModel, rawBody []byte) (*http.Response, error) {
	// Rewrite the "model" field to the upstream id; strip router-only fields.
	var body map[string]any
	if err := json.Unmarshal(rawBody, &body); err != nil {
		return nil, err
	}
	body["model"] = m.ModelID
	delete(body, "tier")
	delete(body, "requires_mcp")

	out, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	url := strings.TrimRight(m.APIBaseURL, "/") + "/chat/completions"
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(out))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if key := os.Getenv(m.APIKeyRef); key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	return httpClient.Do(req)
}

// ExtractPrompt returns the last user message and the concatenated system
// prompt, for the router to classify.
func ExtractPrompt(msgs []Message) (user, system string) {
	var sys []string
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == "user" && user == "" {
			user = msgs[i].Content
		}
		if msgs[i].Role == "system" {
			sys = append([]string{msgs[i].Content}, sys...)
		}
	}
	return user, strings.Join(sys, "\n")
}

// Drain is a small helper to fully read+close a response body.
func Drain(r io.ReadCloser) { _, _ = io.Copy(io.Discard, r); _ = r.Close() }
